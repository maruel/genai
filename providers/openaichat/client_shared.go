// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// This file contains all the structures and code that is exactly the same between the OpenAI chat and responses APIs.

package openaichat

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
)

// detectModelModalities tries its best to figure out the modality of a model
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) detectModelModalities(ctx context.Context, model string) (genai.Modalities, error) { //nolint:unparam // ctx for future use.
	// Damn you OpenAI! This is super fragile.
	// TODO: Fill out the scoreboard from https://platform.openai.com/docs/models then use that. This is sad
	// because ListModels is useless. See
	// https://discord.com/channels/974519864045756446/1070006915414900886/threads/1408629226864775265 for the
	// author's feature request.
	if strings.HasPrefix(model, "dall") || strings.Contains(model, "image") {
		return genai.Modalities{genai.ModalityImage}, nil
		// TODO } else if strings.Contains(model, "audio") {
		//	return genai.Modalities{genai.ModalityAudio, genai.ModalityText}, nil
	}
	return genai.Modalities{genai.ModalityText}, nil
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	var created base.Time
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Do not select the specialized models automatically.
		if strings.Contains(m.ID, "audio") || strings.Contains(m.ID, "codex") || strings.Contains(m.ID, "deep") || strings.Contains(m.ID, "image") || strings.Contains(m.ID, "realtime") || strings.Contains(m.ID, "search") {
			continue
		}
		// The o family of models is not usable with completion API.
		if strings.HasPrefix(m.ID, "o") {
			continue
		}
		switch {
		case cheap:
			if strings.HasPrefix(m.ID, "gpt") && strings.HasSuffix(m.ID, "-nano") && (created == 0 || m.Created > created) {
				created = m.Created
				selectedModel = m.ID
			}
		case good:
			if strings.HasPrefix(m.ID, "gpt") && strings.HasSuffix(m.ID, "-mini") && (created == 0 || m.Created > created) {
				created = m.Created
				selectedModel = m.ID
			}
		default:
			if strings.HasPrefix(m.ID, "gpt") && !strings.Contains(m.ID, "-nano") && !strings.Contains(m.ID, "-mini") && (created == 0 || m.Created > created) {
				created = m.Created
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestImageModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestImageModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood)
	selectedModel := ""
	var created base.Time
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// OpenAI doesn't report much for each model. :(
		isCheap := strings.HasPrefix(m.ID, "dall")
		isGood := strings.HasSuffix(m.ID, "-mini")
		isSOTA := strings.HasPrefix(m.ID, "gpt-image")
		if !isCheap && !isSOTA {
			continue
		}
		switch {
		case cheap:
			// Use lexicographic ordering for dall-e models since dall-e-2 has a higher
			// Created timestamp than dall-e-3 due to an API data quirk.
			if isCheap {
				if selectedModel == "" || m.ID > selectedModel {
					selectedModel = m.ID
				}
			}
		case good:
			if isGood && (created == 0 || m.Created > created) {
				created = m.Created
				selectedModel = m.ID
			}
		default:
			if isSOTA && !isGood && (created == 0 || m.Created > created) {
				created = m.Created
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestVideoModel selects the most appropriate video model based on the preference (cheap, good, or SOTA).
//
// Video models are identified by the "sora-" prefix.
func (c *Client) selectBestVideoModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Only consider models starting with "sora-"
		if !strings.HasPrefix(m.ID, "sora-") {
			continue
		}
		// Determine if model is pro based on name
		isPro := strings.Contains(m.ID, "pro")
		matches := false
		switch {
		case cheap:
			matches = !isPro
		case good:
			matches = !isPro
		default:
			matches = isPro
		}
		if !matches {
			continue
		}
		// Select the best available model, preferring newer versions lexicographically
		if selectedModel == "" || m.ID > selectedModel {
			selectedModel = m.ID
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a video model automatically")
	}
	return selectedModel, nil
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to fetch results (e.g. videos) generated by the provider.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	if c.isAudio() || c.isImage() || c.isVideo() {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.genDoc(ctx, &msgs[0], opts...)
	}
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if c.isAudio() || c.isImage() || c.isVideo() {
		return base.SimulateStream(ctx, c, msgs, opts...)
	}
	return c.impl.GenStream(ctx, msgs, opts...)
}

// genDoc is a simplified version of GenSync.
func (c *Client) genDoc(ctx context.Context, msg *genai.Message, opts ...genai.GenOption) (genai.Result, error) {
	// https://platform.openai.com/docs/api-reference/images/create
	res := genai.Result{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	req := ImageRequest{}
	if err := req.Init(msg, c.impl.Model, opts...); err != nil {
		return res, err
	}
	url := "https://api.openai.com/v1/images/generations"

	// It is very different because it requires a multi-part upload.
	// https://platform.openai.com/docs/api-reference/images/createEdit
	// url = "https://api.openai.com/v1/images/edits"

	resp := ImageResponse{}
	if err := c.impl.DoRequest(ctx, "POST", url, &req, &resp); err != nil {
		return res, err
	}
	res.Replies = make([]genai.Reply, len(resp.Data))
	for i := range resp.Data {
		n := "content.jpg"
		if len(resp.Data) > 1 {
			n = fmt.Sprintf("content%d.jpg", i+1)
		}
		if url := resp.Data[i].URL; url != "" {
			res.Replies[i].Doc = genai.Doc{Filename: n, URL: url}
		} else if d := resp.Data[i].B64JSON; len(d) != 0 {
			res.Replies[i].Doc = genai.Doc{Filename: n, Src: &bb.BytesBuffer{D: resp.Data[i].B64JSON}}
		} else {
			return res, errors.New("internal error")
		}
	}
	if err := res.Validate(); err != nil {
		return res, err
	}
	return res, nil
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://platform.openai.com/docs/api-reference/models/list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.openai.com/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

func (c *Client) isAudio() bool {
	return slices.Contains(c.impl.OutputModalities, genai.ModalityAudio)
}

func (c *Client) isImage() bool {
	return slices.Contains(c.impl.OutputModalities, genai.ModalityImage)
}

func (c *Client) isVideo() bool {
	return slices.Contains(c.impl.OutputModalities, genai.ModalityVideo)
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests"), 10, 64)
	requestsReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Requests"))

	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)
	tokensReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Tokens"))

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(requestsReset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(tokensReset).Round(10 * time.Millisecond),
		})
	}
	return limits
}
