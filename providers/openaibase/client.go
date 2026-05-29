// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Shared OpenAI-compatible client operations and DTOs.

package openaibase

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/bb"
)

// modelDateSuffixRE matches dated model IDs like "gpt-5.5-2026-04-23" so we can
// prefer their undated aliases (e.g. "gpt-5.5") when selecting the best model.
var modelDateSuffixRE = regexp.MustCompile(`-\d{4}-\d{2}-\d{2}$`)

// Client holds the shared state and methods common to both the OpenAI Chat Completion and Responses API
// providers.
//
// Both openaichat.Client and openairesponses.Client embed this struct and delegate shared operations to it.
type Client struct {
	// Impl is a pointer to the ProviderBase inside the provider-specific base.Provider.
	Impl *base.ProviderBase[*ErrorResponse]
	// BaseURL is the base URL for the OpenAI API, e.g. "https://api.openai.com/v1".
	BaseURL string
	// PreloadedModels is an optional pre-supplied model list to avoid HTTP round-trips.
	PreloadedModels []genai.Model
}

// ListModels returns the list of available models.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.PreloadedModels != nil {
		return c.PreloadedModels, nil
	}
	// https://platform.openai.com/docs/api-reference/models/list
	var resp ModelsResponse
	if err := c.Impl.DoRequest(ctx, "GET", c.BaseURL+"/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// GenDoc generates an image document from a single message.
func (c *Client) GenDoc(ctx context.Context, msg *genai.Message, opts ...genai.GenOption) (genai.Result, error) {
	// https://platform.openai.com/docs/api-reference/images/create
	res := genai.Result{}
	if err := c.Impl.Validate(); err != nil {
		return res, err
	}
	req := ImageRequest{}
	if err := req.Init(msg, c.Impl.Model, opts...); err != nil {
		return res, err
	}
	u := c.BaseURL + "/images/generations"

	// It is very different because it requires a multi-part upload.
	// https://platform.openai.com/docs/api-reference/images/createEdit
	// url = "https://api.openai.com/v1/images/edits"

	resp := ImageResponse{}
	if err := c.Impl.DoRequest(ctx, "POST", u, &req, &resp); err != nil {
		return res, err
	}
	res.Replies = make([]genai.Reply, len(resp.Data))
	for i := range resp.Data {
		n := "content.jpg"
		if len(resp.Data) > 1 {
			n = fmt.Sprintf("content%d.jpg", i+1)
		}
		if u := resp.Data[i].URL; u != "" {
			res.Replies[i].Doc = genai.Doc{Filename: n, URL: u}
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

// FileAdd uploads a file. The TTL is one month.
func (c *Client) FileAdd(ctx context.Context, filename string, r io.ReadSeeker) (string, error) {
	// https://platform.openai.com/docs/api-reference/files/create
	buf := bytes.Buffer{}
	w := multipart.NewWriter(&buf)
	// We don't need this to be random, and setting it to be deterministic makes HTTP playback possible.
	_ = w.SetBoundary("80309819a837f26826233a299e185d0ccf3f559362092bd3278b8a045ee1")
	if err := w.WriteField("purpose", "batch"); err != nil {
		return "", err
	}
	part, err := w.CreateFormFile("file", filename)
	if err != nil {
		return "", err
	}
	if _, err = io.Copy(part, r); err != nil {
		return "", err
	}
	if err := w.Close(); err != nil {
		return "", err
	}
	u := c.BaseURL + "/files"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, u, &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	resp, err := c.Impl.Client.Do(req)
	if err != nil {
		if resp != nil {
			_ = resp.Body.Close()
		}
		return "", err
	}
	f := File{}
	err = c.Impl.DecodeResponse(resp, u, &f)
	return f.ID, err
}

// FileGet retrieves a file's contents.
func (c *Client) FileGet(ctx context.Context, id string) (io.ReadCloser, error) {
	// https://platform.openai.com/docs/api-reference/files/retrieve-contents
	u := c.BaseURL + "/files/" + url.PathEscape(id) + "/content"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, http.NoBody)
	if err != nil {
		return nil, err
	}
	resp, err := c.Impl.Client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, c.Impl.DecodeError(u, resp)
	}
	return resp.Body, nil
}

// FileDel deletes a file.
func (c *Client) FileDel(ctx context.Context, id string) error {
	// https://platform.openai.com/docs/api-reference/files/delete
	u := c.BaseURL + "/files/" + url.PathEscape(id)
	out := FileDeleteResponse{}
	return c.Impl.DoRequest(ctx, "DELETE", u, nil, &out)
}

// FilesListRaw lists files.
func (c *Client) FilesListRaw(ctx context.Context) ([]File, error) {
	// TODO: Pagination. It defaults at 10000 items per page.
	resp := FileListResponse{}
	err := c.Impl.DoRequest(ctx, "GET", c.BaseURL+"/files", nil, &resp)
	return resp.Data, err
}

// DetectModelModalities tries its best to figure out the modality of a model.
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) DetectModelModalities(ctx context.Context, model string) (genai.Modalities, error) { //nolint:unparam // ctx for future use.
	// Damn you OpenAI! This is super fragile.
	// TODO: Fill out the scoreboard from https://platform.openai.com/docs/models then use that. This is sad
	// because ListModels is useless. See
	// https://discord.com/channels/974519864045756446/1070006915414900886/threads/1408629226864775265 for the
	// author's feature request.
	if strings.HasPrefix(model, "dall") || strings.Contains(model, "image") {
		return genai.Modalities{genai.ModalityImage}, nil
	} else if strings.Contains(model, "audio") {
		return genai.Modalities{genai.ModalityAudio, genai.ModalityText}, nil
	}
	return genai.Modalities{genai.ModalityText}, nil
}

// SelectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
func (c *Client) SelectBestTextModel(ctx context.Context, preference string) (string, error) {
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
		// Prefer undated aliases (e.g. "gpt-5.5") over dated versions
		// (e.g. "gpt-5.5-2026-04-23").
		if modelDateSuffixRE.MatchString(m.ID) {
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
			// Pro variants are significantly more expensive; skip them for
			// the SOTA tier.
			if strings.HasPrefix(m.ID, "gpt") && !strings.Contains(m.ID, "-nano") && !strings.Contains(m.ID, "-mini") && !strings.Contains(m.ID, "-pro") && (created == 0 || m.Created > created) {
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

// SelectBestImageModel selects the most appropriate image model based on the preference (cheap, good, or SOTA).
func (c *Client) SelectBestImageModel(ctx context.Context, preference string) (string, error) {
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
		// Prefer undated aliases (e.g. "gpt-image-2") over dated
		// versions (e.g. "gpt-image-2-2026-04-21").
		if modelDateSuffixRE.MatchString(m.ID) {
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

// SelectBestVideoModel selects the most appropriate video model based on the preference (cheap, good, or SOTA).
//
// Video models are identified by the "sora-" prefix.
func (c *Client) SelectBestVideoModel(ctx context.Context, preference string) (string, error) {
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

// IsAudio returns true if the output modality is audio.
func (c *Client) IsAudio() bool {
	return slices.Contains(c.Impl.OutputModalities, genai.ModalityAudio)
}

// IsImage returns true if the output modality is image.
func (c *Client) IsImage() bool {
	return slices.Contains(c.Impl.OutputModalities, genai.ModalityImage)
}

// IsVideo returns true if the output modality is video.
func (c *Client) IsVideo() bool {
	return slices.Contains(c.Impl.OutputModalities, genai.ModalityVideo)
}

// ProcessHeaders extracts rate limit information from OpenAI HTTP response headers.
func ProcessHeaders(h http.Header) []genai.RateLimit {
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
