// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package pollinations implements a client for the Pollinations API.
//
// It is described at https://github.com/pollinations/pollinations/blob/master/APIDOCS.md
package pollinations

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Pollinations.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Pollinations platform API.
//
// The value for ProviderOptionAPIKey can be either an API key retrieved from https://auth.pollinations.ai/ or a referrer.
// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#referrer-
//
// ProviderOptionAPIKey is optional. Providing one, either via environment variable POLLINATIONS_API_KEY, will increase quota.
//
// To use multiple models, create multiple clients.
// Models are listed at https://docs.perplexity.ai/guides/model-cards
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionAPIKey:
			apiKey = string(v)
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			modalities = genai.Modalities(v)
		case genai.ProviderOptionPreloadedModels:
			preloadedModels = []genai.Model(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	var h http.Header
	if apiKey == "" {
		apiKey = os.Getenv("POLLINATIONS_API_KEY")
	}
	if apiKey != "" {
		if strings.HasPrefix(apiKey, "http://") || strings.HasPrefix(apiKey, "https://") {
			h = http.Header{"Referrer": {apiKey}}
		} else {
			h = http.Header{"Authorization": {"Bearer " + apiKey}}
		}
	}
	// Default to text generation.
	preferText := true
	switch len(modalities) {
	case 0:
		// Auto-detect below.
	case 1:
		switch modalities[0] {
		case genai.ModalityAudio:
			return nil, fmt.Errorf("unexpected option Modalities %s, only image or text is implemented (send PR to add support)", modalities)
		case genai.ModalityImage:
			preferText = false
		case genai.ModalityText:
		case genai.ModalityDocument, genai.ModalityVideo:
			return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are supported", modalities)
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are supported", modalities)
		}
	default:
		return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are supported", modalities)
	}
	t := base.DefaultTransport
	if r, ok := t.(*roundtrippers.Retry); ok {
		// Make a copy so we can edit it.
		c := *r
		if p, ok := c.Policy.(*roundtrippers.ExponentialBackoff); ok {
			// Tweak the policy.
			c.Policy = &exponentialBackoff{ExponentialBackoff: *p}
		} else {
			return nil, fmt.Errorf("unsupported retry policy %T", c.Policy)
		}
		t = &c
	} else {
		return nil, fmt.Errorf("unsupported transport %T", t)
	}
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://text.pollinations.ai/openai",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			LieToolCalls:    true,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				Lenient: internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    h,
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	var err error
	switch model {
	case "":
	case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
		if preferText {
			if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = genai.Modalities{genai.ModalityText}
		} else {
			if c.impl.Model, err = c.selectBestImageModel(ctx); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = genai.Modalities{genai.ModalityImage}
		}
	default:
		c.impl.Model = model
		if len(modalities) == 0 {
			c.impl.OutputModalities, err = c.detectModelModalities(ctx, model)
		} else {
			c.impl.OutputModalities = modalities
		}
	}
	return c, err
}

// detectModelModalities tries its best to figure out the modality of a model
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) detectModelModalities(ctx context.Context, model string) (genai.Modalities, error) {
	mod, err := c.modelModality(ctx, model)
	return genai.Modalities{mod}, err
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	// We only list text models here, not images generation ones.
	mdls, err := c.ListTextModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*TextModel)
		if m.Audio || strings.HasSuffix(m.Name, "roblox") {
			continue
		}
		// This is meh.
		switch {
		case cheap:
			if m.Name == "openai-fast" {
				selectedModel = m.Name
			}
		case good:
			if strings.HasPrefix(m.Name, "openai") && !m.Reasoning {
				selectedModel = m.Name
			}
		default:
			if !strings.HasPrefix(m.Name, "openai") && m.Reasoning {
				selectedModel = m.Name
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestImageModel selects the most appropriate image model.
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestImageModel(ctx context.Context) (string, error) {
	mdls, err := c.ListImageGenModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically detect the model modality: %w", err)
	}
	// TODO: Figure out how to select a best model. There's literally no information to make an informed choice.
	return mdls[0].GetID(), nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "pollinations"
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
	if c.isAudio() || c.isImage() {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.genImage(ctx, &msgs[0], opts...)
	}
	if err := c.validateModality(ctx, genai.ModalityText); err != nil {
		return genai.Result{}, err
	}
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if c.isAudio() || c.isImage() {
		return base.SimulateStream(ctx, c, msgs, opts...)
	}
	if err := c.validateModality(ctx, genai.ModalityText); err != nil {
		return yieldNoFragment, func() (genai.Result, error) {
			return genai.Result{}, err
		}
	}
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// genImage is a simplified version of GenSync only for images.
//
// Use it to generate images.
//
// Default rate limit is 0.2 QPS / IP.
func (c *Client) genImage(ctx context.Context, msg *genai.Message, opts ...genai.GenOption) (genai.Result, error) {
	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#1-text-to-image-get-%EF%B8%8F
	// TODO:
	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#4-text-to-speech-get-%EF%B8%8F%EF%B8%8F
	res := genai.Result{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	if err := msg.Validate(); err != nil {
		return res, err
	}
	for i := range msg.Requests {
		if msg.Requests[i].Text == "" {
			return res, errors.New("only text can be passed as input")
		}
	}
	if err := c.validateModality(ctx, genai.ModalityImage); err != nil {
		return genai.Result{}, err
	}
	qp := url.Values{}
	qp.Add("model", c.impl.Model)
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return res, err
		}
		switch v := opt.(type) {
		case *genai.GenOptionImage:
			if v.Width != 0 {
				qp.Add("width", strconv.Itoa(v.Width))
			}
			if v.Height != 0 {
				qp.Add("height", strconv.Itoa(v.Height))
			}
		case *genai.GenOptionText:
			// TODO: Deny most flags.
		case genai.GenOptionSeed:
			// Defaults to 42 otherwise.
			qp.Add("seed", strconv.FormatInt(int64(v), 10))
		default:
			return genai.Result{}, &base.ErrNotSupported{Options: []string{internal.TypeName(opt)}}
		}
	}

	qp.Add("nologo", "true")
	qp.Add("private", "true") // "nofeed"
	qp.Add("enhance", "false")
	qp.Add("safe", "false")
	// Other supported options: negative_prompt.
	qp.Add("quality", "medium")
	for _, mc := range msg.Requests {
		if mc.Doc.Src != nil {
			return res, errors.New("inline document is not supported")
		}
		if mc.Doc.URL != "" {
			qp.Add("image", mc.Doc.URL)
		}
	}

	prompt := url.QueryEscape(msg.String())
	u := "https://image.pollinations.ai/prompt/" + url.PathEscape(prompt) + "?" + qp.Encode()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, http.NoBody)
	if err != nil {
		return res, err
	}
	resp, err := c.impl.Client.Do(req)
	if err != nil {
		return res, err
	}
	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		return res, c.impl.DecodeError(u, resp)
	}
	b, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil {
		return res, err
	}
	res.Replies = []genai.Reply{{Doc: genai.Doc{Src: &bb.BytesBuffer{D: b}}}}
	if ct := resp.Header.Get("Content-Type"); strings.HasPrefix(ct, "image/jpeg") {
		res.Replies[0].Doc.Filename = "content.jpg"
	} else {
		return res, fmt.Errorf("unknown Content-Type: %s", ct)
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
	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#list-available-image-models-
	img, err1 := c.ListImageGenModels(ctx)
	txt, err2 := c.ListTextModels(ctx)
	out := make([]genai.Model, 0, len(img)+len(txt))
	out = append(out, img...)
	out = append(out, txt...)
	if err1 == nil {
		err1 = err2
	}
	return out, err1
}

// ListImageGenModels lists available image generation models.
func (c *Client) ListImageGenModels(ctx context.Context) ([]genai.Model, error) {
	var resp ImageModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://enter.pollinations.ai/api/generate/image/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ListTextModels lists available text models.
func (c *Client) ListTextModels(ctx context.Context) ([]genai.Model, error) {
	var resp TextModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://enter.pollinations.ai/api/generate/text/models", nil, &resp); err != nil {
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

// modelModality returns the modality of model if found.
func (c *Client) modelModality(ctx context.Context, model string) (genai.Modality, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", err
	}
	for _, mdl := range mdls {
		if mdl.GetID() == model {
			if _, ok := mdl.(*TextModel); ok {
				return genai.ModalityText, nil
			}
			if _, ok := mdl.(*ImageModel); ok {
				return genai.ModalityImage, nil
			}
			break
		}
	}
	return "", fmt.Errorf("model %q not supported by pollinations", model)
}

// validateModality returns nil if the modality is supported by the model.
func (c *Client) validateModality(ctx context.Context, mod genai.Modality) error {
	if got, err := c.modelModality(ctx, c.ModelID()); err != nil {
		return err
	} else if got != mod {
		return fmt.Errorf("modality %s not supported", mod)
	}
	return nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if pkt.Usage.PromptTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				}
				if len(pkt.Choices) != 1 {
					continue
				}
				if fr := pkt.Choices[0].FinishReason; fr != "" {
					u.FinishReason = fr.ToFinishReason()
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt)}
					return
				}
				f := genai.Reply{
					Text:      pkt.Choices[0].Delta.Content,
					Reasoning: pkt.Choices[0].Delta.ReasoningContent,
				}
				// Pollinations streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
				if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
					if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID != "" {
						// A new call.
						if pendingToolCall.ID == "" {
							pendingToolCall = t
							if !f.IsZero() {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
								return
							}
							continue
						}
						// Flush.
						pendingToolCall.To(&f.ToolCall)
						pendingToolCall = t
					} else if pendingToolCall.ID != "" {
						// Continuation.
						pendingToolCall.Function.Arguments += t.Function.Arguments
						if !f.IsZero() {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
							return
						}
						continue
					}
				} else if pendingToolCall.ID != "" {
					// Flush.
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				}
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

type exponentialBackoff struct {
	roundtrippers.ExponentialBackoff
}

func (e *exponentialBackoff) ShouldRetry(ctx context.Context, start time.Time, try int, err error, resp *http.Response) bool {
	if resp != nil && resp.StatusCode == http.StatusPaymentRequired {
		return true
	}
	return e.ExponentialBackoff.ShouldRetry(ctx, start, try, err, resp)
}

func yieldNoFragment(yield func(genai.Reply) bool) {
}

var _ genai.Provider = &Client{}
