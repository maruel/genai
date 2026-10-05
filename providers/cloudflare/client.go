// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cloudflare implements a client for the Cloudflare AI API.
//
// It is described at https://developers.cloudflare.com/api/resources/ai/
//
// # Decision inference
//
// CLEF and CLEF-Flash use [Client.SystemOne] instead of chat. Select an explicit Workers AI model,
// such as "@cf/cloudflare/clef" or "@cf/cloudflare/clef-flash", through genai.ProviderOptionModel.
// SystemOne uses the configured model's full route and its final path component as the body selector.
//
// Requests contain text or JSON state, typed questions and optional inline PNG, JPEG or WebP images.
// Video and decision streaming are not supported. GenSync and GenStream remain chat operations.
// [SystemOneRequest.From] bounds document reads; [SystemOneRequest.Validate] owns question and image
// limits. These checks are best-effort; the service is authoritative and its limits may change.
// The server enforces the encoded request size limit.
//
// [Client.SystemOneRaw] accepts a full Workers AI ID or a short selector under @cf/cloudflare,
// independently of the configured model. Accepting future model IDs does not establish their schema or media support.
package cloudflare

// See official client at https://github.com/cloudflare/cloudflare-go

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// AccountID provides an account ID for Cloudflare Workers AI.
//
// Get your account ID at https://dash.cloudflare.com/profile/api-tokens
type AccountID string

// Validate implements genai.Validatable.
func (a AccountID) Validate() error {
	if a == "" {
		return errors.New("cloudflare.AccountID cannot be empty")
	}
	return nil
}

// Scoreboard for Cloudflare.
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
	impl      base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
	accountID string
}

// New creates a new client to talk to the Cloudflare Workers AI platform API.
//
// If AccountID is not provided, it tries to load it from the CLOUDFLARE_ACCOUNT_ID environment variable.
// If ProviderOptionAPIKey is not provided, it tries to load it from the CLOUDFLARE_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your account ID and API key at https://dash.cloudflare.com/profile/api-tokens
//
// To use multiple models, create multiple clients.
// Use one of the model from https://developers.cloudflare.com/workers-ai/models/
//
// SystemOne requests use CLEF's current question and media limits.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, accountID, model string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	if err := base.CheckDuplicateProviderOptions(opts); err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionAPIKey:
			apiKey = string(v)
		case AccountID:
			accountID = string(v)
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
	const apiKeyURL = "https://dash.cloudflare.com/profile/api-tokens"
	var err error
	if accountID == "" {
		if accountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID"); accountID == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "CLOUDFLARE_ACCOUNT_ID", URL: apiKeyURL}
		}
	}
	if apiKey == "" {
		if apiKey = os.Getenv("CLOUDFLARE_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "CLOUDFLARE_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if strings.HasPrefix(model[strings.LastIndexByte(model, '/')+1:], "clef") {
		mod = genai.Modalities{genai.ModalityDecision}
	}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		// TODO: Cloudflare supports other output modalities but they are not currently implemented.
		// https://developers.cloudflare.com/workers-ai/models/?tasks=Text-to-Image
		if model != "" || slices.ContainsFunc(modalities, func(m genai.Modality) bool { return m != genai.ModalityText && m != genai.ModalityDecision }) {
			return nil, fmt.Errorf("unexpected option Modalities %s, model supports %s", modalities, mod)
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	// Investigate websockets?
	// https://blog.cloudflare.com/workers-ai-streaming/ and
	// https://developers.cloudflare.com/workers/examples/websockets/
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
		accountID: accountID,
	}
	if err == nil {
		switch model {
		case "":
			c.impl.OutputModalities = modalities
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
				return nil, err
			}
			// Important: the model must not be path escaped!
			c.impl.GenSyncURL = "https://api.cloudflare.com/client/v4/accounts/" + url.PathEscape(accountID) + "/ai/run/" + c.impl.Model
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.GenSyncURL = "https://api.cloudflare.com/client/v4/accounts/" + url.PathEscape(accountID) + "/ai/run/" + c.impl.Model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
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
	price := 100000.
	if !cheap {
		price = 0.
	}
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if strings.Contains(m.Name, "guard") || strings.HasPrefix(m.Name, "@cf/meta/llama-2") {
			// llama-guard is not a generation model.
			// @cf/meta/llama-2-7b-chat-fp16 is super expensive.
			continue
		}
		_, out := m.Price()
		if out == 0 {
			continue
		}
		switch {
		case cheap:
			if strings.HasPrefix(m.Name, "@cf/meta/") && out < price {
				price = out
				selectedModel = m.Name
			}
		case good:
			if strings.HasPrefix(m.Name, "@cf/meta/") && out > price {
				price = out
				selectedModel = m.Name
			}
		default:
			if strings.HasPrefix(m.Name, "@cf/deepseek-ai/") && out > price {
				price = out
				selectedModel = m.Name
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// Close implements io.Closer. It currently does nothing.
func (c *Client) Close() error {
	return nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "cloudflare"
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
	return c.impl.GenSync(ctx, msgs, opts...)
}

// SystemOneRaw runs a hosted decision request using the CLEF-compatible schema.
// Model accepts a full Workers AI ID or a short @cf/cloudflare selector,
// independently of the client's configured model.
func (c *Client) SystemOneRaw(ctx context.Context, in *SystemOneRequest, out *SystemOneResponse) error {
	if err := in.Validate(); err != nil {
		return err
	}
	// The hosted schema permits surrounding whitespace in the body selector.
	req := *in
	model := strings.TrimSpace(in.Model)
	var u string
	if strings.HasPrefix(model, "@") && strings.Contains(model, "/") {
		u = systemOneURL(c.accountID, strings.Split(model, "/")...)
		req.Model = model[strings.LastIndex(model, "/")+1:]
	} else {
		u = systemOneURL(c.accountID, "@cf", "cloudflare", model)
		req.Model = model
	}
	// Decode the decision envelope ourselves: the chat error fallback cannot
	// interpret a populated decision result when schema validation fails.
	var raw json.RawMessage
	if err := c.impl.DoRequest(ctx, "POST", u, &req, &raw); err != nil {
		return err
	}
	resp := SystemOneResponse{}
	if err := internal.UnmarshalJSON(raw, &resp); err != nil {
		if api, ok := errors.AsType[*ErrorResponse](err); ok {
			return api
		}
		return &internal.BadError{Err: err}
	}
	if err := resp.validateQuestions(&in.Questions); err != nil {
		return &internal.BadError{Err: err}
	}
	*out = resp
	return nil
}

// systemOneURL escapes route components without encoding a configured model's separators.
func systemOneURL(accountID string, parts ...string) string {
	for i, p := range parts {
		parts[i] = url.PathEscape(p)
		if p == "." || p == ".." {
			parts[i] = strings.Repeat("%2E", len(p))
		}
	}
	return "https://api.cloudflare.com/client/v4/accounts/" + url.PathEscape(accountID) + "/ai/run/" + strings.Join(parts, "/")
}

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{SystemOne: true}
}

// SystemOne implements genai.Provider using Workers AI's hosted decision API.
// The configured model is a full Workers AI ID or a short @cf/cloudflare selector.
// Docs accepts inline PNG, JPEG and WebP images. Other document types are rejected.
// Each image read is bounded to 4 MiB; the hosted request validator checks formats,
// dimensions and total size.
func (c *Client) SystemOne(ctx context.Context, in *genai.SystemOneRequest) (*genai.SystemOneResponse, error) {
	switch c.impl.Model {
	case "", string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
		return nil, errors.New("SystemOne requires an explicit Workers AI model, not a chat tier alias")
	}
	req := SystemOneRequest{Model: c.impl.Model}
	if err := req.From(in); err != nil {
		return nil, err
	}
	out := SystemOneResponse{}
	if err := c.SystemOneRaw(ctx, &req, &out); err != nil {
		return nil, err
	}
	res := &genai.SystemOneResponse{}
	if err := out.To(res); err != nil {
		return nil, err
	}
	return res, nil
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://developers.cloudflare.com/api/resources/ai/subresources/models/methods/list/
	var models []genai.Model
	for page := 1; ; page++ {
		out := ModelsResponse{}
		// Cloudflare's pagination is surprisingly brittle.
		u := fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/models/search?page=%d&per_page=100&hide_experimental=false", url.PathEscape(c.accountID), page)
		err := c.impl.DoRequest(ctx, "GET", u, nil, &out)
		if err != nil {
			return nil, err
		}
		for i := range out.Result {
			models = append(models, &out.Result[i])
		}
		if len(models) >= int(out.ResultInfo.TotalCount) || len(out.Result) == 0 {
			break
		}
	}
	return models, nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if pkt.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetail.CachedTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
					// Cloudflare doesn't provide FinishReason.
				}
				// TODO: Tools.
				if !yield(genai.Reply{Text: string(pkt.Response)}) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

var _ genai.Provider = &Client{}
