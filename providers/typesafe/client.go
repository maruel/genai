// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package typesafe implements a client for the TypeSafe System One API, to use Jev.
//
// It is described at https://docs.typesafe.ai/api
//
// TypeSafe is unlike the other providers: it does not generate text. Every request evaluates one state
// with a set of typed questions about it, and returns one typed answer per question. The questions are
// declared as fields of a struct of Noul, Choice and Score, passed with
// genai.GenOptionText.DecodeAs, and the same struct receives the answers.
//
// The state is the single message passed: its text, or a JSON document for structured data, like the
// record the questions are about; several requests become an array state. The API evaluates a string as
// text, so it never parses JSON passed as one.
//
// QuestionsFrom returns the questions a struct declares, to review them, to pass them back with
// DecodeAs, or to call GenSyncRaw directly. With a Questions, the answers decode into Answers, keyed by
// question name.
//
// There is no multi-turn conversation: pass the whole context as the state.
package typesafe

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"slices"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for TypeSafe.
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
	impl            base.ProviderBase[*ErrorResponse]
	remote          string
	preloadedModels []genai.Model
}

// New creates a new client to talk to the TypeSafe API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the TYPESAFE_API_KEY environment
// variable. If not found, it will still return a client
// coupled with a base.ErrAPIKeyRequired error. Get your API key at
// https://console.typesafe.ai/settings/keys
//
// ProviderOptionModel accepts a model ID or alias, e.g. "jev-latest", "jev-preview" or "jev-1.13.0".
// ModelCheap, ModelGood and ModelSOTA all resolve to "jev-latest", TypeSafe serving a single model.
//
// ProviderOptionRemote defaults to "https://api.typesafe.ai".
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
	var preloadedModels []genai.Model
	remote := "https://api.typesafe.ai"
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
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			if mod := genai.Modalities(v); len(mod) != 0 && !slices.Equal(mod, genai.Modalities{genai.ModalityText}) {
				return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
			}
		case genai.ProviderOptionPreloadedModels:
			preloadedModels = []genai.Model(v)
		case genai.ProviderOptionRemote:
			remote = string(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://console.typesafe.ai/settings/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("TYPESAFE_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "TYPESAFE_API_KEY", URL: apiKeyURL}
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		remote:          remote,
		preloadedModels: preloadedModels,
		impl: base.ProviderBase[*ErrorResponse]{
			APIKeyURL: apiKeyURL,
			Lenient:   internal.BeLenient,
			Client: http.Client{
				Transport: &roundtrippers.Header{
					Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
					Transport: &roundtrippers.RequestID{Transport: t},
				},
			},
		},
	}
	if err == nil {
		c.impl.OutputModalities = genai.Modalities{genai.ModalityText}
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			c.impl.Model = c.selectBestModel(model)
		default:
			c.impl.Model = model
		}
	}
	return c, err
}

// selectBestModel selects the model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break
// one day or another.
func (c *Client) selectBestModel(preference string) string {
	// TypeSafe serves a single model behind aliases, so every preference resolves to the alias that
	// tracks the latest release. There is nothing to choose between.
	return "jev-latest"
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "typesafe"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate. TypeSafe only
// generates the typed answers, which are returned as JSON text.
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to make requests with.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
//
// The message is the state: its text, or a JSON document. The questions to ask are declared with
// genai.GenOptionText.DecodeAs, a pointer to a struct of Noul, Choice and Score fields, which then holds
// the answers.
//
// The reply is the JSON object of the answers keyed by question name, as returned by the API. Decode it
// with Result.Decode into the same struct, or into an Answers.
//
// The versioned model that answered, which can differ from the requested alias, is only reported by
// GenSyncRaw.
//
// Recommended reading: https://docs.typesafe.ai/concepts/state
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	if err := base.CheckDuplicateGenOptions(opts); err != nil {
		return genai.Result{}, err
	}
	res := genai.Result{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	if len(msgs) != 1 {
		return res, errors.New("must pass exactly one message")
	}
	req := &SystemOneRequest{Model: c.impl.Model}
	if err := req.From(&msgs[0]); err != nil {
		return res, err
	}
	if err := req.FromOptions(opts...); err != nil {
		return res, err
	}
	resp := &SystemOneResponse{}
	if err := c.GenSyncRaw(ctx, req, resp); err != nil {
		return res, err
	}
	res, err := resp.ToResult()
	if err != nil {
		return res, &internal.BadError{Err: err}
	}
	for name := range req.Questions {
		if _, ok := resp.Answers[name]; !ok {
			return res, &internal.BadError{Err: fmt.Errorf("no answer returned for question %q", name)}
		}
	}
	if err := res.Validate(); err != nil {
		return res, &internal.BadError{Err: err}
	}
	return res, nil
}

// GenStream implements genai.Provider.
//
// TypeSafe has no streaming API, so the whole reply is simulated from GenSync and yielded at once.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return base.SimulateStream(ctx, c, msgs, opts...)
}

// GenSyncRaw runs a System One request with the raw API types.
func (c *Client) GenSyncRaw(ctx context.Context, in *SystemOneRequest, out *SystemOneResponse) error {
	// https://docs.typesafe.ai/api
	return c.impl.DoRequest(ctx, "POST", c.remote+"/v1/systemone", in, out)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if len(c.preloadedModels) != 0 {
		return c.preloadedModels, nil
	}
	models, err := c.ListModelsRaw(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]genai.Model, len(models))
	for i := range models {
		out[i] = &models[i]
	}
	return out, nil
}

// ListModelsRaw runs a model list request with the raw API types.
func (c *Client) ListModelsRaw(ctx context.Context) ([]Model, error) {
	// https://docs.typesafe.ai/models
	out := &ListModelsResponse{}
	if err := c.impl.DoRequest(ctx, "GET", c.remote+"/v1/models", nil, out); err != nil {
		return nil, err
	}
	return out.Models, nil
}
