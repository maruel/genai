// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package typesafe implements a client for the TypeSafe System One API, to use Jev.
//
// It is described at https://docs.typesafe.ai/api
//
// # Decision inference
//
// TypeSafe does not generate text. Every request evaluates one state
// with a set of typed questions about it, and returns one typed answer per question. Set the named
// questions in genai.Questions and read the same names from SystemOneResponse.Answers.
//
// SystemOne accepts text or JSON state and typed questions. The API evaluates a string as text,
// so it never parses JSON passed as one. GenSync and GenStream are not supported.
//
// There is no multi-turn conversation: pass the whole context as the state. Images, audio and video
// are not supported. [SystemOneRequest.From] converts shared decision input and rejects attachments.
// [SystemOneRequest.Validate] owns provider constraints. [Client.SystemOneRaw] exposes native requests.
// See [New] for authentication, model aliases and endpoint configuration.
package typesafe

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
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
			if mod := genai.Modalities(v); len(mod) != 0 && !slices.Equal(mod, genai.Modalities{genai.ModalityDecision}) {
				return nil, fmt.Errorf("unexpected option Modalities %s, only decision is supported", mod)
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
		c.impl.OutputModalities = genai.Modalities{genai.ModalityDecision}
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

// Close implements io.Closer. It currently does nothing.
func (c *Client) Close() error {
	return nil
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
// returns typed decision answers.
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

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{SystemOne: true}
}

// SystemOne implements genai.Provider using TypeSafe's typed decision API.
func (c *Client) SystemOne(ctx context.Context, in *genai.SystemOneRequest) (*genai.SystemOneResponse, error) {
	req := SystemOneRequest{Model: c.impl.Model}
	if err := req.From(in); err != nil {
		return nil, err
	}
	if req.Model == "" {
		return nil, errors.New("a model is required")
	}
	out := SystemOneResponse{}
	if err := c.SystemOneRaw(ctx, &req, &out); err != nil {
		return nil, err
	}
	res := &genai.SystemOneResponse{}
	if err := out.To(res); err != nil {
		return nil, err
	}
	if err := res.ValidateQuestions(&req.Questions); err != nil {
		return nil, &internal.BadError{Err: err}
	}
	return res, nil
}

// SystemOneRaw runs a System One request with the raw API types.
func (c *Client) SystemOneRaw(ctx context.Context, in *SystemOneRequest, out *SystemOneResponse) error {
	if err := in.Validate(); err != nil {
		return err
	}
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
