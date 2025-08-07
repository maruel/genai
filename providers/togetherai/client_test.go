// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package togetherai_test

import (
	"context"
	_ "embed"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/togetherai"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
	"github.com/maruel/httpjson"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("TOGETHER_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := togetherai.New(&genai.OptionsProvider{APIKey: apiKey, Model: model.Model}, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
		t.Fatal("implement me")
	}
	if strings.HasPrefix(model.Model, "black-forest-labs/") {
		return &smallImageGen{hideHTTP500{c}}
	}
	// If anyone at Together.AI reads this, please get your shit together.
	return &hideHTTP500{c}
}

type smallImageGen struct {
	hideHTTP500
}

func (o *smallImageGen) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("disabled to save on performance")
}

func (o *smallImageGen) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("disabled to save on performance")
}

func (o *smallImageGen) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	if v, ok := opts.(*genai.OptionsImage); ok {
		// Ask for a smaller size.
		n := *v
		n.Width = 256
		n.Height = 256
		opts = &n
	}
	return o.hideHTTP500.GenDoc(ctx, msg, opts)
}

type hideHTTP500 struct {
	*togetherai.Client
}

func (h *hideHTTP500) Unwrap() genai.Provider {
	return h.Client
}

func (h *hideHTTP500) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenSync(ctx, msgs, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; together.ai throws HTTP 500 on unsupported file formats.
			return resp, errors.New("together.ai is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

func (h *hideHTTP500) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenStream(ctx, msgs, chunks, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; together.ai throws HTTP 500 on unsupported file formats.
			return resp, errors.New("together.ai is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

func (h *hideHTTP500) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenDoc(ctx, msg, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; together.ai throws HTTP 500 on unsupported file formats.
			return resp, errors.New("together.ai is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

func TestClient_Scoreboard(t *testing.T) {
	genaiModels, err := getClient(t, "").ListModels(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var models []scoreboardtest.Model
	for _, m := range genaiModels {
		models = append(models, scoreboardtest.Model{Model: m.GetID()})
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"},
		{base.PreferredGood, "Qwen/Qwen2.5-7B-Instruct-Turbo"},
		{base.PreferredSOTA, "Qwen/Qwen3-235B-A22B-Thinking-2507"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			if got := getClient(t, line.name).ModelID(); got != line.want {
				t.Fatalf("got model %q, want %q", got, line.want)
			}
		})
	}
}

func TestClient_Provider_errors(t *testing.T) {
	data := []internaltest.ProviderError{
		{
			Name:         "bad apiKey",
			APIKey:       "bad apiKey",
			Model:        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
			ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
			ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
			ErrGenDoc:    "can only generate audio and images",
			ErrListModel: "http 401\nUnauthorized\nget a new API key at https://api.together.xyz/settings/api-keys",
		},
		{
			Name:         "bad apiKey image",
			APIKey:       "bad apiKey",
			Model:        "black-forest-labs/FLUX.1-schnell",
			ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
			ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
			ErrGenDoc:    "http 401\ninvalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
			ErrListModel: "http 401\nUnauthorized\nget a new API key at https://api.together.xyz/settings/api-keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
			ErrGenStream: "http 404\nmodel_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
			ErrGenDoc:    "can only generate audio and images",
		},
	}
	f := func(t *testing.T, apiKey, model string) (genai.Provider, error) {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *togetherai.Client {
	t.Parallel()
	c, err := getClientInner(t, "", m)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, apiKey, m string) (*togetherai.Client, error) {
	if apiKey == "" && os.Getenv("TOGETHER_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	return togetherai.New(&genai.OptionsProvider{APIKey: apiKey, Model: m}, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
}

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
