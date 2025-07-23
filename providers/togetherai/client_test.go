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

func gc(t testing.TB, name, m string) (genai.Provider, http.RoundTripper) {
	var rt http.RoundTripper
	fn := func(h http.RoundTripper) http.RoundTripper {
		if name == "" {
			rt = h
			return h
		}
		r, err2 := testRecorder.Records.Record(name, h)
		if err2 != nil {
			t.Fatal(err2)
		}
		t.Cleanup(func() {
			if err3 := r.Stop(); err3 != nil {
				t.Error(err3)
			}
		})
		rt = r
		return r
	}
	apiKey := ""
	if os.Getenv("TOGETHER_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := togetherai.New(apiKey, m, fn)
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(m, "black-forest-labs/") {
		return &smallImageGen{hideHTTP500{c}}, rt
	}
	// If anyone at Together.AI reads this, please get your shit together.
	return &hideHTTP500{c}, rt
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
	t.Parallel()
	usage := genai.Usage{}
	cc, _ := gc(t, t.Name()+"/ListModels", "")
	models, err2 := cc.(genai.ProviderModel).ListModels(t.Context())
	if err2 != nil {
		t.Fatal(err2)
	}
	for _, m := range models {
		id := m.GetID()
		t.Run(id, func(t *testing.T) {
			// Run one model at a time otherwise we can't collect the total usage.
			usage.Add(scoreboardtest.RunOneModel(t, func(t testing.TB, sn string) (genai.Provider, http.RoundTripper) {
				return gc(t, sn, id)
			}))
		})
	}
	t.Logf("Usage: %#v", usage)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"},
		{base.PreferredGood, "Qwen/Qwen2.5-72B-Instruct-Turbo"},
		{base.PreferredSOTA, "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			if got := getClient(t, line.name).ModelID(); got != line.want {
				t.Fatalf("got model %q, want %q", got, line.want)
			}
		})
	}
}

func TestClient_ProviderGen_errors(t *testing.T) {
	data := []internaltest.ProviderGenError{
		{
			Name:         "bad apiKey",
			ApiKey:       "bad apiKey",
			Model:        "meta-llama/Llama-3.2-3B-Instruct-Turbo",
			ErrGenSync:   "http 401: error invalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
			ErrGenStream: "http 401: error invalid_api_key (invalid_request_error): Invalid API key provided. You can find your API key at https://api.together.xyz/settings/api-keys.",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404: error model_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
			ErrGenStream: "http 404: error model_not_available (invalid_request_error): Unable to access model bad model. Please visit https://api.together.ai/models to view the list of supported models.",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ProviderGen {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_ProviderGen_errors(t, f, data)
}

func TestClient_ProviderModel_errors(t *testing.T) {
	data := []internaltest.ProviderModelError{
		{
			Name:   "bad apiKey",
			ApiKey: "badApiKey",
			Err:    "http 401: error (): Unauthorized. You can get a new API key at https://api.together.xyz/settings/api-keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *togetherai.Client {
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *togetherai.Client {
	if apiKey == "" && os.Getenv("TOGETHER_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := togetherai.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
	if err != nil {
		t.Fatal(err)
	}
	return c
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
