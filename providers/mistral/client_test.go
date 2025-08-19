// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	"context"
	_ "embed"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/mistral"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
	"github.com/maruel/httpjson"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("MISTRAL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := mistral.New(&genai.ProviderOptions{APIKey: apiKey, Model: model.Model}, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
		t.Fatal("implement me")
	}
	if strings.HasPrefix(model.Model, "voxtral") {
		// If anyone at Mistral reads this, please get your shit together.
		return &hideHTTP500{c}
	}
	return c
}

type hideHTTP500 struct {
	*mistral.Client
}

func (h *hideHTTP500) Unwrap() genai.Provider {
	return h.Client
}

func (h *hideHTTP500) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenSync(ctx, msgs, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; voxtral throws HTTP 500 on unsupported file format, e.g. AAC.
			return resp, errors.New("voxtral doesn't support this input format")
		}
		return resp, err
	}
	return resp, err
}

func (h *hideHTTP500) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenStream(ctx, msgs, chunks, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; voxtral throws HTTP 500 on unsupported file format, e.g. AAC.
			return resp, errors.New("voxtral doesn't support this input format")
		}
		return resp, err
	}
	return resp, err
}

func TestClient_Scoreboard(t *testing.T) {
	genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
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
		{genai.ModelCheap, "mistral-tiny-latest"},
		{genai.ModelGood, "mistral-medium-latest"},
		{genai.ModelSOTA, "mistral-large-latest"},
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
			Model:        "ministral-3b-latest",
			ErrGenSync:   "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
			ErrGenStream: "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
			ErrListModel: "http 401\nUnauthorized\nget a new API key at https://console.mistral.ai/api-keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 400\ninvalid_model: Invalid model: bad model",
			ErrGenStream: "http 400\ninvalid_model: Invalid model: bad model",
		},
	}
	f := func(t *testing.T, apiKey, model string) (genai.Provider, error) {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *mistral.Client {
	t.Parallel()
	c, err := getClientInner(t, "", m)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, apiKey, m string) (*mistral.Client, error) {
	if apiKey == "" && os.Getenv("MISTRAL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	return mistral.New(&genai.ProviderOptions{APIKey: apiKey, Model: m}, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
