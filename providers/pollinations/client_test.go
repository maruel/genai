// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package pollinations_test

import (
	"context"
	_ "embed"
	"net/http"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/pollinations"
)

func TestClient_Scoreboard(t *testing.T) {
	var models []genai.Model
	t.Run("ListModel", func(t *testing.T) {
		c := getClientInner(t, "")
		var err error
		models, err = c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
	})
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen {
		c := getClient(t, m)
		isImage := false
		for i := range models {
			if models[i].GetID() == c.Model {
				_, isImage = models[i].(pollinations.ImageModel)
			}
		}
		if isImage {
			return &imageModelClient{Client: c}
		}
		if m == "deepseek-reasoning" {
			return &adapters.ProviderGenThinking{ProviderGen: c, TagName: "think"}
		}
		return c
	}, nil)
}

type imageModelClient struct {
	*pollinations.Client
}

func (i *imageModelClient) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	if v, ok := opts.(*genai.OptionsImage); ok {
		n := *v
		n.Width = 256
		n.Height = 256
		opts = &n
	}
	return i.Client.GenDoc(ctx, msg, opts)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "llamascout"},
		{base.PreferredGood, "openai-large"},
		{base.PreferredSOTA, "deepseek-reasoning"},
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
	t.Skip("TODO")
	data := []internaltest.ProviderError{
		{
			Name:         "bad apiKey",
			APIKey:       "bad apiKey",
			Model:        "llama3-8b-8192",
			ErrGenSync:   "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
			ErrGenStream: "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404: error model_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
			ErrGenStream: "http 404: error model_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.Provider {
		return getClientInner(t, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *pollinations.Client {
	t.Parallel()
	return getClientInner(t, m)
}

func getClientInner(t *testing.T, m string) *pollinations.Client {
	c, err := pollinations.New("genai-unittests", m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
