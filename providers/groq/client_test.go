// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq_test

import (
	"context"
	_ "embed"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/groq"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen {
		c := getClient(t, m)
		if strings.HasPrefix(m, "qwen/") {
			return &handleGroqReasoning{
				Client: &adapters.ProviderGenAppend{
					ProviderGen: c,
					Append:      genai.NewTextMessage(genai.User, "/think"),
				},
				t: t,
			}
		}
		if m == "deepseek-r1-distill-llama-70b" {
			return &handleGroqReasoning{Client: c, t: t}
		}
		return c
	}, nil)
}

type handleGroqReasoning struct {
	Client genai.ProviderGen
	t      *testing.T
}

func (h *handleGroqReasoning) Name() string {
	return h.Client.Name()
}

func (h *handleGroqReasoning) ModelID() string {
	return h.Client.ModelID()
}

func (h *handleGroqReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	if opts != nil {
		if o := opts.(*genai.OptionsText); len(o.Tools) != 0 || o.DecodeAs != nil || o.ReplyAsJSON {
			opts = &groq.OptionsText{ReasoningFormat: groq.ReasoningFormatParsed, OptionsText: *o}
			return h.Client.GenSync(ctx, msgs, opts)
		}
	}
	c := adapters.ProviderGenThinking{ProviderGen: h.Client, TagName: "think"}
	return c.GenSync(ctx, msgs, opts)
}

func (h *handleGroqReasoning) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if opts != nil {
		if o := opts.(*genai.OptionsText); len(o.Tools) != 0 || o.DecodeAs != nil || o.ReplyAsJSON {
			opts = &groq.OptionsText{ReasoningFormat: groq.ReasoningFormatParsed, OptionsText: *o}
			return h.Client.GenStream(ctx, msgs, replies, opts)
		}
	}
	c := adapters.ProviderGenThinking{ProviderGen: h.Client, TagName: "think"}
	return c.GenStream(ctx, msgs, replies, opts)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "llama-3.1-8b-instant"},
		{base.PreferredGood, "meta-llama/llama-4-maverick-17b-128e-instruct"},
		{base.PreferredSOTA, "qwen/qwen3-32b"},
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
			Model:        "llama-3.1-8b-instant",
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
			Err:    "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *groq.Client {
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *groq.Client {
	if apiKey == "" && os.Getenv("GROQ_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := groq.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
