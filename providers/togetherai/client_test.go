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
)

func TestClient_Scoreboard(t *testing.T) {
	f := func(m genai.Model) bool {
		model := m.(*togetherai.Model)
		if model.ID == "arcee-ai/maestro-reasoning" || // Requires CoT processing.
			model.ID == "google/gemma-2b-it" || // Doesn't follow instruction.
			model.ID == "deepseek-ai/DeepSeek-V3-p-dp" || // Causes HTTP 503.
			model.ID == "meta-llama/Llama-3.3-70B-Instruct-Turbo" || // rate_limit even if been a while.
			model.ID == "togethercomputer/Refuel-Llm-V2-Small" || // Fails because Seed option.
			strings.HasPrefix(model.ID, "deepseek-ai/DeepSeek-R1") || // Requires CoT processing.
			strings.HasPrefix(model.ID, "perplexity-ai/r1-") || // Requires CoT processing.
			strings.HasPrefix(model.ID, "Qwen/QwQ-32B") || // Requires CoT processing.
			strings.HasPrefix(model.ID, "Qwen/Qwen3-235B-A22B-") || // Requires CoT processing.
			strings.HasPrefix(model.ID, "togethercomputer/MoA-1") { // Causes HTTP 500.
			return false
		}
		return model.Type == "chat" || model.Type == "image"
	}
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen {
		c := getClient(t, m)
		// TODO: Use Scoreboard list.
		if strings.HasPrefix(c.Model, "black-forest-labs/") {
			return &injectOption{Client: c, t: t, opts: genai.OptionsImage{Width: 256, Height: 256}}
		}
		return c
	}, f)
}

type injectOption struct {
	*togetherai.Client
	t    *testing.T
	opts genai.OptionsImage
}

func (i *injectOption) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		return genai.Result{}, errors.New("implement me")
	}
	opts = &n
	return i.Client.GenSync(ctx, msgs, opts)
}

func (i *injectOption) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		return genai.Result{}, errors.New("implement me")
	}
	opts = &n
	return i.Client.GenStream(ctx, msgs, replies, opts)
}

func (i *injectOption) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		return genai.Result{}, errors.New("implement me")
	}
	opts = &n
	return i.Client.GenDoc(ctx, msg, opts)
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
	testRecorder.Signal(t)
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
