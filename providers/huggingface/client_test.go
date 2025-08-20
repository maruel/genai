// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/huggingface"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	c, err := huggingface.New(&genai.ProviderOptions{APIKey: getAPIKeyTest(t), Model: model.Model}, fn)
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(model.Model, "Qwen/Qwen3") {
		if model.Thinking {
			return &adapters.ProviderThinking{
				Provider:           &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/think"}},
				ThinkingTokenStart: "<think>",
				ThinkingTokenEnd:   "</think>",
			}
		} else {
			return &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/no_think"}}
		}
	}
	return c
}

func TestClient_Scoreboard(t *testing.T) {
	// We do not want to test thousands of models, so get the ones already in the scoreboard.
	sb := getClient(t, genai.ModelNone).Scoreboard()
	var models []scoreboardtest.Model
	for _, sc := range sb.Scenarios {
		for _, model := range sc.Models {
			models = append(models, scoreboardtest.Model{Model: model, Thinking: sc.Thinking})
		}
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, "meta-llama/Llama-3.2-3B-Instruct"},
		{genai.ModelGood, "Qwen/Qwen3-Coder-480B-A35B-Instruct"},
		{genai.ModelSOTA, "deepseek-ai/DeepSeek-R1"},
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
			Model:        "Qwen/Qwen3-4B",
			ErrGenSync:   "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
			ErrGenStream: "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
			ErrListModel: "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 400\ninvalid_request_error (model_not_found): model: The requested model 'bad model' does not exist.",
			ErrGenStream: "http 400\ninvalid_request_error (model_not_found): model: The requested model 'bad model' does not exist.",
		},
	}
	f := func(t *testing.T, apiKey, model string) (genai.Provider, error) {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *huggingface.Client {
	t.Parallel()
	c, err := getClientInner(t, "", m)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, apiKey, m string) (*huggingface.Client, error) {
	if apiKey == "" {
		apiKey = getAPIKeyTest(t)
	}
	return huggingface.New(&genai.ProviderOptions{APIKey: apiKey, Model: m}, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
}

func getAPIKeyTest(t testing.TB) string {
	apiKey, err := getAPIKey()
	if err != nil {
		t.Fatal(err)
	}
	return apiKey
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
