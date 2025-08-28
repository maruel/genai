// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"net/http"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/providers/huggingface"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientRT(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	opts := genai.ProviderOptions{
		APIKey:          getAPIKeyTest(t),
		Model:           model.Model,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := huggingface.New(t.Context(), &opts, fn)
	if err != nil {
		t.Fatal(err)
	}
	if strings.HasPrefix(model.Model, "Qwen/Qwen3") {
		if !model.Thinking {
			return &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/no_think"}}
		}
		// Check if it has predefined thinking tokens.
		for _, sc := range c.Scoreboard().Scenarios {
			if sc.Thinking && slices.Contains(sc.Models, model.Model) {
				if sc.ThinkingTokenStart != "" && sc.ThinkingTokenEnd != "" {
					return &adapters.ProviderThinking{
						Provider:           &adapters.ProviderAppend{Provider: c, Append: genai.Request{Text: "\n\n/think"}},
						ThinkingTokenStart: sc.ThinkingTokenStart,
						ThinkingTokenEnd:   sc.ThinkingTokenEnd,
					}
				}
				break
			}
		}
	}
	return c
}

func TestClient_Scoreboard(t *testing.T) {
	// We do not want to test thousands of models, so get the ones already in the scoreboard.
	sb := getClient(t, genai.ModelNone).Scoreboard()
	var models []smoketest.Model
	for _, sc := range sb.Scenarios {
		for _, model := range sc.Models {
			models = append(models, smoketest.Model{Model: model, Thinking: sc.Thinking})
		}
	}
	smoketest.Run(t, getClientRT, models, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		// It oscillates between models.
		{genai.ModelCheap, "meta-llama/Llama-3.2-1B-Instruct"},
		{genai.ModelGood, "Qwen/Qwen3-Coder-480B-A35B-Instruct"},
		{genai.ModelSOTA, "deepseek-ai/DeepSeek-V3.1"},
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
			Name: "bad apiKey",
			Opts: genai.ProviderOptions{
				APIKey: "bad apiKey",
				Model:  "Qwen/Qwen3-4B",
			},
			ErrGenSync:   "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
			ErrGenStream: "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
			ErrListModel: "http 401\nInvalid credentials in Authorization header\nget a new API key at https://huggingface.co/settings/tokens",
		},
		{
			Name: "bad model",
			Opts: genai.ProviderOptions{
				Model: "bad model",
			},
			ErrGenSync:   "http 400\ninvalid_request_error (model_not_found): model: The requested model 'bad model' does not exist.",
			ErrGenStream: "http 400\ninvalid_request_error (model_not_found): model: The requested model 'bad model' does not exist.",
		},
	}
	f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
		opts.OutputModalities = genai.Modalities{genai.ModalityText}
		return getClientInner(t, opts)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *huggingface.Client {
	t.Parallel()
	opts := genai.ProviderOptions{
		Model:           m,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := getClientInner(t, opts)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, opts genai.ProviderOptions) (*huggingface.Client, error) {
	if opts.APIKey == "" {
		opts.APIKey = getAPIKeyTest(t)
	}
	return huggingface.New(t.Context(), &opts, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
}

func getAPIKeyTest(t testing.TB) string {
	apiKey, err := getAPIKey()
	if err != nil {
		t.Fatal(err)
	}
	return apiKey
}

func loadCachedModelsList(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r myrecorder.Recorder
		var err2 error
		ctx := t.Context()
		opts := genai.ProviderOptions{
			APIKey: getAPIKeyTest(t),
			Model:  genai.ModelNone,
		}
		c, err := huggingface.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
			r, err2 = testRecorder.Records.Record("WarmupCache", h)
			return r
		})
		if err != nil {
			t.Fatal(err)
		}
		if err2 != nil {
			t.Fatal(err2)
		}
		if cachedModels, err = c.ListModels(ctx); err != nil {
			t.Fatal(err)
		}
		if err = r.Stop(); err != nil {
			t.Fatal(err)
		}
	})
	return cachedModels
}

var doOnce sync.Once

var cachedModels []genai.Model

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
