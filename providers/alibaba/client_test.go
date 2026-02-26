// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package alibaba_test

import (
	"net/http"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/alibaba"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

// hasRegionKey reports whether any region-specific DASHSCOPE_API_KEY_* env var is set.
func hasRegionKey() bool {
	return os.Getenv("DASHSCOPE_API_KEY_INTL") != "" || os.Getenv("DASHSCOPE_API_KEY_US") != "" || os.Getenv("DASHSCOPE_API_KEY_CN") != ""
}

func getClientInner(t *testing.T, fn func(http.RoundTripper) http.RoundTripper, opts ...genai.ProviderOption) (genai.Provider, error) {
	hasAPIKey := false
	hasBackend := false
	for _, opt := range opts {
		switch opt.(type) {
		case genai.ProviderOptionAPIKey:
			hasAPIKey = true
		case genai.ProviderOptionRemote, alibaba.ProviderOptionBackend:
			hasBackend = true
		}
	}
	if !hasAPIKey && os.Getenv("DASHSCOPE_API_KEY") == "" && !hasRegionKey() {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	if !hasBackend {
		if u := os.Getenv("DASHSCOPE_BASE_URL"); u != "" {
			opts = append(opts, genai.ProviderOptionRemote(u))
		} else if !hasRegionKey() {
			// Default to US for recording playback.
			opts = append(opts, alibaba.BackendUS)
		}
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	return alibaba.New(t.Context(), opts...)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.RecordWithName(t, t.Name()+"/Warmup", h)
	})
	if err2 != nil {
		t.Fatal(err2)
	}
	cachedModels, err2 := cl.ListModels(t.Context())
	if err2 != nil {
		t.Fatal(err2)
	}
	getClient := func(t *testing.T, m string) genai.Provider {
		t.Parallel()
		opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
		if m != "" {
			opts = append(opts, genai.ProviderOptionModel(m))
		}
		ci, err := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		}, opts...)
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, ""))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		var models []scoreboard.Model
		for _, sc := range alibaba.Scoreboard().Scenarios {
			for _, m := range sc.Models {
				models = append(models, scoreboard.Model{Model: m, Reason: sc.Reason})
			}
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("DASHSCOPE_API_KEY") == "" && !hasRegionKey() {
				opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if u := os.Getenv("DASHSCOPE_BASE_URL"); u != "" {
				opts = append(opts, genai.ProviderOptionRemote(u))
			} else if !hasRegionKey() {
				opts = append(opts, alibaba.BackendUS)
			}
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
			}
			c, err := alibaba.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			return getClientInner(st, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			}, opts...)
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, string(genai.ModelCheap))
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("qwen3-30b-a3b"),
				},
				ErrGenSync:   "http 401\ninvalid_request_error: Incorrect API key provided. For details, see: https://help.aliyun.com/zh/model-studio/error-code#apikey-error\nget a new API key at https://modelstudio.console.alibabacloud.com/",
				ErrGenStream: "http 401\ninvalid_request_error: Incorrect API key provided. For details, see: https://help.aliyun.com/zh/model-studio/error-code#apikey-error\nget a new API key at https://modelstudio.console.alibabacloud.com/",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 404\ninvalid_request_error: The model `bad model` does not exist or you do not have access to it.",
				ErrGenStream: "http 404\ninvalid_request_error: The model `bad model` does not exist or you do not have access to it.",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			return getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			}, opts...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func init() {
	internal.BeLenient = false
}
