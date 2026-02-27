// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package alibaba_test

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
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

	backends := []struct {
		name           string
		opt            alibaba.ProviderOptionBackend
		envKey         string
		sbFile         string
		docModel       string // model for TextOutputDocInput (must support GenSync)
		errModel       string // model for error tests
		errBadAPIKeyRE string // error substring for bad API key (differs per region)
	}{
		{"US", alibaba.BackendUS, "DASHSCOPE_API_KEY_US", "scoreboard_us.json", string(genai.ModelCheap), "qwen3-30b-a3b", "help.aliyun.com/zh/"},
		{"Intl", alibaba.BackendIntl, "DASHSCOPE_API_KEY_INTL", "scoreboard_intl.json", string(genai.ModelSOTA), "qwen3.5-35b-a3b", "www.alibabacloud.com/help/en/"},
	}
	for _, b := range backends {
		t.Run(b.name, func(t *testing.T) {
			// Skip if no recordings and no API key.
			warmupPath := filepath.Join("testdata", t.Name(), "Warmup.yaml")
			hasKey := os.Getenv(b.envKey) != "" || os.Getenv("DASHSCOPE_API_KEY") != ""
			if _, err := os.Stat(warmupPath); err != nil && !hasKey {
				t.Skipf("no recordings and no API key for %s", b.name)
			}

			cl, err2 := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.RecordWithName(t, t.Name()+"/Warmup", h)
			}, b.opt)
			if err2 != nil {
				t.Fatal(err2)
			}
			cachedModels, err2 := cl.ListModels(t.Context())
			if err2 != nil {
				t.Fatal(err2)
			}
			getClient := func(t *testing.T, m string) genai.Provider {
				t.Parallel()
				opts := []genai.ProviderOption{
					genai.ProviderOptionPreloadedModels(cachedModels),
					b.opt,
				}
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
				// Build the Reason lookup from the existing scoreboard.
				reasonModels := map[string]bool{}
				for _, sc := range alibaba.ScoreboardForBackend(b.opt).Scenarios {
					for _, m := range sc.Models {
						reasonModels[m] = sc.Reason
					}
				}
				// Build model list from ListModels so new models are discovered.
				// Qwen3.5 models support hybrid-thinking: test both with and without thinking.
				models := make([]scoreboard.Model, 0, len(cachedModels))
				for _, m := range cachedModels {
					id := m.GetID()
					if strings.HasPrefix(id, "qwen3.5-") {
						models = append(models, scoreboard.Model{Model: id, Reason: false})
						models = append(models, scoreboard.Model{Model: id, Reason: true})
					} else {
						reason, ok := reasonModels[id]
						if !ok {
							reason = strings.Contains(id, "thinking") || strings.Contains(id, "qwq") || strings.Contains(id, "qvq")
						}
						models = append(models, scoreboard.Model{Model: id, Reason: reason})
					}
				}
				getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
					opts := []genai.ProviderOption{
						genai.ProviderOptionPreloadedModels(cachedModels),
						b.opt,
					}
					if model.Model != "" {
						opts = append(opts, genai.ProviderOptionModel(model.Model))
					}
					if os.Getenv("DASHSCOPE_API_KEY") == "" && !hasRegionKey() {
						opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
					}
					if fn != nil {
						opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
					}
					c, err := alibaba.New(t.Context(), opts...)
					if err != nil {
						t.Fatal(err)
					}
					return &internaltest.InjectOptions{
						Provider: c,
						Opts:     []genai.GenOption{&alibaba.GenOption{EnableThinking: model.Reason}},
					}
				}
				smoketest.Run(t, getClientRT, models, testRecorder.Records, smoketest.WithScoreboardFile(b.sbFile))
			})

			t.Run("Preferred", func(t *testing.T) {
				internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
					opts := []genai.ProviderOption{
						genai.ProviderOptionModalities{modality},
						genai.ProviderOptionPreloadedModels(cachedModels),
						b.opt,
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
					return getClient(t, b.docModel)
				})
			})

			t.Run("errors", func(t *testing.T) {
				errBadAPIKey := "http 401\ninvalid_request_error: Incorrect API key provided. For details, see: https://" + b.errBadAPIKeyRE + "model-studio/error-code#apikey-error\nget a new API key at https://modelstudio.console.alibabacloud.com/"
				data := []internaltest.ProviderError{
					{
						Name: "bad apiKey",
						Opts: []genai.ProviderOption{
							genai.ProviderOptionAPIKey("bad apiKey"),
							genai.ProviderOptionModel(b.errModel),
						},
						ErrGenSync:   errBadAPIKey,
						ErrGenStream: errBadAPIKey,
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
					opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText}, b.opt)
					return getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
						return testRecorder.Record(t, h)
					}, opts...)
				}
				internaltest.TestClientProviderErrors(t, f, data)
			})
		})
	}
}

func TestGenOption(t *testing.T) {
	msgs := genai.Messages{genai.NewTextMessage("test")}
	t.Run("enabled", func(t *testing.T) {
		var req alibaba.ChatRequest
		if err := req.Init(msgs, "qwen3.5-397b-a17b", &alibaba.GenOption{EnableThinking: true}); err != nil {
			t.Fatal(err)
		}
		if !req.EnableThinking {
			t.Error("EnableThinking = false, want true")
		}
	})
	t.Run("disabled", func(t *testing.T) {
		var req alibaba.ChatRequest
		if err := req.Init(msgs, "qwen3.5-397b-a17b", &alibaba.GenOption{EnableThinking: false}); err != nil {
			t.Fatal(err)
		}
		if req.EnableThinking {
			t.Error("EnableThinking = true, want false")
		}
	})
	t.Run("budget", func(t *testing.T) {
		var req alibaba.ChatRequest
		if err := req.Init(msgs, "qwen3.5-397b-a17b", &alibaba.GenOption{EnableThinking: true, ThinkingBudget: 4096}); err != nil {
			t.Fatal(err)
		}
		if req.ThinkingBudget != 4096 {
			t.Errorf("ThinkingBudget = %d, want 4096", req.ThinkingBudget)
		}
	})
	t.Run("json_false_present", func(t *testing.T) {
		// Verify enable_thinking:false is serialized (not omitted).
		var req alibaba.ChatRequest
		if err := req.Init(msgs, "qwen3.5-397b-a17b", &alibaba.GenOption{}); err != nil {
			t.Fatal(err)
		}
		b, err := json.Marshal(&req)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(b), `"enable_thinking":false`) {
			t.Errorf("expected enable_thinking:false in JSON, got: %s", b)
		}
	})
}

func init() {
	internal.BeLenient = false
}
