// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Xiaomi MiMo provider client.

package xiaomi_test

import (
	"context"
	"iter"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/xiaomi"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

// ttsAdapter wraps a provider to transform messages for TTS models.
//
// The smoke test sends text in a user message, but Xiaomi TTS requires it
// in an assistant message. This adapter moves the text accordingly.
type ttsAdapter struct {
	genai.Provider
}

func (a *ttsAdapter) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	return a.Provider.GenSync(ctx, a.transform(msgs), opts...)
}

func (a *ttsAdapter) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return a.Provider.GenStream(ctx, a.transform(msgs), opts...)
}

// transform moves text from a user message to an assistant message for TTS.
func (a *ttsAdapter) transform(msgs genai.Messages) genai.Messages {
	out := make(genai.Messages, 0, len(msgs))
	for _, m := range msgs {
		if m.Role() == "user" && len(m.Requests) == 1 && m.Requests[0].Text != "" {
			// Convert user text to assistant text (the text to synthesize).
			out = append(out, genai.Message{
				Replies: []genai.Reply{{Text: m.Requests[0].Text}},
			})
		} else {
			out = append(out, m)
		}
	}
	return out
}

func getClientInner(t *testing.T, fn func(http.RoundTripper) http.RoundTripper, opts ...genai.ProviderOption) (genai.Provider, error) {
	hasAPIKey := false
	for _, opt := range opts {
		if _, ok := opt.(genai.ProviderOptionAPIKey); ok {
			hasAPIKey = true
			break
		}
	}
	if !hasAPIKey && os.Getenv("MIMO_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	return xiaomi.New(t.Context(), opts...)
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
		sb := xiaomi.Scoreboard()
		var models []scoreboard.Model
		for _, sc := range sb.Scenarios {
			if sc.Untested() {
				continue
			}
			for _, model := range sc.Models {
				models = append(models, scoreboard.Model{Model: model, Reason: sc.Reason})
			}
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("MIMO_API_KEY") == "" {
				opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
			}
			c, err := xiaomi.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			p := genai.Provider(c)
			if strings.Contains(model.Model, "tts") {
				p = &ttsAdapter{Provider: p}
			}
			return &internaltest.InjectOptions{
				Provider: p,
				Opts:     []genai.GenOption{&xiaomi.GenOption{Thinking: model.Reason}},
			}
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records, nil)
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
					genai.ProviderOptionModel("mimo-v2.5-pro"),
				},
				ErrGenSync:   "http 401\ninvalid_key: Invalid API Key\nget a new API key at https://platform.xiaomimimo.com/#/console",
				ErrGenStream: "http 401\ninvalid_key: Invalid API Key\nget a new API key at https://platform.xiaomimimo.com/#/console",
				ErrListModel: "http 401\ninvalid_key: Invalid API Key\nget a new API key at https://platform.xiaomimimo.com/#/console",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\n: Param Incorrect",
				ErrGenStream: "http 400\n: Param Incorrect",
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
