// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Pollinations provider client.

package pollinations_test

import (
	"context"
	_ "embed"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/pollinations"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

// providerOptions is a convenience struct used only in tests to group provider options.
type providerOptions struct {
	apiKey           string
	Model            string
	OutputModalities genai.Modalities
	PreloadedModels  []genai.Model
}

func (o *providerOptions) toOptions() []genai.ProviderOption {
	var opts []genai.ProviderOption
	if o.Model != "" {
		opts = append(opts, genai.ProviderOptionModel(o.Model))
	}
	if o.apiKey != "" {
		opts = append(opts, genai.ProviderOptionAPIKey(o.apiKey))
	}
	if len(o.OutputModalities) > 0 {
		opts = append(opts, genai.ProviderOptionModalities(o.OutputModalities))
	}
	if len(o.PreloadedModels) > 0 {
		opts = append(opts, genai.ProviderOptionPreloadedModels(o.PreloadedModels))
	}
	return opts
}

func getClientInner(t *testing.T, opts *providerOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	// Pollinations API rejects invalid keys but works without auth.
	// Don't use a placeholder key for playback - leave empty to skip auth header.
	provOpts := opts.toOptions()
	if fn != nil {
		provOpts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, provOpts...)
	}
	return pollinations.New(t.Context(), provOpts...)
}

func TestImageModelsResponse(t *testing.T) {
	t.Run("metadata fields", func(t *testing.T) {
		body := `[{"name":"klein","aliases":["flux-klein"],"category":"image","brand":"Black Forest Labs","pricing":{"currency":"pollen","completionImageTokens":"0.01"},"title":"FLUX.2 Klein 4B","description":"FLUX.2 Klein 4B - Fast image generation and editing","input_modalities":["text","image"],"output_modalities":["image"],"max_reference_images":10,"capabilities":[],"alpha":true,"added_date":1768608000000}]`
		var resp pollinations.ImageModelsResponse
		dec := json.NewDecoder(strings.NewReader(body))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&resp); err != nil {
			t.Fatal(err)
		}
		if !resp[0].Alpha {
			t.Fatal("Alpha = false, want true")
		}
		if resp[0].AddedDate != 1768608000000 {
			t.Fatalf("AddedDate = %d, want 1768608000000", resp[0].AddedDate)
		}
	})
}

func TestTextModelsResponse(t *testing.T) {
	t.Run("metadata fields", func(t *testing.T) {
		body := `[{"name":"openai","aliases":["gpt-5.4-nano"],"category":"text","brand":"OpenAI","pricing":{"currency":"pollen","promptTextTokens":"0.00000015","promptCachedTokens":"0.000000015","completionTextTokens":"0.0000009375"},"title":"GPT-5.4 Nano","description":"GPT-5.4 Nano - Fast & Balanced","input_modalities":["text","image"],"output_modalities":["text"],"max_reference_images":10,"capabilities":["tool_calling"],"tools":true,"context_length":400000,"is_specialized":false,"alpha":true,"added_date":1759795200000}]`
		var resp pollinations.TextModelsResponse
		dec := json.NewDecoder(strings.NewReader(body))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&resp); err != nil {
			t.Fatal(err)
		}
		if got := resp[0].Capabilities; !slices.Equal(got, []string{"tool_calling"}) {
			t.Fatalf("Capabilities = %q, want [tool_calling]", got)
		}
		if !resp[0].Alpha {
			t.Fatal("Alpha = false, want true")
		}
		if resp[0].AddedDate != 1759795200000 {
			t.Fatalf("AddedDate = %d, want 1759795200000", resp[0].AddedDate)
		}
	})
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, &providerOptions{}, func(h http.RoundTripper) http.RoundTripper {
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
		opts := providerOptions{Model: m, PreloadedModels: cachedModels}
		ci, err := getClientInner(t, &opts, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, ""))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		sbModels := make([]scoreboard.Model, 0, len(cachedModels))
		for _, m := range cachedModels {
			id := m.GetID()
			sbModels = append(sbModels, scoreboard.Model{Model: id, Reason: id == "deepseek-reasoning"})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
			}
			c, err := pollinations.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			c2 := &smallImage{Provider: &internaltest.HideHTTPCode{Provider: c, StatusCode: 500}}
			if model.Reason {
				for _, sc := range c.Scoreboard().Scenarios {
					if sc.Reason && slices.Contains(sc.Models, model.Model) {
						if sc.ReasoningTokenStart != "" && sc.ReasoningTokenEnd != "" {
							return &adapters.ProviderReasoning{
								Provider:            c2,
								ReasoningTokenStart: sc.ReasoningTokenStart,
								ReasoningTokenEnd:   sc.ReasoningTokenEnd,
							}
						}
						break
					}
				}
			}
			return c2
		}
		smoketest.Run(t, getClientRT, sbModels, testRecorder.Records, nil)
	})

	// Note: Skipping Preferred test as pollinations is a router to multiple backends
	// and preferred model selection is handled dynamically by the provider's
	// selectBestTextModel/selectBestImageModel logic rather than static scoreboard entries.

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, string(genai.ModelCheap))
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "model \"bad model\" not supported by pollinations",
				ErrGenStream: "model \"bad model\" not supported by pollinations",
			},
			{
				Name: "bad model image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "model \"bad model\" not supported by pollinations",
				ErrGenStream: "model \"bad model\" not supported by pollinations",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			if !hasModalities(opts) {
				opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			}
			return pollinations.New(t.Context(), append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})}, opts...)...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

// smallImage speeds up image generation.
type smallImage struct {
	genai.Provider
}

func (h *smallImage) Unwrap() genai.Provider {
	return h.Provider
}

func (h *smallImage) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	for i := range opts {
		if v, ok := opts[i].(*genai.GenOptionImage); ok {
			// Ask for a smaller size.
			n := *v
			n.Width = 256
			n.Height = 256
			opts[i] = &n
		}
	}
	return h.Provider.GenSync(ctx, msgs, opts...)
}

func hasModalities(opts []genai.ProviderOption) bool {
	return slices.ContainsFunc(opts, func(o genai.ProviderOption) bool {
		_, ok := o.(genai.ProviderOptionModalities)
		return ok
	})
}

func init() {
	internal.BeLenient = false
}
