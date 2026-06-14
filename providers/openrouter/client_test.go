// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenRouter provider client.

package openrouter_test

import (
	"encoding/json"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openrouter"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, opts []genai.ProviderOption, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if os.Getenv("OPENROUTER_API_KEY") == "" && !slices.ContainsFunc(opts, func(o genai.ProviderOption) bool { _, ok := o.(genai.ProviderOptionAPIKey); return ok }) {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	if fn != nil {
		opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
	}
	return openrouter.New(t.Context(), opts...)
}

func TestChatRequest(t *testing.T) {
	t.Run("structured output", func(t *testing.T) {
		var req openrouter.ChatRequest
		err := req.Init(
			genai.Messages{genai.NewTextMessage("Reply in JSON.")},
			"qwen/qwen3.5-35b-a3b",
			&genai.GenOptionText{DecodeAs: genai.JSONSchema(`{"type":"object"}`)},
		)
		if err != nil {
			t.Fatal(err)
		}
		got, err := json.Marshal(req.ResponseFormat)
		if err != nil {
			t.Fatal(err)
		}
		want := `{"type":"json_schema","json_schema":{"name":"response","schema":{"type":"object"},"strict":true}}`
		if string(got) != want {
			t.Fatalf("ResponseFormat mismatch:\n got: %s\nwant: %s", got, want)
		}
	})
}

func TestChatResponse(t *testing.T) {
	t.Run("service tier and string content", func(t *testing.T) {
		body := `{"id":"gen-test","object":"chat.completion","created":1781467681,"model":"qwen/qwen3.5-35b-a3b-20260224","provider":"Ambient","service_tier":"auto","choices":[{"index":0,"finish_reason":"stop","native_finish_reason":"stop","message":{"role":"assistant","content":"Hello"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`
		var resp openrouter.ChatResponse
		dec := json.NewDecoder(strings.NewReader(body))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&resp); err != nil {
			t.Fatal(err)
		}
		got, err := resp.ToResult()
		if err != nil {
			t.Fatal(err)
		}
		if got.String() != "Hello" {
			t.Fatalf("String() = %q, want Hello", got.String())
		}
		if got.Usage.ServiceTier != "auto" {
			t.Fatalf("ServiceTier = %q, want auto", got.Usage.ServiceTier)
		}
	})
}

func TestChatStreamChunkResponse(t *testing.T) {
	t.Run("service tier and string content", func(t *testing.T) {
		body := `{"id":"gen-test","object":"chat.completion.chunk","created":1781467681,"model":"qwen/qwen3.5-35b-a3b-20260224","provider":"Ambient","service_tier":"auto","choices":[{"index":0,"delta":{"content":"Hello","role":"assistant"},"finish_reason":"stop","native_finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}`
		var resp openrouter.ChatStreamChunkResponse
		dec := json.NewDecoder(strings.NewReader(body))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&resp); err != nil {
			t.Fatal(err)
		}
		if resp.ServiceTier != "auto" {
			t.Fatalf("ServiceTier = %q, want auto", resp.ServiceTier)
		}
	})
}

func TestModelsResponse(t *testing.T) {
	t.Run("benchmarks", func(t *testing.T) {
		body := `{"data":[{"id":"moonshotai/kimi-k2.7-code","canonical_slug":"moonshotai/kimi-k2.7-code","hugging_face_id":null,"name":"Kimi K2.7 Code","created":1781371647,"description":"test","context_length":262144,"architecture":{"modality":"text->text","input_modalities":["text"],"output_modalities":["text"],"tokenizer":"Moonshot","instruct_type":null},"pricing":{"prompt":"0.0000005","completion":"0.000002"},"top_provider":{"context_length":262144,"max_completion_tokens":65536,"is_moderated":false},"per_request_limits":null,"supported_parameters":["tools"],"default_parameters":{"temperature":null,"top_p":null,"top_k":null,"frequency_penalty":null,"presence_penalty":null,"repetition_penalty":null},"supported_voices":null,"knowledge_cutoff":null,"expiration_date":null,"links":{"details":"/api/v1/models/moonshotai/kimi-k2.7-code/endpoints"},"benchmarks":{"artificial_analysis":{"intelligence_index":64.9,"coding_index":62,"agentic_index":80.6},"design_arena":[{"arena":"agents","category":"fullstack","elo":1210,"win_rate":52.7,"rank":11}]}}]}`
		var resp openrouter.ModelsResponse
		dec := json.NewDecoder(strings.NewReader(body))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&resp); err != nil {
			t.Fatal(err)
		}
		got := resp.Data[0].Benchmarks.ArtificialAnalysis.AgenticIndex
		if got != 80.6 {
			t.Fatalf("AgenticIndex = %g, want 80.6", got)
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
	cl, err2 := getClientInner(t, nil, func(h http.RoundTripper) http.RoundTripper {
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
		ci, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
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
		c := getClient(t, "")
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Scenarios
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			id := m.GetID()
			reason := false
			for _, sc := range scenarios {
				if slices.Contains(sc.Models, id) {
					reason = sc.Reason
					break
				}
			}
			models = append(models, scoreboard.Model{Model: id, Reason: reason})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("OPENROUTER_API_KEY") == "" {
				opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if fn != nil {
				opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
			}
			c, err := openrouter.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			return c
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
			return getClientInner(st, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			})
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
					genai.ProviderOptionModel("openai/gpt-4o-mini"),
				},
				ErrGenSync:   "http 401\n401 (): Missing Authentication header\nget a new API key at https://openrouter.ai/settings/keys",
				ErrGenStream: "http 401\n401 (): Missing Authentication header\nget a new API key at https://openrouter.ai/settings/keys",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad/model"),
				},
				ErrGenSync:   "http 400\n400 (): bad/model is not a valid model ID",
				ErrGenStream: "http 400\n400 (): bad/model is not a valid model ID",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func init() {
	internal.BeLenient = false
}
