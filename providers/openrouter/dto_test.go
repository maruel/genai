// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenRouter wire types.

package openrouter_test

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/maruel/genai/providers/openrouter"
)

func TestModel(t *testing.T) {
	t.Run("String", func(t *testing.T) {
		m := openrouter.Model{
			ID:            "openai/gpt-test",
			Name:          "GPT Test",
			Created:       1735689600,
			ContextLength: 1048576,
			Pricing: openrouter.ModelPricing{
				Prompt:     "0.0000005",
				Completion: "0.000002",
			},
		}
		m.Architecture.Modality = "text+image->text"
		m.TopProvider.MaxCompletionTokens = 32768
		want := "openai/gpt-test (2025-01-01): GPT Test (text+image->text) Context: 1048576/32768; in: 0.50$/Mt out: 2.00$/Mt"
		if got := m.String(); got != want {
			t.Fatalf("String() = %q, want %q", got, want)
		}
	})
	t.Run("String sparse", func(t *testing.T) {
		m := openrouter.Model{ID: "test/model", Name: "Test Model", ContextLength: 4096}
		want := "test/model: Test Model Context: 4096"
		if got := m.String(); got != want {
			t.Fatalf("String() = %q, want %q", got, want)
		}
	})
	t.Run("String malformed pricing", func(t *testing.T) {
		m := openrouter.Model{
			ID:            "test/model",
			Name:          "Test Model",
			ContextLength: 4096,
			Pricing: openrouter.ModelPricing{
				Prompt:     "malformed",
				Completion: "0.000001",
			},
		}
		want := "test/model: Test Model Context: 4096; in: malformed$/t out: 1.00$/Mt"
		if got := m.String(); got != want {
			t.Fatalf("String() = %q, want %q", got, want)
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
	t.Run("pricing and pagination", func(t *testing.T) {
		body := `{"data":[{"pricing":{"prompt":"0.000001","completion":"0.000002","audio":"0.000003","audio_output":"0.000004","discount":0.1,"image":"0.000005","image_output":"0.000006","image_token":"0.000007","input_audio_cache":"0.000008","input_cache_read":"0.000009","input_cache_write":"0.000010","input_cache_write_1h":"0.000011","internal_reasoning":"0.000012","overrides":[{"audio":"0.000013","completion":"0.000014","input_audio_cache":"0.000015","input_cache_read":"0.000016","input_cache_write":"0.000017","input_cache_write_1h":"0.000018","min_prompt_tokens":200000,"prompt":"0.000019","utc_start":100,"utc_end":400}],"request":"0.02","web_search":"0.03"}}],"total_count":1,"links":{"next":null}}`
		var resp openrouter.ModelsResponse
		dec := json.NewDecoder(strings.NewReader(body))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&resp); err != nil {
			t.Fatal(err)
		}
		want := openrouter.ModelPricing{
			Prompt:            "0.000001",
			Completion:        "0.000002",
			Audio:             "0.000003",
			AudioOutput:       "0.000004",
			Discount:          0.1,
			Image:             "0.000005",
			ImageOutput:       "0.000006",
			ImageToken:        "0.000007",
			InputAudioCache:   "0.000008",
			InputCacheRead:    "0.000009",
			InputCacheWrite:   "0.000010",
			InputCacheWrite1h: "0.000011",
			InternalReasoning: "0.000012",
			Overrides: []openrouter.ModelPricingOverride{{
				Audio:             "0.000013",
				Completion:        "0.000014",
				InputAudioCache:   "0.000015",
				InputCacheRead:    "0.000016",
				InputCacheWrite:   "0.000017",
				InputCacheWrite1h: "0.000018",
				MinPromptTokens:   200000,
				Prompt:            "0.000019",
				UTCStart:          100,
				UTCEnd:            400,
			}},
			Request:   "0.02",
			WebSearch: "0.03",
		}
		if got := resp.Data[0].Pricing; !reflect.DeepEqual(got, want) {
			t.Fatalf("Pricing = %#v, want %#v", got, want)
		}
		if resp.TotalCount != 1 {
			t.Fatalf("TotalCount = %d, want 1", resp.TotalCount)
		}
		if resp.Links.Next != "" {
			t.Fatalf("Links.Next = %q, want empty", resp.Links.Next)
		}
	})
}
