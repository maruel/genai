// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/huggingface"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen {
		c := getClient(t, m)
		if m == "Qwen/QwQ-32B" {
			return &adapters.ProviderGenThinking{ProviderGen: c, TagName: "think", SkipJSON: true}
		}
		return c
	}, nil)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "meta-llama/Llama-3.2-1B-Instruct"},
		{base.PreferredGood, "Qwen/Qwen3-235B-A22B"},
		{base.PreferredSOTA, "deepseek-ai/DeepSeek-R1-0528"},
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
			Model:        "Qwen/Qwen3-4B",
			ErrGenSync:   "http 401: error Invalid credentials in Authorization header. You can get a new API key at https://huggingface.co/settings/tokens",
			ErrGenStream: "http 401: error Invalid credentials in Authorization header. You can get a new API key at https://huggingface.co/settings/tokens",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404: Not Found",
			ErrGenStream: "http 404: Not Found",
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
			Err:    "http 401: error Invalid credentials in Authorization header. You can get a new API key at https://huggingface.co/settings/tokens",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *huggingface.Client {
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *huggingface.Client {
	if apiKey == "" && os.Getenv("HUGGINGFACE_API_KEY") == "" {
		// Fallback to loading from the python client's cache.
		h, err := os.UserHomeDir()
		if err != nil {
			t.Fatal("can't find home directory")
		}
		if _, err := os.Stat(filepath.Join(h, ".cache", "huggingface", "token")); err != nil {
			apiKey = "<insert_api_key_here>"
		}
	}
	c, err := huggingface.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
