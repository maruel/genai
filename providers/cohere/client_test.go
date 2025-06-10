// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cohere"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen { return getClient(t, m) }, nil)
}

func TestClient_Citations(t *testing.T) {
	// https://docs.cohere.com/v2/docs/rag-citations
	c := getClient(t, "command-r7b-12-2024")
	msgs := genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{
					Document: strings.NewReader("The capital of Quackiland is Quack. The Big Canard Statue is located in Quack."),
					Filename: "capital_info.txt",
				},
				{
					Text: "What is the capital of Quackiland?",
				},
			},
		},
	}
	res, err := c.GenSync(t.Context(), msgs, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.FinishReason != genai.FinishedStop {
		t.Errorf("finish reason: %s", res.FinishReason)
	}
	t.Logf("Usage: %d input tokens, %d output tokens", res.InputTokens, res.OutputTokens)
	t.Logf("Text: %q", res.AsText())
	if s := res.AsText(); !strings.Contains(s, "Quackiland") {
		t.Errorf("expected Quackiland, got: %q", s)
	}
	foundCitations := false
	for _, content := range res.Contents {
		if len(content.Citations) > 0 {
			foundCitations = true
			t.Logf("Found %d citations in content", len(content.Citations))
			for i, citation := range content.Citations {
				t.Logf("Citation %d: text=%q, type=%q, start=%d, end=%d", i, citation.Text, citation.Type, citation.StartIndex, citation.EndIndex)
				if len(citation.Sources) > 0 {
					t.Logf("  Sources: %+v", citation.Sources)
				}
			}
		}
	}
	if !foundCitations {
		t.Errorf("expected citations in response, but found none")
	}
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "command-light"},
		{base.PreferredGood, "command-r7b-12-2024"},
		{base.PreferredSOTA, "command-a-03-2025"},
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
			Model:        "command-r7b-12-2024",
			ErrGenSync:   "http 401: error invalid api token. You can get a new API key at https://dashboard.cohere.com/api-keys",
			ErrGenStream: "http 401: error invalid api token. You can get a new API key at https://dashboard.cohere.com/api-keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404: error model 'bad model' not found, make sure the correct model ID was used and that you have access to the model.",
			ErrGenStream: "http 404: error model 'bad model' not found, make sure the correct model ID was used and that you have access to the model.",
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
			Err:    "http 401: error invalid api token. You can get a new API key at https://dashboard.cohere.com/api-keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *cohere.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *cohere.Client {
	if apiKey == "" && os.Getenv("COHERE_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := cohere.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
