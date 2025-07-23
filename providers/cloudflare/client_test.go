// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

import (
	"net/http"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cloudflare"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen { return getClient(t, m) }, nil)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "@cf/meta/llama-3.2-1b-instruct"},
		{base.PreferredGood, "@cf/meta/llama-3.3-70b-instruct-fp8-fast"},
		{base.PreferredSOTA, "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"},
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
			Model:        "@hf/nousresearch/hermes-2-pro-mistral-7b",
			ErrGenSync:   "http 401: error Authentication error. You can get a new API key at https://dash.cloudflare.com/profile/api-tokens",
			ErrGenStream: "http 401: error Authentication error. You can get a new API key at https://dash.cloudflare.com/profile/api-tokens",
			ErrListModel: "http 400: error Unable to authenticate request",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 400: error No route for that URI",
			ErrGenStream: "http 400: error No route for that URI",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.Provider {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *cloudflare.Client {
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *cloudflare.Client {
	if apiKey == "" && os.Getenv("CLOUDFLARE_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if accountID == "" {
		accountID = "ACCOUNT_ID"
	}
	wrapper := func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.Record(t, h, recorder.WithHook(trimRecordingInternal, recorder.AfterCaptureHook), recorder.WithMatcher(matchCassetteInternal))
	}
	c, err := cloudflare.New(accountID, apiKey, m, wrapper)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// trimRecording trims API key and noise from the recording.
func trimRecordingInternal(i *cassette.Interaction) error {
	// Zap the account ID from the URL path before saving.
	i.Request.URL = reAccount.ReplaceAllString(i.Request.URL, "/accounts/ACCOUNT_ID/")
	return nil
}

func matchCassetteInternal(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	// When matching, ignore the account ID from the URL path.
	r.URL.Path = reAccount.ReplaceAllString(r.URL.Path, "/accounts/ACCOUNT_ID/")
	return internal.DefaultMatcher(r, i)
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
