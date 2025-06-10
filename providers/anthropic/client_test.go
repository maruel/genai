// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	_ "embed"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/anthropic"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen { return getClient(t, m) }, nil)
}

// This is a tricky test since batch operations can take up to 24h to complete.
func TestClient_Batch(t *testing.T) {
	ctx := t.Context()
	// Using an extremely old cheap model that nobody uses helps a lot on reducing the latency, I got it to work
	// within a few minutes.
	c := getClient(t, "claude-3-haiku-20240307")
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Tell a joke in 10 words")}
	job, err := c.GenAsync(ctx, msgs, nil)
	if err != nil {
		t.Fatal(err)
	}
	// TODO: Detect when recording and sleep only in this case.
	is_recording := os.Getenv("RECORD") == "1"
	for {
		res, err := c.PokeResult(ctx, job)
		if err != nil {
			t.Fatal(err)
		}
		if res.FinishReason == genai.Pending {
			if is_recording {
				t.Logf("Waiting...")
				time.Sleep(time.Second)
			}
			continue
		}
		if res.InputTokens == 0 || res.OutputTokens == 0 {
			t.Error("expected usage")
		}
		if res.FinishReason != genai.FinishedStop {
			t.Errorf("finish reason: %s", res.FinishReason)
		}
		if s := res.AsText(); len(s) < 15 {
			t.Errorf("not enough text: %q", s)
		}
		break
	}
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "claude-3-haiku-20240307"},
		{base.PreferredGood, "claude-sonnet-4-20250514"},
		{base.PreferredSOTA, "claude-opus-4-20250514"},
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
			Model:        "claude-3-haiku-20240307",
			ErrGenSync:   "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
			ErrGenStream: "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404: error not_found_error: model: bad model",
			ErrGenStream: "http 404: error not_found_error: model: bad model",
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
			Err:    "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *anthropic.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *anthropic.Client {
	if apiKey == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	wrapper := func(h http.RoundTripper) http.RoundTripper {
		// The Anthropic-Version header was not historically saved in the recordings. So instead of re-creating all the recordings,
		// I'm removing it from the matcher.
		// TODO: Redo all recordings then stop skipping this header.
		m := cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("X-Api-Key", "Anthropic-Version", "X-Request-Id"))
		return testRecorder.Record(t, h, recorder.WithMatcher(m))
	}
	c, err := anthropic.New(apiKey, m, wrapper)
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
