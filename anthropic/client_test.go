// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	_ "embed"
	"os"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen { return getClient(t, m) }, nil)
}

// This is a tricky test since batch operations can take up to 24h to complete.
func TestClient_Batch(t *testing.T) {
	ctx := t.Context()
	c := getClient(t, "claude-3-haiku-20240307")
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Tell a joke in 10 words")}
	job, err := c.GenAsync(ctx, msgs, nil)
	if err != nil {
		t.Fatal(err)
	}
	// TODO: Detect when recording and sleep only in this case.
	is_recording := false
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
	c, err := anthropic.New(apiKey, m, nil)
	if err != nil {
		t.Fatal(err)
	}
	c.ClientJSON.Client.Transport = testRecorder.Record(t, c.ClientJSON.Client.Transport)
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
