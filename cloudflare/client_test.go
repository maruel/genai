// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

import (
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient:            func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:                "@hf/nousresearch/hermes-2-pro-mistral-7b",
		FinishReasonIsBroken: true,
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
			id := m.GetID()
			// Only test a few models because there are too many.
			return id == "@cf/qwen/qwen2.5-coder-32b-instruct" || id == "@cf/meta/llama-4-scout-17b-16e-instruct"
		})
}

func TestClient_Chat_simple(t *testing.T) {
	testCases.TestChatSimple_simple(t, &internaltest.Settings{Model: "@cf/meta/llama-3.2-3b-instruct"})
}

func TestClient_ChatStream_simple(t *testing.T) {
	testCases.TestChatStream_simple(t, &internaltest.Settings{Model: "@cf/meta/llama-3.2-3b-instruct"})
}

func TestClient_max_tokens(t *testing.T) {
	testCases.TestChatMaxTokens(t, &internaltest.Settings{UsageIsBroken: true})
}

func TestClient_stop_sequence(t *testing.T) {
	t.Skip("Cloudflare doesn't support stop sequences.")
	testCases.TestChatStopSequence(t, nil)
}

func TestClient_Chat_jSON(t *testing.T) {
	testCases.TestChatJSON(t, &internaltest.Settings{UsageIsBroken: true})
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, &internaltest.Settings{UsageIsBroken: true})
}

func TestClient_Chat_tool_use_reply(t *testing.T) {
	t.Skip("To be fixed later.")
	testCases.TestChatToolUseReply(t, &internaltest.Settings{Model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast"})
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	s := &internaltest.Settings{Model: "@cf/meta/llama-3.3-70b-instruct-fp8-fast"}
	t.Run("Chat", func(t *testing.T) {
		testCases.TestChatToolUsePositionBiasCore(t, s, false, false)
	})
	t.Run("ChatStream", func(t *testing.T) {
		t.Skip("cloudflare has broken streaming tool calling")
		testCases.TestChatToolUsePositionBiasCore(t, s, false, true)
	})
}

func TestClient_ChatProvider_errors(t *testing.T) {
	data := []internaltest.ChatProviderError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "@hf/nousresearch/hermes-2-pro-mistral-7b",
			ErrChat:       "http 401: error Authentication error. You can get a new API key at https://dash.cloudflare.com/profile/api-tokens",
			ErrChatStream: "http 401: error Authentication error. You can get a new API key at https://dash.cloudflare.com/profile/api-tokens",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 400: error No route for that URI",
			ErrChatStream: "http 400: error No route for that URI",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ChatProvider {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_ChatProvider_errors(t, f, data)
}

func TestClient_ModelProvider_errors(t *testing.T) {
	data := []internaltest.ModelProviderError{
		{
			Name:   "bad apiKey",
			ApiKey: "badApiKey",
			Err:    "http 400: error Unable to authenticate request",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ModelProvider {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ModelProvider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *cloudflare.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *cloudflare.Client {
	if apiKey == "" && os.Getenv("CLOUDFLARE_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	realAccountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	accountID := ""
	if realAccountID == "" {
		accountID = "INSERT_ACCOUNTID_KEY_HERE"
		realAccountID = accountID
	}
	c, err := cloudflare.New(accountID, apiKey, m, nil)
	if err != nil {
		t.Fatal(err)
	}
	fnMatch := func(r *http.Request, i cassette.Request) bool {
		r = r.Clone(r.Context())
		r.URL.Path = strings.Replace(r.URL.Path, realAccountID, "ACCOUNT_ID", 1)
		return defaultMatcher(r, i)
	}
	fnSave := func(i *cassette.Interaction) error {
		i.Request.URL = strings.Replace(i.Request.URL, realAccountID, "ACCOUNT_ID", 1)
		return nil
	}
	c.ClientJSON.Client.Transport = testRecorder.Record(t, c.ClientJSON.Client.Transport, recorder.WithHook(fnSave, recorder.AfterCaptureHook), recorder.WithMatcher(fnMatch))
	return c
}

var defaultMatcher = cassette.NewDefaultMatcher()

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
