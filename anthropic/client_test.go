// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		Model:     "claude-3-haiku-20240307",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(t, nil)
}

func TestClient_Chat_thinking(t *testing.T) {
	// https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
	// TODO: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#preserving-thinking-blocks
	testCases.TestChatThinking(t, &internaltest.Settings{
		Model: "claude-3-7-sonnet-20250219",
		Options: func(opts *genai.ChatOptions) genai.Validatable {
			return &anthropic.ChatOptions{ChatOptions: *opts, ThinkingBudget: opts.MaxTokens - 1}
		},
	})
}

func TestClient_Chat_simple(t *testing.T) {
	testCases.TestChatSimple_simple(t, nil)
}

func TestClient_ChatStream_simple(t *testing.T) {
	testCases.TestChatStream_simple(t, nil)
}

func TestClient_max_tokens(t *testing.T) {
	testCases.TestChatMaxTokens(t, nil)
}

func TestClient_stop_sequence(t *testing.T) {
	testCases.TestChatStopSequence(t, nil)
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	// Using very small model for testing. As of March 2025,
	// claude-3-haiku-20240307 is 0.20$/1.25$ while claude-3-5-haiku-20241022 is
	// 0.80$/4.00$. 3.0 supports images, 3.5 supports PDFs.
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	testCases.TestChatVisionJPGInline(t, nil)
}

func TestClient_Chat_vision_pDF_inline(t *testing.T) {
	// 3.0 doesn't support PDFs.
	testCases.TestChatVisionPDFInline(t, &internaltest.Settings{Model: "claude-3-5-haiku-20241022"})
}

func TestClient_Chat_vision_pDF_uRL(t *testing.T) {
	testCases.TestChatVisionPDFURL(t, &internaltest.Settings{Model: "claude-3-5-haiku-20241022"})
}

func TestClient_Chat_tool_use_reply(t *testing.T) {
	testCases.TestChatToolUseReply(t, &internaltest.Settings{Model: "claude-3-5-haiku-20241022"})
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, nil, false)
}

func TestClient_ChatProvider_errors(t *testing.T) {
	data := []internaltest.ChatProviderError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "claude-3-haiku-20240307",
			ErrChat:       "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
			ErrChatStream: "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 404: error not_found_error: model: bad model",
			ErrChatStream: "http 404: error not_found_error: model: bad model",
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
			Err:    "http 401: error authentication_error: invalid x-api-key. You can get a new API key at https://console.anthropic.com/settings/keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ModelProvider {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ModelProvider_errors(t, f, data)
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
	c.ClientBase.ClientJSON.Client.Transport = testRecorder.Record(t, c.ClientBase.ClientJSON.Client.Transport)
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
