// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package groq_test

import (
	"context"
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderChat {
		c := getClient(t, m)
		if m == "qwen-qwq-32b" || m == "deepseek-r1-distill-llama-70b" {
			return &handleReasoning{Client: c, t: t}
		}
		return c
	}, nil)
}

type handleReasoning struct {
	*groq.Client
	t *testing.T
}

func (h *handleReasoning) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	if opts != nil {
		if o := opts.(*genai.ChatOptions); len(o.Tools) != 0 || o.DecodeAs != nil || o.ReplyAsJSON {
			opts = &groq.ChatOptions{ReasoningFormat: groq.ReasoningFormatParsed, ChatOptions: *o}
			return h.Client.Chat(ctx, msgs, opts)
		}
	}
	c := genai.ProviderChatThinking{ProviderChat: h.Client, TagName: "think"}
	return c.Chat(ctx, msgs, opts)
}

func (h *handleReasoning) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, replies chan<- genai.MessageFragment) (genai.ChatResult, error) {
	if opts != nil {
		if o := opts.(*genai.ChatOptions); len(o.Tools) != 0 || o.DecodeAs != nil || o.ReplyAsJSON {
			opts = &groq.ChatOptions{ReasoningFormat: groq.ReasoningFormatParsed, ChatOptions: *o}
			return h.Client.ChatStream(ctx, msgs, opts, replies)
		}
	}
	c := genai.ProviderChatThinking{ProviderChat: h.Client, TagName: "think"}
	return c.ChatStream(ctx, msgs, opts, replies)
}

func TestClient_ProviderChat_errors(t *testing.T) {
	data := []internaltest.ProviderChatError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "llama3-8b-8192",
			ErrChat:       "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
			ErrChatStream: "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 404: error model_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
			ErrChatStream: "http 404: error model_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ProviderChat {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_ProviderChat_errors(t, f, data)
}

func TestClient_ProviderModel_errors(t *testing.T) {
	data := []internaltest.ProviderModelError{
		{
			Name:   "bad apiKey",
			ApiKey: "badApiKey",
			Err:    "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *groq.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *groq.Client {
	if apiKey == "" && os.Getenv("GROQ_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := groq.New(apiKey, m, nil)
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
