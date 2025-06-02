// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package pollinations_test

import (
	"context"
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/pollinations"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderChat {
		c := getClient(t, m)
		if m == "flux" || m == "gptimage" || m == "turbo" {
			return &injectOption{Client: c, t: t, opts: pollinations.ChatOptions{Width: 512, Height: 512}}
		}
		return c
	}, nil)
}

type injectOption struct {
	*pollinations.Client
	t    *testing.T
	opts pollinations.ChatOptions
}

func (i *injectOption) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		n.ChatOptions = *opts.(*genai.ChatOptions)
	}
	opts = &n
	return i.Client.Chat(ctx, msgs, opts)
}

func (i *injectOption) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, replies chan<- genai.MessageFragment) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		n.ChatOptions = *opts.(*genai.ChatOptions)
	}
	opts = &n
	return i.Client.ChatStream(ctx, msgs, opts, replies)
}

func TestClient_ProviderChat_errors(t *testing.T) {
	t.Skip("TODO")
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
		return getClientInner(t, model)
	}
	internaltest.TestClient_ProviderChat_errors(t, f, data)
}

func getClient(t *testing.T, m string) *pollinations.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, m)
}

func getClientInner(t *testing.T, m string) *pollinations.Client {
	c, err := pollinations.New("genai-unittests", m, nil)
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
