// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openai_test

import (
	"context"
	_ "embed"
	"os"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/openai"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderChat {
		c := getClient(t, m)
		if m == "o4-mini" {
			return &injectOption{Client: c, t: t, opts: openai.ChatOptions{
				// This will lead to spurious HTTP 500 but it is 25% of the cost.
				ServiceTier:     openai.ServiceTierFlex,
				ReasoningEffort: openai.ReasoningEffortHigh,
			}}
		}
		return c
	}, nil)
}

type injectOption struct {
	*openai.Client
	t    *testing.T
	opts openai.ChatOptions
}

func (i *injectOption) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	n := i.opts
	if opts != nil {
		n.ChatOptions = *opts.(*genai.ChatOptions)
	}
	opts = &n
	return i.Client.Chat(ctx, msgs, opts)
}

func (i *injectOption) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, replies chan<- genai.MessageFragment) (genai.ChatResult, error) {
	n := i.opts
	if opts != nil {
		n.ChatOptions = *opts.(*genai.ChatOptions)
	}
	opts = &n
	return i.Client.ChatStream(ctx, msgs, opts, replies)
}

func TestClient_ProviderChat_errors(t *testing.T) {
	data := []internaltest.ProviderChatError{
		{
			Name:          "bad apiKey",
			ApiKey:        "bad apiKey",
			Model:         "gpt-4.1-nano",
			ErrChat:       "http 401: error invalid_api_key (): Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			ErrChatStream: "http 401: error invalid_api_key (): Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
		},
		{
			Name:          "bad model",
			Model:         "bad model",
			ErrChat:       "http 400: error invalid_request_error: invalid model ID",
			ErrChatStream: "http 400: error invalid_request_error: invalid model ID",
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
			Err:    "http 401: error invalid_api_key (): Incorrect API key provided: badApiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
		},
	}
	f := func(t *testing.T, apiKey string) genai.ProviderModel {
		return getClientInner(t, apiKey, "")
	}
	internaltest.TestClient_ProviderModel_errors(t, f, data)
}

func getClient(t *testing.T, m string) *openai.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *openai.Client {
	if apiKey == "" && os.Getenv("OPENAI_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := openai.New(apiKey, m, nil)
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

func TestUnsupportedContinuableError(t *testing.T) {
	// Create a request with an unsupported feature (TopK)
	req := &openai.ChatRequest{}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Hello"),
	}
	opts := &genai.ChatOptions{
		TopK: 50, // OpenAI doesn't support TopK
	}

	// Initialize the request
	err := req.Init(msgs, opts, "gpt-4")

	// Check that it returns an UnsupportedContinuableError
	uce, ok := err.(*genai.UnsupportedContinuableError)
	if !ok {
		t.Fatalf("Expected UnsupportedContinuableError, got %T: %v", err, err)
	}

	// Check that the unsupported field is reported
	if len(uce.Unsupported) != 1 || uce.Unsupported[0] != "TopK" {
		t.Errorf("Expected Unsupported=[TopK], got %v", uce.Unsupported)
	}
}
