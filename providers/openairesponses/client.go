// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openairesponses implements a client for the OpenAI Responses API.
//
// It is described at https://platform.openai.com/docs/api-reference/responses/create
package openairesponses

// See official client at https://github.com/openai/openai-go

import (
	"fmt"
	"net/http"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Client is a client for the OpenAI Responses API.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ResponseRequest, *ResponseResponse, ResponseStreamChunkResponse]
}

// ErrorResponse represents an error response from the OpenAI API.
type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
	} `json:"error"`
}

func (e *ErrorResponse) String() string {
	return fmt.Sprintf("openai responses error: %s (type: %s, code: %s)", e.Error.Message, e.Error.Type, e.Error.Code)
}

// New creates a new client to talk to the OpenAI Responses API.
//
// If apiKey is not provided, it tries to load it from the OPENAI_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://platform.openai.com/settings/organization/api-keys
//
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://platform.openai.com/docs/models
//
// Pass model base.PreferredCheap to use a good cheap model, base.PreferredGood for a good model or
// base.PreferredSOTA for a state-of-the-art model.
func New(apiKey, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "OPENAI_API_KEY", URL: apiKeyURL}
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ResponseRequest, *ResponseResponse, ResponseStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.openai.com/v1/responses",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "openairesponses",
				APIKeyURL:    "", // OpenAI error message prints the api key URL already.
				ClientJSON: httpjson.Client{
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
							Transport: &roundtrippers.RequestID{Transport: t},
						},
					},
				},
			},
		},
	}
	return c, err
}

// processStreamPackets processes stream packets for the OpenAI Responses API.
// This is a placeholder - will be implemented when GenStream is added.
func processStreamPackets(ch <-chan ResponseStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	// TODO: Implement when GenStream support is added
	return fmt.Errorf("streaming not yet implemented for OpenAI Responses API")
}

// ResponseStreamChunkResponse represents a streaming response chunk.
// This is a placeholder - will be implemented when GenStream is added.
type ResponseStreamChunkResponse struct {
	// TODO: Implement when GenStream support is added
}

// ResponseRequest represents a request to the OpenAI Responses API.
type ResponseRequest struct {
	Model  string `json:"model"`
	Stream bool   `json:"stream,omitempty"`
}

// Init implements base.InitializableRequest.
func (r *ResponseRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	r.Model = model
	// TODO: Implement request initialization from messages and options
	return nil
}

// SetStream implements base.InitializableRequest.
func (r *ResponseRequest) SetStream(stream bool) {
	r.Stream = stream
}

// ResponseResponse represents a response from the OpenAI Responses API.
type ResponseResponse struct {
	ID        string `json:"id"`
	CreatedAt int64  `json:"created_at"`
}

// ToResult implements base.ResultConverter.
func (r *ResponseResponse) ToResult() (genai.Result, error) {
	// TODO: Implement response conversion
	return genai.Result{}, nil
}
