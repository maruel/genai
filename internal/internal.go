// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal is awesome sauce.
package internal

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/sse"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
)

// BeLenient is used by all clients to enable or disable httpjson.Client.Lenient.
//
// It is true by default. Tests must manually set it to false.
var BeLenient = true

//

// ClientBase implements the shared HTTP client functionality used across all API clients.
type ClientBase[Err fmt.Stringer] struct {
	// ClientJSON is exported for testing replay purposes.
	ClientJSON httpjson.Client
	// APIKeyURL is the URL to present to the user upon authentication error.
	APIKeyURL string
}

// DoRequest performs an HTTP request and handles error responses.
//
// It takes care of sending the request, decoding the response, and handling errors.
// All API clients should use this method for their HTTP communication needs.
func (c *ClientBase[Err]) DoRequest(ctx context.Context, method, url string, in, out any) error {
	resp, err := c.ClientJSON.Request(ctx, method, url, nil, in)
	if err != nil {
		return err
	}
	var er Err
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				// Check if the error message already contains an API key URL
				errorMsg := fmt.Sprintf("%s", er)
				if !strings.Contains(errorMsg, "API key") || !strings.Contains(errorMsg, "http") {
					return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, errorMsg, c.APIKeyURL)
				}
				return fmt.Errorf("%w: %s", herr, errorMsg)
			}
			return fmt.Errorf("%w: %s", herr, er)
		}
		return errors.New(er.String())
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, http.StatusText(herr.StatusCode), c.APIKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, http.StatusText(herr.StatusCode))
		}
		return err
	}
}

// DecodeError handles HTTP error responses from API calls.
//
// It handles JSON decoding of error responses and provides appropriate error messages
// with context such as API key URLs for unauthorized errors.
func (c *ClientBase[Err]) DecodeError(ctx context.Context, url string, resp *http.Response) error {
	var er Err
	switch i, err := httpjson.DecodeResponse(resp, &er); i {
	case 0:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, er, c.APIKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, er)
		}
		return errors.New(er.String())
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s. You can get a new API key at %s", herr, http.StatusText(herr.StatusCode), c.APIKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, http.StatusText(herr.StatusCode))
		}
		return err
	}
}

//

// Obj is a generic interface for chat-related types (both requests and responses)
type Obj interface{ any }

// InitializableRequest is an interface for request types that can be initialized.
type InitializableRequest interface {
	// Init initializes the request with messages, options, and model.
	Init(msgs genai.Messages, opts genai.Validatable, model string) error
	// SetStream set the stream mode.
	SetStream(bool)
}

// ResultConverter converts a provider-specific result to a genai.ChatResult.
type ResultConverter interface {
	ToResult() (genai.ChatResult, error)
}

//

// ClientChat implements common functionality for clients that provide chat capabilities.
// It embeds ClientBase and adds a Model field and common chat methods.
type ClientChat[Err fmt.Stringer, ChatRequest InitializableRequest, ChatResponse ResultConverter, StreamChunk Obj] struct {
	ClientBase[Err]
	// Model is the default model used for chat requests
	Model string
	// ChatURL is the endpoint URL for chat API requests
	ChatURL string
	// ChatStreamURL is the endpoint URL for chat stream API requests. It defaults to ChatURL if unset.
	ChatStreamURL        string
	AllowOpaqueFields    bool
	ProcessStreamPackets func(ch <-chan StreamChunk, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error

	mu           sync.Mutex
	chatRequest  reflect.Type
	chatResponse reflect.Type
}

func (c *ClientChat[Err, ChatRequest, ChatResponse, StreamChunk]) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	result := genai.ChatResult{}
	// Check for non-empty Opaque field unless explicitly allowed
	if !c.AllowOpaqueFields {
		for i, msg := range msgs {
			for j, content := range msg.Contents {
				if len(content.Opaque) != 0 {
					return result, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
				}
			}
		}
	}

	c.lateInit()
	in := reflect.New(c.chatRequest).Interface().(ChatRequest)
	var continuableErr error
	if err := in.Init(msgs, opts, c.Model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	out := reflect.New(c.chatResponse).Interface().(ChatResponse)
	if err := c.ChatRaw(ctx, in, out); err != nil {
		return result, err
	}
	result, err := out.ToResult()
	if err != nil {
		return result, err
	}
	return result, continuableErr
}

func (c *ClientChat[Err, ChatRequest, ChatResponse, StreamChunk]) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.ChatResult, error) {
	result := genai.ChatResult{}
	// Check for non-empty Opaque field unless explicitly allowed
	if !c.AllowOpaqueFields {
		for i, msg := range msgs {
			for j, content := range msg.Contents {
				if len(content.Opaque) != 0 {
					return result, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
				}
			}
		}
	}

	c.lateInit()
	in := reflect.New(c.chatRequest).Interface().(ChatRequest)
	var continuableErr error
	if err := in.Init(msgs, opts, c.Model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	ch := make(chan StreamChunk)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return c.ProcessStreamPackets(ch, chunks, &result)
	})
	err := c.ChatStreamRaw(ctx, in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	if err != nil {
		return result, err
	}
	return result, continuableErr
}

// ChatRaw is the generic raw implementation for the Chat API endpoint.
// It sets Stream to false and sends a request to the chat URL.
func (c *ClientChat[Err, ChatRequest, ChatResponse, StreamChunk]) ChatRaw(ctx context.Context, in ChatRequest, out ChatResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.SetStream(false)
	return c.DoRequest(ctx, "POST", c.ChatURL, in, out)
}

// ChatStreamRaw is the generic raw implementation for streaming Chat API endpoints.
// It sets Stream to true, enables stream options if available, and handles the SSE response.
func (c *ClientChat[Err, ChatRequest, ChatResponse, StreamChunk]) ChatStreamRaw(ctx context.Context, in ChatRequest, out chan<- StreamChunk) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.SetStream(true)
	url := c.ChatStreamURL
	if url == "" {
		url = c.ChatURL
	}
	resp, err := c.ClientJSON.Request(ctx, "POST", url, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.DecodeError(ctx, url, resp)
	}
	var er Err
	return sse.Process(resp.Body, out, &er)
}

func (c *ClientChat[Err, ChatRequest, ChatResponse, StreamChunk]) Validate() error {
	if c.Model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *ClientChat[Err, ChatRequest, ChatResponse, StreamChunk]) lateInit() {
	// TODO: Figure out how to not use reflection.
	c.mu.Lock()
	if c.chatRequest == nil {
		var in ChatRequest
		c.chatRequest = reflect.TypeOf(in).Elem()
		var out ChatResponse
		c.chatResponse = reflect.TypeOf(out).Elem()
	}
	c.mu.Unlock()
}

//

// ListModelsResponse is an interface for responses that contain model data.
type ListModelsResponse interface {
	// ToModels converts the provider-specific models to a slice of genai.Model
	ToModels() []genai.Model
}

// ListModels is a generic function that implements the common pattern for listing models across providers.
// It makes an HTTP GET request to the specified URL and converts the response to a slice of genai.Model.
func ListModels[Err fmt.Stringer, R ListModelsResponse](ctx context.Context, c *ClientBase[Err], url string) ([]genai.Model, error) {
	var resp R
	if err := c.DoRequest(ctx, "GET", url, nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}
