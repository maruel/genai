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
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
)

// BeLenient is used by all clients to enable or disable httpjson.Client.Lenient.
//
// It is true by default. Tests must manually set it to false.
var BeLenient = true

// ClientBase implements the shared HTTP client functionality used across all API clients.
type ClientBase[E any] struct {
	// ClientJSON is exported for testing replay purposes.
	ClientJSON httpjson.Client
	// APIKeyURL is the URL to present to the user upon authentication error.
	APIKeyURL string
}

// DoRequest performs an HTTP request and handles error responses.
//
// It takes care of sending the request, decoding the response, and handling errors.
// All API clients should use this method for their HTTP communication needs.
func (c *ClientBase[E]) DoRequest(ctx context.Context, method, url string, in, out any) error {
	resp, err := c.ClientJSON.Request(ctx, method, url, nil, in)
	if err != nil {
		return err
	}
	var er E
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				// Check if the error message already contains an API key URL
				errorMsg := fmt.Sprintf("%v", &er)
				if !strings.Contains(errorMsg, "API key") || !strings.Contains(errorMsg, "http") {
					return fmt.Errorf("%w: %v. You can get a new API key at %s", herr, errorMsg, c.APIKeyURL)
				}
				return fmt.Errorf("%w: %s", herr, errorMsg)
			}
			return fmt.Errorf("%w: %v", herr, &er)
		}
		return fmt.Errorf("%v", &er)
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
func (c *ClientBase[E]) DecodeError(ctx context.Context, url string, resp *http.Response) error {
	var er E
	switch i, err := httpjson.DecodeResponse(resp, &er); i {
	case 0:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			herr.PrintBody = false
			if c.APIKeyURL != "" && herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %v. You can get a new API key at %s", herr, &er, c.APIKeyURL)
			}
			return fmt.Errorf("%w: %v", herr, &er)
		}
		return fmt.Errorf("%v", &er)
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

// ChatStreamRequest is an interface for request types used in ChatStreamRaw methods
type ChatStreamRequest interface{ any }

// InitializableRequest is an interface for request types that can be initialized
type InitializableRequest interface {
	// Init initializes the request with messages, options, and model
	Init(msgs genai.Messages, opts genai.Validatable, model string) error
}

// ChatStreamResponse is an interface for response types used in ChatStreamRaw methods
type ChatStreamResponse interface{ any }

// ChatStreamRawFunc is a function type for client-specific ChatStreamRaw methods
type ChatStreamRawFunc[TRequest ChatStreamRequest, TResponse ChatStreamResponse] func(ctx context.Context, in *TRequest, out chan<- TResponse) error

// ProcessStreamPacketsFunc is a function type for client-specific processStreamPackets methods
type ProcessStreamPacketsFunc[TResponse any] func(ch <-chan TResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error

// ChatStream is a generic function that implements the common pattern of ChatStream methods across providers.
// It is meant to be called by client-specific ChatStream methods to avoid code duplication.
func ChatStream[TRequest ChatStreamRequest, TResponse ChatStreamResponse](
	ctx context.Context,
	msgs genai.Messages,
	opts genai.Validatable,
	chunks chan<- genai.MessageFragment,
	model string,
	chatStreamRaw ChatStreamRawFunc[TRequest, TResponse],
	processStreamPackets ProcessStreamPacketsFunc[TResponse],
	allowOpaqueFields bool,
) (genai.ChatResult, error) {
	result := genai.ChatResult{}
	// Check for non-empty Opaque field unless explicitly allowed
	if !allowOpaqueFields {
		for i, msg := range msgs {
			for j, content := range msg.Contents {
				if len(content.Opaque) != 0 {
					return result, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
				}
			}
		}
	}

	in := new(TRequest)
	var continuableErr error
	if err := any(in).(InitializableRequest).Init(msgs, opts, model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	ch := make(chan TResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks, &result)
	})
	err := chatStreamRaw(ctx, in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	if err != nil {
		return result, err
	}
	return result, continuableErr
}
