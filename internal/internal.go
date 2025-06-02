// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal is awesome sauce.
package internal

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
type ClientBase[PErrorResponse fmt.Stringer] struct {
	// ClientJSON is exported for testing replay purposes.
	ClientJSON httpjson.Client
	// APIKeyURL is the URL to present to the user upon authentication error.
	APIKeyURL string

	mu            sync.Mutex
	errorResponse reflect.Type
}

// DoRequest performs an HTTP request and handles error responses.
//
// It takes care of sending the request, decoding the response, and handling errors.
// All API clients should use this method for their HTTP communication needs.
func (c *ClientBase[PErrorResponse]) DoRequest(ctx context.Context, method, url string, in, out any) error {
	c.lateInit()
	resp, err := c.ClientJSON.Request(ctx, method, url, nil, in)
	if err != nil {
		return err
	}
	if resp.StatusCode != 200 {
		return c.DecodeError(ctx, url, resp)
	}
	b, err := io.ReadAll(resp.Body)
	if err2 := resp.Body.Close(); err == nil {
		err = err2
	}
	if err != nil {
		return err
	}
	// It's an HTTP 200, it generally should be a success.
	r := bytes.NewReader(b)
	var r2 io.ReadSeeker
	d := json.NewDecoder(r)
	if !c.ClientJSON.Lenient {
		d.DisallowUnknownFields()
		r2 = r
	}
	var errs []error
	if foundExtraKeys, err2 := decodeJSON(d, out, r2); err2 == nil {
		// It may have succeeded but not decoded anything.
		if v := reflect.ValueOf(out); !reflect.DeepEqual(out, reflect.Zero(v.Type()).Interface()) {
			return nil
		}
	} else if foundExtraKeys {
		errs = append(errs, err2)
	}
	if _, err = r.Seek(0, 0); err != nil {
		return err
	}
	d = json.NewDecoder(r)
	if !c.ClientJSON.Lenient {
		d.DisallowUnknownFields()
		r2 = r
	}
	er := reflect.New(c.errorResponse).Interface().(PErrorResponse)
	if foundExtraKeys, err := decodeJSON(d, er, r2); err == nil {
		// It may have succeeded but not decoded anything.
		if v := reflect.ValueOf(er); !reflect.DeepEqual(out, reflect.Zero(v.Type()).Interface()) {
			return nil
		}
	} else if foundExtraKeys {
		// This is confusing, not sure it's a good idea. The problem is that we need to detect when error fields
		// appear too!
		errs = append(errs, err)
	} else {
		return err
	}
	return errors.Join(errs...)
}

// DecodeError handles HTTP error responses from API calls.
//
// It handles JSON decoding of error responses and provides appropriate error messages
// with context such as API key URLs for unauthorized errors.
func (c *ClientBase[PErrorResponse]) DecodeError(ctx context.Context, url string, resp *http.Response) error {
	c.lateInit()
	// When we are in lenient mode, we do not want to buffer the result. When in strict mode, we want to buffer
	// to give more details.
	er := reflect.New(c.errorResponse).Interface().(PErrorResponse)
	b, err := io.ReadAll(resp.Body)
	if err2 := resp.Body.Close(); err == nil {
		err = err2
	}
	if err != nil {
		return err
	}
	r := bytes.NewReader(b)
	d := json.NewDecoder(r)
	var r2 io.ReadSeeker
	if !c.ClientJSON.Lenient {
		d.DisallowUnknownFields()
		r2 = r
	}
	if foundExtraKeys, err := decodeJSON(d, er, r2); err == nil {
		if c.APIKeyURL != "" && resp.StatusCode == http.StatusUnauthorized {
			if s := er.String(); !strings.Contains(s, c.APIKeyURL) {
				return fmt.Errorf("http %d: %s. You can get a new API key at %s", resp.StatusCode, s, c.APIKeyURL)
			}
		}
		return fmt.Errorf("http %d: %s", resp.StatusCode, er)
	} else if foundExtraKeys {
		// In strict mode, return the decoding error instead.
		return fmt.Errorf("http %d: %w", resp.StatusCode, err)
	}
	if c.APIKeyURL != "" && resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("http %d: %s. You can get a new API key at %s", resp.StatusCode, http.StatusText(resp.StatusCode), c.APIKeyURL)
	}
	return fmt.Errorf("http %d: %s", resp.StatusCode, http.StatusText(resp.StatusCode))
}

func (c *ClientBase[PErrorResponse]) lateInit() {
	// TODO: Figure out how to not use reflection.
	c.mu.Lock()
	if c.errorResponse == nil {
		var in PErrorResponse
		c.errorResponse = reflect.TypeOf(in).Elem()
	}
	c.mu.Unlock()
}

// Duplicate from httpjson.go in https://github.com/maruel/httpjson.
func decodeJSON(d *json.Decoder, out any, r io.ReadSeeker) (bool, error) {
	d.UseNumber()
	if err := d.Decode(out); err != nil {
		// decode.object() in encoding/json.go does not return a structured error
		// when an unknown field is found or when the type is wrong. Process it manually.
		if r != nil {
			if s := err.Error(); strings.Contains(s, "json: unknown field ") || strings.Contains(s, "json: cannot unmarshal ") {
				// Decode again but this time capture all errors. Try first as a map (JSON object), then as a slice
				// (JSON list).
				for _, t := range []any{map[string]any{}, []any{}} {
					if _, err2 := r.Seek(0, 0); err2 != nil {
						// Unexpected.
						return false, err2
					}
					d = json.NewDecoder(r)
					d.UseNumber()
					if err2 := d.Decode(&t); err2 == nil {
						if err2 = errors.Join(httpjson.FindExtraKeys(reflect.TypeOf(out), t)...); err2 != nil {
							return true, err2
						}
					}
				}
			}
		}
		return false, err
	}
	return false, nil
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
	ToResult() (genai.Result, error)
}

//

// ClientChat implements common functionality for clients that provide chat capabilities.
// It embeds ClientBase and adds a Model field and common chat methods.
type ClientChat[PErrorResponse fmt.Stringer, PChatRequest InitializableRequest, PChatResponse ResultConverter, ChatStreamChunkResponse Obj] struct {
	ClientBase[PErrorResponse]
	// Model is the default model used for chat requests
	Model string
	// ChatURL is the endpoint URL for chat API requests
	ChatURL string
	// ChatStreamURL is the endpoint URL for chat stream API requests. It defaults to ChatURL if unset.
	ChatStreamURL string
	// ModelOptional is true if a model name is not required to use the provider.
	ModelOptional bool
	// AllowOpaqueFields is true if the client allows the Opaque field in messages.
	AllowOpaqueFields bool
	// ProcessStreamPackets is the function that processes stream packets used by ChatStream.
	ProcessStreamPackets func(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.Result) error
	// LieToolCalls lie the FinishReason on tool calls.
	LieToolCalls bool

	// Protected by ClientBase.mu.
	chatRequest  reflect.Type
	chatResponse reflect.Type
}

func (c *ClientChat[PErrorResponse, PChatRequest, PChatResponse, ChatStreamChunkResponse]) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.Result, error) {
	result := genai.Result{}
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
	in := reflect.New(c.chatRequest).Interface().(PChatRequest)
	var continuableErr error
	if err := in.Init(msgs, opts, c.Model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	out := reflect.New(c.chatResponse).Interface().(PChatResponse)
	if err := c.ChatRaw(ctx, in, out); err != nil {
		return result, err
	}
	result, err := out.ToResult()
	if err != nil {
		return result, err
	}
	return result, continuableErr
}

func (c *ClientChat[PErrorResponse, PChatRequest, PChatResponse, ChatStreamChunkResponse]) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.Result, error) {
	result := genai.Result{}
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
	in := reflect.New(c.chatRequest).Interface().(PChatRequest)
	var continuableErr error
	if err := in.Init(msgs, opts, c.Model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	ch := make(chan ChatStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return c.ProcessStreamPackets(ch, chunks, &result)
	})
	err := c.ChatStreamRaw(ctx, in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	if c.LieToolCalls && len(result.ToolCalls) != 0 && result.FinishReason == genai.FinishedStop {
		// Lie for the benefit of everyone.
		result.FinishReason = genai.FinishedToolCalls
	}
	if err != nil {
		return result, err
	}
	return result, continuableErr
}

// ChatRaw is the generic raw implementation for the Chat API endpoint.
// It sets Stream to false and sends a request to the chat URL.
func (c *ClientChat[PErrorResponse, PChatRequest, PChatResponse, ChatStreamChunkResponse]) ChatRaw(ctx context.Context, in PChatRequest, out PChatResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.SetStream(false)
	return c.DoRequest(ctx, "POST", c.ChatURL, in, out)
}

// ChatStreamRaw is the generic raw implementation for streaming Chat API endpoints.
// It sets Stream to true, enables stream options if available, and handles the SSE response.
func (c *ClientChat[PErrorResponse, PChatRequest, PChatResponse, ChatStreamChunkResponse]) ChatStreamRaw(ctx context.Context, in PChatRequest, out chan<- ChatStreamChunkResponse) error {
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
	er := reflect.New(c.errorResponse).Interface().(PErrorResponse)
	return sse.Process(resp.Body, out, er, c.ClientJSON.Lenient)
}

func (c *ClientChat[PErrorResponse, PChatRequest, PChatResponse, ChatStreamChunkResponse]) ModelID() string {
	return c.Model
}

func (c *ClientChat[PErrorResponse, PChatRequest, PChatResponse, ChatStreamChunkResponse]) Validate() error {
	if !c.ModelOptional && c.Model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *ClientChat[PErrorResponse, PChatRequest, PChatResponse, ChatStreamChunkResponse]) lateInit() {
	// TODO: Figure out how to not use reflection.
	c.ClientBase.lateInit()
	c.mu.Lock()
	if c.chatRequest == nil {
		var in PChatRequest
		c.chatRequest = reflect.TypeOf(in).Elem()
		var out PChatResponse
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
func ListModels[PErrorResponse fmt.Stringer, R ListModelsResponse](ctx context.Context, c *ClientBase[PErrorResponse], url string) ([]genai.Model, error) {
	var resp R
	if err := c.DoRequest(ctx, "GET", url, nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}
