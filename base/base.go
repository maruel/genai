// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package base is awesome sauce to reduce code duplication across most providers.
package base

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/internal/sse"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// DefaultTransport integrates HTTP retries.
//
// It uses a quite long retry count. If latency matters for you, you may want to use a shorter retry policy.
// Do this by passing the `wrapper` argument in the `New()` function and ignore the `http.RoundTripper` passed
// in.
var DefaultTransport http.RoundTripper = &roundtrippers.Retry{
	Transport: http.DefaultTransport,
	Policy: &roundtrippers.ExponentialBackoff{
		MaxTryCount: 10,
		MaxDuration: 60 * time.Second,
		Exp:         1.5,
	},
}

const (
	// NoModel explicitly tells the providers' New() function to not automatically select a model. The use case
	// is when the only intended call is ListModel(), thus there's no point into selecting a model automatically.
	NoModel = "NO_MODEL"

	// PreferredCheap is a marker for providers' New() function to tell it to use the cheapest model it can find.
	PreferredCheap = "PREFERRED_CHEAP"

	// PreferredGood is a marker for providers' New() function to tell it to use a good every day model.
	PreferredGood = "PREFERRED_GOOD"

	// PreferredSOTA is a marker for providers' New() function to tell it to use the best state-of-the-art model
	// is can find.
	PreferredSOTA = "PREFERRED_SOTA"
)

// ErrAPIKeyRequired is returned by the providers New() function when no key was found.
type ErrAPIKeyRequired struct {
	EnvVar string
	URL    string
}

func (e *ErrAPIKeyRequired) Error() string {
	if e.URL != "" {
		return fmt.Sprintf("api key is required; set environment variable %s to it, and get a key at %s", e.EnvVar, e.URL)
	}
	return fmt.Sprintf("api key is required; set environment variable %s to it", e.EnvVar)
}

// ErrAPI is returned when the API returns a structured error.
type ErrAPI interface {
	error
	IsAPIError() bool
}

// Provider implements genai.Provider (except for ModelID()).
//
// It contains the shared HTTP client functionality used across all API clients.
type Provider[PErrorResponse ErrAPI] struct {
	// ClientJSON is exported for testing replay purposes.
	ClientJSON httpjson.Client
	// APIKeyURL is the URL to present to the user upon authentication error.
	APIKeyURL    string
	ProviderName string

	mu            sync.Mutex
	errorResponse reflect.Type
}

func (c *Provider[PErrorResponse]) Name() string {
	return c.ProviderName
}

// DoRequest performs an HTTP request and handles error responses.
//
// It takes care of sending the request, decoding the response, and handling errors.
// All API clients should use this method for their HTTP communication needs.
func (c *Provider[PErrorResponse]) DoRequest(ctx context.Context, method, url string, in, out any) error {
	c.lateInit()
	resp, err := c.ClientJSON.Request(ctx, method, url, nil, in)
	if err != nil {
		return err
	}
	return c.DecodeResponse(resp, url, out)
}

func (c *Provider[PErrorResponse]) DecodeResponse(resp *http.Response, url string, out any) error {
	if resp.StatusCode != 200 {
		return c.DecodeError(url, resp)
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
		errs = append(errs, er)
	} else if foundExtraKeys {
		// This is confusing, not sure it's a good idea. The problem is that we need to detect when error fields
		// appear too!
		errs = append(errs, err)
	} else {
		// Return only the original error.
		return err
	}
	return errors.Join(errs...)
}

// DecodeError handles HTTP error responses from API calls.
//
// It handles JSON decoding of error responses and provides appropriate error messages
// with context such as API key URLs for unauthorized errors.
func (c *Provider[PErrorResponse]) DecodeError(url string, resp *http.Response) error {
	c.lateInit()
	// When we are in lenient mode, we do not want to buffer the result. When in strict mode, we want to buffer
	// to give more details.
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
	var errs []error
	herr := &httpjson.Error{StatusCode: resp.StatusCode, ResponseBody: b}
	er := reflect.New(c.errorResponse).Interface().(PErrorResponse)
	if foundExtraKeys, err := decodeJSON(d, er, r2); err == nil {
		errs = append(errs, herr, er)
	} else if foundExtraKeys {
		// In strict mode, return the decoding error instead.
		errs = append(errs, err)
	} else {
		errs = append(errs, herr)
	}
	if c.APIKeyURL != "" && resp.StatusCode == http.StatusUnauthorized && !strings.Contains(er.Error(), c.APIKeyURL) {
		errs = append(errs, fmt.Errorf("get a new API key at %s", c.APIKeyURL))
	}
	return errors.Join(errs...)
}

func (c *Provider[PErrorResponse]) lateInit() {
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
	Init(msgs genai.Messages, opts genai.Options, model string) error
	// SetStream set the stream mode.
	SetStream(bool)
}

// ResultConverter converts a provider-specific result to a genai.Result.
type ResultConverter interface {
	ToResult() (genai.Result, error)
}

//

// ProviderGen implements genai.ProviderGen.
//
// It includes common functionality for clients that provide chat capabilities.
// It embeds Provider and adds the missing ModelID() method and implement genai.Validatable.
//
// It only accepts text modality.
type ProviderGen[PErrorResponse ErrAPI, PGenRequest InitializableRequest, PGenResponse ResultConverter, GenStreamChunkResponse Obj] struct {
	Provider[PErrorResponse]
	// Model is the default model used for chat requests
	Model string
	// GenSyncURL is the endpoint URL for chat API requests
	GenSyncURL string
	// GenStreamURL is the endpoint URL for chat stream API requests. It defaults to GenURL if unset.
	GenStreamURL string
	// ModelOptional is true if a model name is not required to use the provider.
	ModelOptional bool
	// AllowOpaqueFields is true if the client allows the Opaque field in messages.
	AllowOpaqueFields bool
	// ProcessStreamPackets is the function that processes stream packets used by GenStream.
	ProcessStreamPackets func(ch <-chan GenStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error
	// LieToolCalls lie the FinishReason on tool calls.
	LieToolCalls bool

	// Protected by Base.mu.
	chatRequest  reflect.Type
	chatResponse reflect.Type
}

func (c *ProviderGen[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	result := genai.Result{}
	if err := msgs.Validate(); err != nil {
		return result, err
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return result, err
		}
		// TODO: Specify as a field.
		if supported := opts.Modalities(); !slices.Contains(supported, genai.ModalityText) {
			return result, fmt.Errorf("modality %s not supported", supported)
		}
	}
	// Check for non-empty Opaque field unless explicitly allowed
	if !c.AllowOpaqueFields {
		for i, msg := range msgs {
			for j, content := range msg.Contents {
				if len(content.Opaque) != 0 {
					return result, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
				}
			}
			for j, tool := range msg.ToolCalls {
				if len(tool.Opaque) != 0 {
					return result, fmt.Errorf("message #%d tool call #%d: field Opaque not supported", i, j)
				}
			}
		}
	}

	c.lateInit()
	in := reflect.New(c.chatRequest).Interface().(PGenRequest)
	var continuableErr error
	if err := in.Init(msgs, opts, c.Model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	out := reflect.New(c.chatResponse).Interface().(PGenResponse)
	if err := c.GenSyncRaw(ctx, in, out); err != nil {
		return result, err
	}
	result, err := out.ToResult()
	if err != nil {
		return result, err
	}
	return result, continuableErr
}

func (c *ProviderGen[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	result := genai.Result{}
	if err := msgs.Validate(); err != nil {
		return result, err
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return result, err
		}
		// TODO: Specify as a field.
		if supported := opts.Modalities(); !slices.Contains(supported, genai.ModalityText) {
			return result, fmt.Errorf("modality %s not supported", supported)
		}
	}
	// Check for non-empty Opaque field unless explicitly allowed
	if !c.AllowOpaqueFields {
		for i, msg := range msgs {
			for j, content := range msg.Contents {
				if len(content.Opaque) != 0 {
					return result, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
				}
			}
			for j, tool := range msg.ToolCalls {
				if len(tool.Opaque) != 0 {
					return result, fmt.Errorf("message #%d tool call #%d: field Opaque not supported", i, j)
				}
			}
		}
	}

	c.lateInit()
	in := reflect.New(c.chatRequest).Interface().(PGenRequest)
	var continuableErr error
	if err := in.Init(msgs, opts, c.Model); err != nil {
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			continuableErr = uce
		} else {
			return result, err
		}
	}
	ch := make(chan GenStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return c.ProcessStreamPackets(ch, chunks, &result)
	})
	err := c.GenStreamRaw(ctx, in, ch)
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

// GenSyncRaw is the generic raw implementation for the generation API endpoint.
// It sets Stream to false and sends a request to the chat URL.
func (c *ProviderGen[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) GenSyncRaw(ctx context.Context, in PGenRequest, out PGenResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.SetStream(false)
	return c.DoRequest(ctx, "POST", c.GenSyncURL, in, out)
}

// GenStreamRaw is the generic raw implementation for streaming Gen API endpoints.
// It sets Stream to true, enables stream options if available, and handles the SSE response.
func (c *ProviderGen[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) GenStreamRaw(ctx context.Context, in PGenRequest, out chan<- GenStreamChunkResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.SetStream(true)
	url := c.GenStreamURL
	if url == "" {
		url = c.GenSyncURL
	}
	resp, err := c.ClientJSON.Request(ctx, "POST", url, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.DecodeError(url, resp)
	}
	er := reflect.New(c.errorResponse).Interface().(PErrorResponse)
	return sse.Process(resp.Body, out, er, c.ClientJSON.Lenient)
}

func (c *ProviderGen[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) ModelID() string {
	return c.Model
}

func (c *ProviderGen[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) Validate() error {
	if !c.ModelOptional && c.Model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *ProviderGen[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) lateInit() {
	// TODO: Figure out how to not use reflection.
	c.Provider.lateInit()
	c.mu.Lock()
	if c.chatRequest == nil {
		var in PGenRequest
		c.chatRequest = reflect.TypeOf(in).Elem()
		var out PGenResponse
		c.chatResponse = reflect.TypeOf(out).Elem()
	}
	c.mu.Unlock()
}

//

// Time is a JSON encoded unix timestamp. This is used by many providers.
type Time int64

// AsTime returns the time as UTC so its string value doesn't depend on the local time zone.
func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0).UTC()
}

// ListModelsResponse is an interface for responses that contain model data.
type ListModelsResponse interface {
	// ToModels converts the provider-specific models to a slice of genai.Model
	ToModels() []genai.Model
}

// ListModels is a generic function that implements the common pattern for listing models across providers.
// It makes an HTTP GET request to the specified URL and converts the response to a slice of genai.Model.
func ListModels[PErrorResponse ErrAPI, R ListModelsResponse](ctx context.Context, c *Provider[PErrorResponse], url string) ([]genai.Model, error) {
	var resp R
	if err := c.DoRequest(ctx, "GET", url, nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// SimulateStream simulates GenStream for APIs that do not support streaming.
func SimulateStream(ctx context.Context, c genai.ProviderGen, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	res, err := c.GenSync(ctx, msgs, opts)
	if err == nil {
		for i := range res.Contents {
			if url := res.Contents[i].URL; url != "" {
				chunks <- genai.ContentFragment{
					Filename: res.Contents[i].Filename,
					URL:      res.Contents[i].URL,
				}
			} else if d := res.Contents[i].Document; d != nil {
				chunks <- genai.ContentFragment{
					Filename:         res.Contents[i].Filename,
					DocumentFragment: res.Contents[i].Document.(*bb.BytesBuffer).D,
				}
			} else {
				return res, fmt.Errorf("expected ContentFragment with URL or Document, got %#v", res.Contents)
			}
		}
	}
	return res, err
}
