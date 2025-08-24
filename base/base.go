// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package base is awesome sauce to reduce code duplication across most providers.
//
// It is not meant to be used by end users.
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
	"github.com/maruel/genai/internal"
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

// ErrNotSupported is returned when a method is not implemented because the provider doesn't support it.
//
// For example Perplexity doesn't have an API to lists its supported models (this may change in the future).
var ErrNotSupported = errors.New("not supported")

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

// NotImplemented implements most genai.Provider methods all returning ErrNotSupported.
type NotImplemented struct{}

func (*NotImplemented) GenSync(context.Context, genai.Messages, genai.Options) (genai.Result, error) {
	return genai.Result{}, ErrNotSupported
}

func (*NotImplemented) GenStream(context.Context, genai.Messages, chan<- genai.ReplyFragment, genai.Options) (genai.Result, error) {
	return genai.Result{}, ErrNotSupported
}

func (*NotImplemented) ListModels(context.Context) ([]genai.Model, error) {
	return nil, ErrNotSupported
}

// ProviderBase implements the basse functionality to help implementing a base.Provider.
//
// It contains the shared HTTP client functionality used across all API clients.
type ProviderBase[PErrorResponse ErrAPI] struct {
	// Client is exported for testing replay purposes.
	Client http.Client
	// Lenient allows unknown fields in the response.
	//
	// This inhibits from calling DisallowUnknownFields() on the JSON decoder, which will generally return a
	// *UnknownFieldError.
	//
	// Use this in production so that your client doesn't break when the server
	// add new fields.
	Lenient bool
	// APIKeyURL is the URL to present to the user upon authentication error.
	APIKeyURL string
	// Model is the default model used for chat requests
	Model string
	// OutputModalities is the output modalities supported by the provider.
	OutputModalities genai.Modalities
	// ModelOptional is true if a model name is not required to use the provider.
	ModelOptional bool

	mu            sync.Mutex
	errorResponse reflect.Type
	lastResp      http.Header
}

// JSONRequest simplifies doing an HTTP PATCH/DELETE/PUT in JSON.
//
// In is optional.
//
// It initiates the requests and returns the response back for further processing.
// Buffers post data in memory.
func (c *ProviderBase[PErrorResponse]) JSONRequest(ctx context.Context, method, url string, in any) (*http.Response, error) {
	var b io.Reader
	if in != nil {
		buf := &bytes.Buffer{}
		e := json.NewEncoder(buf)
		// OMG this took me a while to figure this out. This affects LLM token encoding.
		e.SetEscapeHTML(false)
		if err := e.Encode(in); err != nil {
			return nil, fmt.Errorf("internal error: %w", err)
		}
		b = buf
	}
	req, err := http.NewRequestWithContext(ctx, method, url, b)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	return c.Client.Do(req)
}

func (c *ProviderBase[PErrorResponse]) Validate() error {
	if !c.ModelOptional && c.Model == "" {
		return errors.New("a model is required")
	}
	return nil
}

// LastResponseHeaders returns the HTTP headers of the last response.
func (c *ProviderBase[PErrorResponse]) LastResponseHeaders() http.Header {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastResp
}

// DoRequest performs an HTTP request and handles error responses.
//
// It takes care of sending the request, decoding the response, and handling errors.
// All API clients should use this method for their HTTP communication needs.
func (c *ProviderBase[PErrorResponse]) DoRequest(ctx context.Context, method, url string, in, out any) error {
	c.lateInit()
	resp, err := c.JSONRequest(ctx, method, url, in)
	if err != nil {
		return err
	}
	return c.DecodeResponse(resp, url, out)
}

func (c *ProviderBase[PErrorResponse]) DecodeResponse(resp *http.Response, url string, out any) error {
	c.mu.Lock()
	c.lastResp = resp.Header
	c.mu.Unlock()
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
	if !c.Lenient {
		d.DisallowUnknownFields()
		r2 = r
	}
	var errs []error
	if foundExtraKeys, err2 := internal.DecodeJSON(d, out, r2); err2 == nil {
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
	if !c.Lenient {
		d.DisallowUnknownFields()
		r2 = r
	}
	er := reflect.New(c.errorResponse).Interface().(PErrorResponse)
	if foundExtraKeys, err := internal.DecodeJSON(d, er, r2); err == nil {
		// It may have succeeded but not decoded anything.
		if v := reflect.ValueOf(er); !reflect.DeepEqual(v, reflect.Zero(c.errorResponse).Interface()) {
			errs = append(errs, er)
		} else {
			errs = append(errs, err)
		}
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
func (c *ProviderBase[PErrorResponse]) DecodeError(url string, resp *http.Response) error {
	c.mu.Lock()
	c.lastResp = resp.Header
	c.mu.Unlock()
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
	if !c.Lenient {
		d.DisallowUnknownFields()
		r2 = r
	}
	var errs []error
	herr := &httpjson.Error{StatusCode: resp.StatusCode, ResponseBody: b}
	er := reflect.New(c.errorResponse).Interface().(PErrorResponse)
	if foundExtraKeys, err := internal.DecodeJSON(d, er, r2); err == nil {
		errs = append(errs, herr)
		// It may have succeeded but not decoded anything.
		if v := reflect.ValueOf(er); !reflect.DeepEqual(v, reflect.Zero(c.errorResponse).Interface()) {
			errs = append(errs, er)
		}
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

func (c *ProviderBase[PErrorResponse]) lateInit() {
	// TODO: Figure out how to not use reflection.
	c.mu.Lock()
	if c.errorResponse == nil {
		var in PErrorResponse
		c.errorResponse = reflect.TypeOf(in).Elem()
	}
	c.mu.Unlock()
}

//

// Obj is a generic interface for chat-related types (both requests and responses)
type Obj interface{ any }

// InitializableRequest is an interface for request types that can be initialized.
type InitializableRequest interface {
	// Init initializes the request with messages, options, and model.
	Init(msgs genai.Messages, model string, opts ...genai.Options) error
	// SetStream set the stream mode.
	SetStream(bool)
}

// ResultConverter converts a provider-specific result to a genai.Result.
type ResultConverter interface {
	ToResult() (genai.Result, error)
}

//

// Provider implements genai.Provider.
//
// It includes common functionality for clients that provide chat capabilities.
//
// It only accepts text modality.
type Provider[PErrorResponse ErrAPI, PGenRequest InitializableRequest, PGenResponse ResultConverter, GenStreamChunkResponse Obj] struct {
	ProviderBase[PErrorResponse]
	// GenSyncURL is the endpoint URL for chat API requests
	GenSyncURL string
	// GenStreamURL is the endpoint URL for chat stream API requests. It defaults to GenURL if unset.
	GenStreamURL string
	// ProcessStreamPackets is the function that processes stream packets used by GenStream.
	ProcessStreamPackets func(ch <-chan GenStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error
	// ProcessHeaders is the function that processes HTTP headers to extract rate limit information.
	ProcessHeaders func(http.Header) []genai.RateLimit
	// LieToolCalls lie the FinishReason on tool calls.
	LieToolCalls bool

	// Protected by Base.mu.
	chatRequest  reflect.Type
	chatResponse reflect.Type
}

func (c *Provider[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	result := genai.Result{}
	if opts != nil {
		// TODO: Specify as a field.
		if supported := opts.Modalities(); !slices.Contains(supported, genai.ModalityText) {
			return result, fmt.Errorf("modality %s not supported", supported)
		}
	}

	c.lateInit()
	in := reflect.New(c.chatRequest).Interface().(PGenRequest)
	var continuableErr error
	var o []genai.Options
	if opts != nil {
		o = append(o, opts)
	}
	if err := in.Init(msgs, c.Model, o...); err != nil {
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
	if err := result.Validate(); err != nil {
		// Catch provider implementation bugs.
		return result, err
	}

	lastResp := c.LastResponseHeaders()
	if c.ProcessHeaders != nil && lastResp != nil {
		result.Usage.Limits = c.ProcessHeaders(lastResp)
	}
	return result, continuableErr
}

func (c *Provider[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	result := genai.Result{}
	if opts != nil {
		// TODO: Specify as a field.
		if supported := opts.Modalities(); !slices.Contains(supported, genai.ModalityText) {
			return result, fmt.Errorf("modality %s not supported", supported)
		}
	}

	c.lateInit()
	in := reflect.New(c.chatRequest).Interface().(PGenRequest)
	var continuableErr error
	var o []genai.Options
	if opts != nil {
		o = append(o, opts)
	}
	if err := in.Init(msgs, c.Model, o...); err != nil {
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
	lastResp := c.LastResponseHeaders()
	if c.ProcessHeaders != nil && lastResp != nil {
		result.Usage.Limits = c.ProcessHeaders(lastResp)
	}
	if c.LieToolCalls && result.Usage.FinishReason == genai.FinishedStop {
		for i := range result.Replies {
			if !result.Replies[i].ToolCall.IsZero() {
				// Lie for the benefit of everyone.
				result.Usage.FinishReason = genai.FinishedToolCalls
				break
			}
		}
	}
	if err != nil {
		return result, err
	}
	if err := result.Validate(); err != nil {
		// Catch provider implementation bugs.
		return result, err
	}
	return result, continuableErr
}

// GenSyncRaw is the generic raw implementation for the generation API endpoint.
// It sets Stream to false and sends a request to the chat URL.
func (c *Provider[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) GenSyncRaw(ctx context.Context, in PGenRequest, out PGenResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.SetStream(false)
	return c.DoRequest(ctx, "POST", c.GenSyncURL, in, out)
}

// GenStreamRaw is the generic raw implementation for streaming Gen API endpoints.
// It sets Stream to true, enables stream options if available, and handles the SSE response.
func (c *Provider[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) GenStreamRaw(ctx context.Context, in PGenRequest, out chan<- GenStreamChunkResponse) error {
	if err := c.Validate(); err != nil {
		return err
	}
	in.SetStream(true)
	url := c.GenStreamURL
	if url == "" {
		url = c.GenSyncURL
	}
	resp, err := c.JSONRequest(ctx, "POST", url, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.DecodeError(url, resp)
	}
	c.mu.Lock()
	c.lastResp = resp.Header
	c.mu.Unlock()
	er := reflect.New(c.errorResponse).Interface().(PErrorResponse)
	return sse.Process(resp.Body, out, er, c.Lenient)
}

func (c *Provider[PErrorResponse, PGenRequest, PGenResponse, GenStreamChunkResponse]) lateInit() {
	// TODO: Figure out how to not use reflection.
	c.ProviderBase.lateInit()
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

// SimulateStream simulates GenStream for APIs that do not support streaming.
func SimulateStream(ctx context.Context, c genai.Provider, msgs genai.Messages, chunks chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	res, err := c.GenSync(ctx, msgs, opts)
	if err == nil {
		for i := range res.Replies {
			if !res.Replies[i].Doc.IsZero() {
				return res, fmt.Errorf("expected Reply with Doc, got %#v", res.Replies[i])
			}
			if url := res.Replies[i].Doc.URL; url != "" {
				chunks <- genai.ReplyFragment{
					Filename: res.Replies[i].Doc.Filename,
					URL:      res.Replies[i].Doc.URL,
				}
			} else {
				chunks <- genai.ReplyFragment{
					Filename:         res.Replies[i].Doc.Filename,
					DocumentFragment: res.Replies[i].Doc.Src.(*bb.BytesBuffer).D,
				}
			}
		}
	}
	return res, err
}

// MimeByExt wraps mime.TypeByExtension.
//
// It overrides audio entries because they vary surprisingly a lot across OSes!
func MimeByExt(ext string) string {
	return internal.MimeByExt(ext)
}
