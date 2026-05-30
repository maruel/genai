// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package anthropic implements a client for the Anthropic API, to use Claude.
//
// It is described at
// https://docs.anthropic.com/en/api/
package anthropic

// See official client at https://github.com/anthropics/anthropic-sdk-go

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Anthropic.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// ProviderOptionMultipartBoundary sets a fixed multipart boundary for
// deterministic HTTP recordings in tests. Leave unset for production use.
type ProviderOptionMultipartBoundary string

// Validate implements genai.Validatable.
func (p ProviderOptionMultipartBoundary) Validate() error {
	if p == "" {
		return errors.New("ProviderOptionMultipartBoundary cannot be empty")
	}
	return nil
}

// GenOptionText defines Anthropic specific options.
type GenOptionText struct {
	// ThinkingBudget is the maximum number of tokens the LLM can use to reason about the answer. When 0,
	// reasoning is disabled. It generally must be above 1024 and below MaxTokens.
	//
	// Ignored when Thinking is ThinkingAdaptive.
	ThinkingBudget int64
	// Thinking controls the thinking mode. When empty, the mode is auto-detected from ThinkingBudget:
	// ThinkingEnabled if ThinkingBudget > 0, ThinkingDisabled otherwise.
	//
	// Set to ThinkingAdaptive for models that support it (e.g. claude-opus-4-6). In adaptive mode, the
	// model decides autonomously whether and how much to think. ThinkingBudget is ignored.
	//
	// https://platform.claude.com/docs/en/build-with-claude/extended-thinking
	Thinking ThinkingType
	// ThinkingDisplay controls whether the thinking text is returned. Valid values are "summarized"
	// and "omitted". Defaults to "summarized" in adaptive mode since opus-4-8/4-7 default to
	// "omitted" which hides the thinking text.
	//
	// https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#controlling-thinking-display
	ThinkingDisplay string
	// MessagesToCache specify the number of messages to cache in the request.
	//
	// By default, the system prompt and tools will be cached.
	//
	// https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
	MessagesToCache int
	// Effort controls the quality/latency tradeoff. When empty, the API default is used.
	//
	// https://platform.claude.com/docs/en/api/messages#body-output-config
	Effort Effort
	// InferenceGeo specifies the geographic region for inference processing. If not specified,
	// the workspace's default_inference_geo is used.
	//
	// https://platform.claude.com/docs/en/api/messages/create
	InferenceGeo string
}

// ThinkingType controls how the model uses extended thinking.
//
// https://platform.claude.com/docs/en/build-with-claude/extended-thinking
type ThinkingType string

const (
	// ThinkingEnabled enables extended thinking with an explicit budget set via ThinkingBudget.
	ThinkingEnabled ThinkingType = "enabled"
	// ThinkingDisabled disables extended thinking.
	ThinkingDisabled ThinkingType = "disabled"
	// ThinkingAdaptive lets the model decide autonomously whether and how much to think.
	// ThinkingBudget is ignored. Use Effort to control thinking depth.
	ThinkingAdaptive ThinkingType = "adaptive"
)

// Validate implements genai.Validatable.
func (t ThinkingType) Validate() error {
	switch t {
	case "", ThinkingEnabled, ThinkingDisabled, ThinkingAdaptive:
		return nil
	default:
		return fmt.Errorf("invalid ThinkingType %q", t)
	}
}

// Effort controls the amount of effort the model puts into its response.
//
// https://platform.claude.com/docs/en/api/messages#body-output-config
type Effort string

const (
	// EffortLow minimizes latency at the cost of quality.
	EffortLow Effort = "low"
	// EffortMedium balances quality and latency.
	EffortMedium Effort = "medium"
	// EffortHigh favors quality over latency.
	EffortHigh Effort = "high"
	// EffortMax maximizes quality.
	EffortMax Effort = "max"
)

// Validate implements genai.Validatable.
func (o *GenOptionText) Validate() error {
	if err := o.Thinking.Validate(); err != nil {
		return err
	}
	if o.Thinking == ThinkingAdaptive && o.ThinkingBudget > 0 {
		return fmt.Errorf("ThinkingBudget must not be set when Thinking is %q", ThinkingAdaptive)
	}
	if o.Thinking == ThinkingEnabled && o.ThinkingBudget == 0 {
		return fmt.Errorf("ThinkingBudget must be set when Thinking is %q", ThinkingEnabled)
	}
	if o.ThinkingDisplay != "" && o.Thinking != ThinkingAdaptive {
		return fmt.Errorf("ThinkingDisplay is only valid with ThinkingAdaptive")
	}
	switch o.ThinkingDisplay {
	case "", "summarized", "omitted":
	default:
		return fmt.Errorf("invalid ThinkingDisplay %q", o.ThinkingDisplay)
	}
	switch o.Effort {
	case "", EffortLow, EffortMedium, EffortHigh, EffortMax:
	default:
		return fmt.Errorf("invalid Effort %q", o.Effort)
	}
	return nil
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
	// multipartBoundary overrides the multipart boundary for deterministic HTTP
	// recordings. Leave empty for production use.
	multipartBoundary string
}

// New creates a new client to talk to the Anthropic platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the ANTHROPIC_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get an API key at https://console.anthropic.com/settings/keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.anthropic.com/en/docs/about-claude/models/all-models
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model, multipartBoundary string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	if err := base.CheckDuplicateOptions(opts); err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionAPIKey:
			apiKey = string(v)
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			modalities = genai.Modalities(v)
		case genai.ProviderOptionPreloadedModels:
			preloadedModels = []genai.Model(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		case ProviderOptionMultipartBoundary:
			multipartBoundary = string(v)
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://console.anthropic.com/settings/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("ANTHROPIC_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "ANTHROPIC_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	// Anthropic allows Opaque fields for thinking signatures
	c := &Client{
		multipartBoundary: multipartBoundary,
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://api.anthropic.com/v1/messages",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProcessHeaders:  processHeaders,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"x-api-key": {apiKey}, "anthropic-version": {"2023-06-01"}},
						Transport: &betaHeader{transport: &roundtrippers.RequestID{Transport: t}},
					},
				},
			},
		},
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestTextModel selects the most recent model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	var date time.Time
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Always select the most recent model.
		if !date.IsZero() && m.CreatedAt.Before(date) {
			continue
		}
		switch {
		case cheap:
			if strings.Contains(m.ID, "-haiku-") {
				date = m.CreatedAt
				selectedModel = m.ID
			}
		case good:
			if strings.Contains(m.ID, "-sonnet-") {
				date = m.CreatedAt
				selectedModel = m.ID
			}
		default:
			if strings.Contains(m.ID, "-opus-") {
				date = m.CreatedAt
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// GenAsync implements genai.ProviderGenAsync.
//
// It requests the providers' batch API and returns the job ID. It can take up to 24 hours to complete.
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Job, error) {
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	c.ensureMaxTokens(ctx, opts)
	// https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
	// https://docs.anthropic.com/en/api/creating-message-batches
	// Anthropic supports creating multiple processing requests as part on one HTTP post. I'm not sure the value
	// of doing that, as it increases the risks of partial failure. So for now do not expose the functionality
	// of creating multiple requests at once unless we realize there's a specific use case.
	b := BatchRequest{Requests: []BatchRequestItem{{}}}
	if err := b.Requests[0].Init(msgs, c.impl.Model, opts...); err != nil {
		return "", err
	}
	resp, err := c.GenAsyncRaw(ctx, b)
	return genai.Job(resp.ID), err
}

// GenAsyncRaw provides access to the raw API structure.
func (c *Client) GenAsyncRaw(ctx context.Context, b BatchRequest) (BatchResponse, error) {
	resp := BatchResponse{}
	u := "https://api.anthropic.com/v1/messages/batches"
	err := c.impl.DoRequest(ctx, "POST", u, &b, &resp)
	return resp, err
}

// PokeResult implements genai.ProviderGenAsync.
//
// It retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	resp, err := c.PokeResultRaw(ctx, id)
	if err != nil && resp.Result.Type == "not_found_error" {
		res.Usage.FinishReason = genai.Pending
		return res, nil
	}
	if resp.Result.Type == "errored" {
		return res, fmt.Errorf("error %s: %s", resp.Result.Error.Error.Type, resp.Result.Error.Error.Message)
	}
	err = resp.To(&res.Message)
	res.Usage.InputTokens = resp.Result.Message.Usage.InputTokens
	res.Usage.InputCachedTokens = resp.Result.Message.Usage.CacheReadInputTokens
	res.Usage.OutputTokens = resp.Result.Message.Usage.OutputTokens
	res.Usage.TotalTokens = res.Usage.InputTokens + res.Usage.InputCachedTokens + res.Usage.OutputTokens
	res.Usage.FinishReason = resp.Result.Message.StopReason.ToFinishReason()
	res.Usage.ServiceTier = resp.Result.Message.Usage.ServiceTier
	if err == nil {
		err = res.Validate()
	}
	return res, err
}

// PokeResultRaw provides access to the raw API structure.
func (c *Client) PokeResultRaw(ctx context.Context, id genai.Job) (BatchQueryResponse, error) {
	resp := BatchQueryResponse{}
	// Warning: The documentation at https://docs.anthropic.com/en/api/retrieving-message-batch-results states
	// that the URL may change in the future.
	u := "https://api.anthropic.com/v1/messages/batches/" + url.PathEscape(string(id)) + "/results"
	err := c.impl.DoRequest(ctx, "GET", u, nil, &resp)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == http.StatusNotFound {
			er := ErrorResponse{}
			if json.Unmarshal(herr.ResponseBody, &er) == nil {
				if er.ErrorVal.Type == "not_found_error" {
					resp.Result.Type = er.ErrorVal.Type
				}
			}
		}
	}
	return resp, err
}

// Cancel implements genai.ProviderBatch.
func (c *Client) Cancel(ctx context.Context, id genai.Job) error {
	_, err := c.CancelRaw(ctx, id)
	return err
}

// CancelRaw cancels a batch request.
func (c *Client) CancelRaw(ctx context.Context, id genai.Job) (BatchResponse, error) {
	u := "https://api.anthropic.com/v1/messages/batches/" + url.PathEscape(string(id)) + "/cancel"
	resp := BatchResponse{}
	err := c.impl.DoRequest(ctx, "POST", u, &struct{}{}, &resp)
	return resp, err
}

// GetBatch retrieves the metadata for a single message batch.
//
// https://docs.anthropic.com/en/api/retrieving-message-batches
func (c *Client) GetBatch(ctx context.Context, id string) (*BatchResponse, error) {
	u := "https://api.anthropic.com/v1/messages/batches/" + url.PathEscape(id)
	resp := &BatchResponse{}
	err := c.impl.DoRequest(ctx, "GET", u, nil, resp)
	return resp, err
}

// ListBatches returns all message batches.
//
// https://docs.anthropic.com/en/api/listing-message-batches
func (c *Client) ListBatches(ctx context.Context) ([]BatchResponse, error) {
	resp, err := c.ListBatchesRaw(ctx)
	if err != nil {
		return nil, err
	}
	return resp.Data, nil
}

// ListBatchesRaw returns the raw paginated response for message batches.
//
// https://docs.anthropic.com/en/api/listing-message-batches
func (c *Client) ListBatchesRaw(ctx context.Context) (*BatchListResponse, error) {
	resp := &BatchListResponse{}
	err := c.impl.DoRequest(ctx, "GET", "https://api.anthropic.com/v1/messages/batches?limit=1000", nil, resp)
	return resp, err
}

// DeleteBatch deletes a message batch that has finished processing.
//
// In-progress batches must be canceled before deletion.
//
// https://docs.anthropic.com/en/api/deleting-message-batches
func (c *Client) DeleteBatch(ctx context.Context, id string) error {
	u := "https://api.anthropic.com/v1/messages/batches/" + url.PathEscape(id)
	resp := BatchDeleteResponse{}
	return c.impl.DoRequest(ctx, "DELETE", u, nil, &resp)
}

// fileDo executes an HTTP request with the files API beta header.
func (c *Client) fileDo(ctx context.Context, method, u string, body io.Reader, contentType string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, method, u, body)
	if err != nil {
		return nil, err
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	req.Header.Set("anthropic-beta", "files-api-2025-04-14")
	return c.impl.Client.Do(req)
}

// fileDoJSON executes a JSON GET/DELETE request with the files API beta header
// and decodes the response.
func (c *Client) fileDoJSON(ctx context.Context, method, u string, out any) error {
	resp, err := c.fileDo(ctx, method, u, nil, "")
	if err != nil {
		if resp != nil {
			_ = resp.Body.Close()
		}
		return err
	}
	return c.impl.DecodeResponse(resp, u, out)
}

// FileUpload uploads a file to the Anthropic files API.
//
// https://docs.anthropic.com/en/api/files
func (c *Client) FileUpload(ctx context.Context, filename string, r io.Reader) (*FileMetadata, error) {
	buf := bytes.Buffer{}
	w := multipart.NewWriter(&buf)
	if c.multipartBoundary != "" {
		_ = w.SetBoundary(c.multipartBoundary)
	}
	part, err := w.CreateFormFile("file", filename)
	if err != nil {
		return nil, err
	}
	if _, err = io.Copy(part, r); err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	u := "https://api.anthropic.com/v1/files?beta=true"
	resp, err := c.fileDo(ctx, "POST", u, &buf, w.FormDataContentType())
	if err != nil {
		if resp != nil {
			_ = resp.Body.Close()
		}
		return nil, err
	}
	f := FileMetadata{}
	err = c.impl.DecodeResponse(resp, u, &f)
	return &f, err
}

// FileDownload downloads a file's content from the Anthropic files API.
//
// The caller must close the returned io.ReadCloser.
//
// https://docs.anthropic.com/en/api/files
func (c *Client) FileDownload(ctx context.Context, id string) (io.ReadCloser, error) {
	u := "https://api.anthropic.com/v1/files/" + url.PathEscape(id) + "/content?beta=true"
	resp, err := c.fileDo(ctx, "GET", u, nil, "")
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, c.impl.DecodeError(u, resp)
	}
	return resp.Body, nil
}

// FileGetMetadata retrieves metadata for a single file.
//
// https://docs.anthropic.com/en/api/files
func (c *Client) FileGetMetadata(ctx context.Context, id string) (*FileMetadata, error) {
	u := "https://api.anthropic.com/v1/files/" + url.PathEscape(id) + "?beta=true"
	f := FileMetadata{}
	err := c.fileDoJSON(ctx, "GET", u, &f)
	return &f, err
}

// FileList returns all files.
//
// https://docs.anthropic.com/en/api/files
func (c *Client) FileList(ctx context.Context) ([]FileMetadata, error) {
	resp, err := c.FileListRaw(ctx)
	if err != nil {
		return nil, err
	}
	return resp.Data, nil
}

// FileListRaw returns the raw paginated response for files.
//
// https://docs.anthropic.com/en/api/files
func (c *Client) FileListRaw(ctx context.Context) (*FileListResponse, error) {
	u := "https://api.anthropic.com/v1/files?beta=true&limit=1000"
	resp := &FileListResponse{}
	err := c.fileDoJSON(ctx, "GET", u, resp)
	return resp, err
}

// FileDelete deletes a file.
//
// https://docs.anthropic.com/en/api/files
func (c *Client) FileDelete(ctx context.Context, id string) error {
	u := "https://api.anthropic.com/v1/files/" + url.PathEscape(id) + "?beta=true"
	resp := FileDeleteResponse{}
	return c.fileDoJSON(ctx, "DELETE", u, &resp)
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "anthropic"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to fetch results (e.g. videos) generated by the provider.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	c.ensureMaxTokens(ctx, opts)
	return c.impl.GenSync(ctxWithBeta(ctx, opts), msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	c.ensureMaxTokens(ctx, opts)
	return c.impl.GenStream(ctxWithBeta(ctx, opts), msgs, opts...)
}

// ctxWithBeta adds the web-fetch beta header to the context if WebFetch is enabled.
func ctxWithBeta(ctx context.Context, opts []genai.GenOption) context.Context {
	for _, o := range opts {
		if v, ok := o.(*genai.GenOptionWeb); ok && v.Fetch {
			return context.WithValue(ctx, ctxBetaKey{}, "web-fetch-2025-09-10")
		}
	}
	return ctx
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// ensureMaxTokens ensures the max_tokens is cached for the model.
// Falls back to ListModels for (new) unknown models.
func (c *Client) ensureMaxTokens(ctx context.Context, opts []genai.GenOption) {
	if c.impl.Model == "" {
		return
	}
	// Check if user already provided MaxTokens.
	for _, o := range opts {
		if v, ok := o.(*genai.GenOptionText); ok && v.MaxTokens != 0 {
			return
		}
	}
	// Check out cache.
	if _, ok := modelsMaxTokens.Load(c.impl.Model); ok {
		return
	}
	// Check upstream.
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return
	}
	for _, m := range mdls {
		if am, ok := m.(*Model); ok && am.ID == c.impl.Model && am.MaxTokens != 0 {
			modelsMaxTokens.Store(c.impl.Model, int(am.MaxTokens))
			return
		}
	}
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://docs.anthropic.com/en/api/models-list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.anthropic.com/v1/models?limit=1000", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// GetModel returns the details for a single model.
//
// https://docs.anthropic.com/en/api/models-get
func (c *Client) GetModel(ctx context.Context, id string) (*Model, error) {
	var resp Model
	u := "https://api.anthropic.com/v1/models/" + url.PathEscape(id)
	if err := c.impl.DoRequest(ctx, "GET", u, nil, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// CountTokens counts the number of tokens in the given messages.
//
// https://docs.anthropic.com/en/api/counting-tokens
func (c *Client) CountTokens(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (*CountTokensResponse, error) {
	c.ensureMaxTokens(ctx, opts)
	var chat ChatRequest
	if err := chat.Init(msgs, c.impl.Model, opts...); err != nil {
		return nil, err
	}
	req := CountTokensRequest{
		Model:        chat.Model,
		Messages:     chat.Messages,
		System:       chat.System,
		Thinking:     chat.Thinking,
		ToolChoice:   chat.ToolChoice,
		Tools:        chat.Tools,
		OutputConfig: chat.OutputConfig,
	}
	return c.CountTokensRaw(ctx, &req)
}

// CountTokensRaw provides raw API access to the count_tokens endpoint.
//
// https://docs.anthropic.com/en/api/messages-count-tokens
func (c *Client) CountTokensRaw(ctx context.Context, in *CountTokensRequest) (*CountTokensResponse, error) {
	var resp CountTokensResponse
	if err := c.impl.DoRequest(ctx, "POST", "https://api.anthropic.com/v1/messages/count_tokens", in, &resp); err != nil {
		return nil, err
	}
	return &resp, nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	var u genai.Usage

	return func(yield func(genai.Reply) bool) {
			// At the moment, only supported for server_tool_use / web_search.
			pendingServerCall := ""
			pendingJSON := ""
			pendingToolCall := genai.ToolCall{}
			for pkt := range chunks {
				f := genai.Reply{}
				// See testdata/TestClient_Chat_thinking/ChatStream.yaml as a great example.
				// TODO: pkt.Index matters here, as the LLM may fill multiple content blocks simultaneously.
				switch pkt.Type {
				case ChunkMessageStart:
					switch pkt.Message.Role {
					case "assistant":
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", pkt.Message.Role)}
						return
					}
					u.InputTokens = pkt.Message.Usage.InputTokens
					u.InputCachedTokens = pkt.Message.Usage.CacheReadInputTokens
					// There's some tokens listed there. Still save it in case it breaks midway.
					u.OutputTokens = pkt.Message.Usage.OutputTokens
					u.TotalTokens = u.InputTokens + u.InputCachedTokens + u.OutputTokens
					u.ServiceTier = pkt.Message.Usage.ServiceTier
					continue
				case ChunkContentBlockStart:
					switch pkt.ContentBlock.Type {
					case ContentText:
						f.Text = pkt.ContentBlock.Text
					case ContentThinking:
						f.Reasoning = pkt.ContentBlock.Thinking
					case ContentToolUse:
						pendingToolCall.ID = pkt.ContentBlock.ID
						pendingToolCall.Name = pkt.ContentBlock.Name
						pendingToolCall.Arguments = ""
						// TODO: Is there anything to do with Input? pendingCall.Arguments = pkt.ContentBlock.Input
					case ContentRedactedThinking:
						f.Opaque = map[string]any{"redacted_thinking": pkt.ContentBlock.Signature}
					case ContentServerToolUse:
						// Discard the data for now. It may be necessary in the future to keep in Opaque.
						pendingServerCall = pkt.ContentBlock.Name
						switch pendingServerCall {
						case "web_search", "web_fetch":
							// Supported server tool calls.
						default:
							// Oops, more work to do!
							if !internal.BeLenient {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement server tool call %q", pendingServerCall)}
								return
							}
						}
					case ContentWebSearchToolResult:
						f.Citation.Sources = make([]genai.CitationSource, len(pkt.ContentBlock.Content))
						for i := range pkt.ContentBlock.Content {
							cc := &pkt.ContentBlock.Content[i]
							if cc.Type != ContentWebSearchResult {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q while processing %q", cc.Type, pkt.ContentBlock.Type)}
								return
							}
							f.Citation.Sources[i].Type = genai.CitationWeb
							f.Citation.Sources[i].URL = cc.URL
							f.Citation.Sources[i].Title = cc.Title
							f.Citation.Sources[i].Date = cc.PageAge
							// EncryptedContent is not really useful?
						}
					case ContentMCPToolUse:
						f.Opaque = map[string]any{"mcp_tool_use": map[string]any{
							"id":    pkt.ContentBlock.ID,
							"input": pkt.ContentBlock.Input,
							"name":  pkt.ContentBlock.Name,
						}}
					case ContentMCPToolResult:
						f.Opaque = map[string]any{"mcp_tool_result": map[string]any{
							"tool_use_id": pkt.ContentBlock.ToolUseID,
							// "is_error":    pkt.ContentBlock.IsError,
							"content":     pkt.ContentBlock.Content,
							"server_name": pkt.ContentBlock.ServerName,
						}}
					case ContentWebFetchToolResult:
						for i := range pkt.ContentBlock.Content {
							cc := &pkt.ContentBlock.Content[i]
							switch cc.Type {
							case ContentWebFetchResult:
								title := cc.Title
								if title == "" && len(cc.Content) > 0 {
									title = cc.Content[0].Title
								}
								f.Citation.Sources = append(f.Citation.Sources, genai.CitationSource{
									Type:  genai.CitationWeb,
									URL:   cc.URL,
									Title: title,
								})
							case ContentWebFetchToolError:
								f.Opaque = map[string]any{"web_fetch_error": cc.ErrorCode}
							default:
								finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q while processing %q", cc.Type, pkt.ContentBlock.Type)}
								return
							}
						}
					case ContentWebSearchResult, ContentWebFetchResult, ContentWebFetchToolError, ContentImage, ContentDocument, ContentToolResult:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content block %q", pkt.ContentBlock.Type)}
						return
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content block %q", pkt.ContentBlock.Type)}
						return
					}
				case ChunkContentBlockDelta:
					switch pkt.Delta.Type {
					case DeltaText:
						f.Text = pkt.Delta.Text
					case DeltaThinking:
						f.Reasoning = pkt.Delta.Thinking
					case DeltaSignature:
						f.Opaque = map[string]any{"signature": pkt.Delta.Signature}
					case DeltaInputJSON:
						pendingJSON += pkt.Delta.PartialJSON
					case DeltaCitations:
						if err := pkt.Delta.Citation.To(&f.Citation); err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to parse citation: %w", err)}
							return
						}
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content block delta %q", pkt.Delta.Type)}
						return
					}
				case ChunkContentBlockStop:
					// Marks a closure of the block pkt.Index. Flush accumulated JSON if appropriate.
					if pendingToolCall.ID != "" {
						pendingToolCall.Arguments = pendingJSON
						f.ToolCall = pendingToolCall
						pendingToolCall = genai.ToolCall{}
					}
					// Why not web_search_20250305 ??
					switch pendingServerCall {
					case "web_search":
						q := WebSearch{}
						d := json.NewDecoder(strings.NewReader(pendingJSON))
						if !internal.BeLenient {
							d.DisallowUnknownFields()
						}
						if err := d.Decode(&q); err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to decode pending server tool call %s: %w", pendingServerCall, err)}
							return
						}
						f.Citation.Sources = []genai.CitationSource{{
							Type:    genai.CitationWebQuery,
							Snippet: q.Query,
						}}
						pendingServerCall = ""
					case "web_fetch":
						q := WebFetch{}
						d := json.NewDecoder(strings.NewReader(pendingJSON))
						if !internal.BeLenient {
							d.DisallowUnknownFields()
						}
						if err := d.Decode(&q); err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to decode pending server tool call %s: %w", pendingServerCall, err)}
							return
						}
						f.Citation.Sources = []genai.CitationSource{{
							Type: genai.CitationWeb,
							URL:  q.URL,
						}}
						pendingServerCall = ""
					case "":
					default:
						// Oops, more work to do!
						if !internal.BeLenient {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement server tool call %q", pendingServerCall)}
							return
						}
					}
					pendingJSON = ""
				case ChunkMessageDelta:
					// Includes finish reason and output tokens usage (but not input tokens!)
					u.FinishReason = pkt.Delta.StopReason.ToFinishReason()
					u.OutputTokens = pkt.Usage.OutputTokens
				case ChunkMessageStop:
					// Doesn't contain anything.
					continue
				case ChunkPing:
					// Doesn't contain anything.
					continue
				case ChunkError:
					// TODO: See it in the field to decode properly.
					finalErr = fmt.Errorf("got error: %+v", pkt)
					return
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("implement stream block %q", pkt.Type)}
					return
				}
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

type ctxBetaKey struct{}

// betaHeader conditionally adds the anthropic-beta header when ctxBetaKey is set in the request context.
type betaHeader struct {
	transport http.RoundTripper
}

func (b *betaHeader) RoundTrip(req *http.Request) (*http.Response, error) {
	if v, _ := req.Context().Value(ctxBetaKey{}).(string); v != "" {
		req = req.Clone(req.Context())
		req.Header.Set("anthropic-beta", v)
	}
	return b.transport.RoundTrip(req)
}

func (b *betaHeader) Unwrap() http.RoundTripper {
	return b.transport
}

func processHeaders(h http.Header) []genai.RateLimit {
	requestsLimit, _ := strconv.ParseInt(h.Get("Anthropic-Ratelimit-Requests-Limit"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("Anthropic-Ratelimit-Requests-Remaining"), 10, 64)
	requestsReset, _ := time.Parse(time.RFC3339, h.Get("Anthropic-Ratelimit-Requests-Reset"))

	tokensLimit, _ := strconv.ParseInt(h.Get("Anthropic-Ratelimit-Tokens-Limit"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("Anthropic-Ratelimit-Tokens-Remaining"), 10, 64)
	tokensReset, _ := time.Parse(time.RFC3339, h.Get("Anthropic-Ratelimit-Tokens-Reset"))

	var limits []genai.RateLimit
	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     requestsReset.Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     tokensReset.Round(10 * time.Millisecond),
		})
	}
	return limits
}

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{
		GenAsync: true,
	}
}

//go:embed models.json
var modelsJSON []byte

// modelsMeta holds embedded model metadata from models.json, keyed by model ID.
type modelsData struct {
	Models map[string]modelData `json:"models"`
}

type modelData struct {
	MaxInputTokens               int64 `json:"max_input_tokens"`
	MaxOutputTokens              int64 `json:"max_output_tokens"`
	AdaptiveThinkingNotSupported bool  `json:"adaptive_thinking_not_supported,omitempty"`
}

var (
	// modelsMaxTokens caches max output tokens per model from models.json.
	modelsMaxTokens sync.Map
	// modelsAdaptiveBlocked is true for models that don't support adaptive thinking.
	modelsAdaptiveBlocked sync.Map
)

func init() {
	data := modelsData{}
	if err := json.Unmarshal(modelsJSON, &data); err != nil {
		panic(fmt.Sprintf("failed to parse embedded models.json: %v", err))
	}
	for k, v := range data.Models {
		modelsMaxTokens.Store(k, int(v.MaxOutputTokens))
		if v.AdaptiveThinkingNotSupported {
			modelsAdaptiveBlocked.Store(k, true)
		}
	}
}

var (
	_ internal.Validatable = &Message{}
	_ internal.Validatable = &Content{}
	_ genai.Provider       = &Client{}
)
