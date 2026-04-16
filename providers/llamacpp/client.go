// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llamacpp implements a client for the llama-server native API, not
// the OpenAI compatible one.
//
// It is described at
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#api-endpoints
//
// The implementation is at https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp
package llamacpp

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"math"
	"net/http"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/sse"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for llama.cpp.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// GenOption is the llama.cpp-specific options.
type GenOption struct {
	// ReasoningFormat sets the reasoning format for the model.
	ReasoningFormat ReasoningFormat
	// Thinking enables thinking mode via chat_template_kwargs.
	Thinking bool
}

// Validate implements genai.Validatable.
func (o *GenOption) Validate() error {
	return nil
}

// Model is a synthetic struct combining the information from both ModelHF and ModelOpenAI.
type Model struct {
	HF     ModelHF
	OpenAI ModelOpenAI
}

// GetID implements genai.Model.
//
// It returns the base path name otherwise it's overwhelming and breaks our test cases.
func (m *Model) GetID() string {
	return filepath.Base(m.OpenAI.ID)
}

func (m *Model) String() string {
	return m.OpenAI.ID
}

// Context returns the training context window size.
func (m *Model) Context() int64 {
	return m.OpenAI.Meta.NCtxTrain
}

// PromptEncoding describes how to encode the prompt.
type PromptEncoding struct {
	// Prompt encoding.
	BeginOfText              string `yaml:"begin_of_text"`
	SystemTokenStart         string `yaml:"system_token_start"`
	SystemTokenEnd           string `yaml:"system_token_end"`
	UserTokenStart           string `yaml:"user_token_start"`
	UserTokenEnd             string `yaml:"user_token_end"`
	AssistantTokenStart      string `yaml:"assistant_token_start"`
	AssistantTokenEnd        string `yaml:"assistant_token_end"`
	ToolsAvailableTokenStart string `yaml:"tools_available_token_start"`
	ToolsAvailableTokenEnd   string `yaml:"tools_available_token_end"`
	ToolCallTokenStart       string `yaml:"tool_call_token_start"`
	ToolCallTokenEnd         string `yaml:"tool_call_token_end"`
	ToolCallResultTokenStart string `yaml:"tool_call_result_token_start"`
	ToolCallResultTokenEnd   string `yaml:"tool_call_result_token_end"`

	_ struct{}
}

// Validate checks for obvious errors in the fields.
func (p *PromptEncoding) Validate() error {
	// TODO: I lied. Please send a PR.
	return nil
}

// TokenPerformance is the performance for the metrics.
type TokenPerformance struct {
	Count    int
	Duration time.Duration
}

func (t *TokenPerformance) String() string {
	return fmt.Sprintf("%d (%s)", t.Count, t.Duration)
}

// Rate is the number of token per second.
func (t *TokenPerformance) Rate() float64 {
	if t.Duration == 0 {
		return 0
	}
	return float64(t.Count) / (float64(t.Duration) / float64(time.Second))
}

// Metrics represents the metrics for the LLM server.
type Metrics struct {
	Prompt             TokenPerformance
	Generated          TokenPerformance
	KVCacheUsage       float64
	KVCacheTokens      int
	RequestsProcessing int
	RequestedPending   int
}

//

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl           base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
	baseURL        string
	completionsURL string
	modelsURL      string
	encoding       *PromptEncoding
}

// New creates a new client to talk to a llama-server instance.
//
// ProviderOptionRemote defaults to "http://localhost:8080".
//
// llama-server doesn't have any mean of authentication so ProviderOptionAPIKey is not supported.
//
// Automatic model selection via ModelCheap, ModelGood, ModelSOTA is not supported. It will ask llama-server
// to determine which model is already loaded.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var baseURL, model string
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
		case genai.ProviderOptionRemote:
			baseURL = string(v)
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			modalities = genai.Modalities(v)
		case genai.ProviderOptionPreloadedModels:
			preloadedModels = []genai.Model(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      baseURL + "/chat/completions",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				ModelOptional: true,
				Lenient:       internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.RequestID{Transport: t},
				},
			},
		},
		baseURL:        baseURL,
		completionsURL: baseURL + "/completions",
		modelsURL:      baseURL + "/v1/models",
	}
	var err error
	switch model {
	case "":
	case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
		if c.impl.Model, err = c.selectBestTextModel(ctx); err == nil {
			c.impl.OutputModalities = mod
		}
	default:
		c.impl.Model = model
		c.impl.OutputModalities = mod
	}
	return c, err
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context) (string, error) {
	// Figure out the model loaded if any.
	m, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	if len(m) > 0 {
		return m[0].GetID(), nil
	}
	return "", nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "llamacpp"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID or what was discovered from the server.
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
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (fragments iter.Seq[genai.Reply], finish func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (chunks iter.Seq[ChatStreamChunkResponse], finish func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", c.modelsURL, nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// Completion sends a completion request and returns the result.
func (c *Client) Completion(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
	// Doc mentions Cache:true causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	if err := msgs.Validate(); err != nil {
		return genai.Result{}, err
	}
	for i := range msgs {
		for j := range msgs[i].Replies {
			if len(msgs[i].Replies[j].Opaque) != 0 {
				return genai.Result{}, fmt.Errorf("message #%d: reply #%d: field Reply.Opaque not supported", i, j)
			}
		}
	}
	rpcin := CompletionRequest{CachePrompt: true}
	if err := rpcin.Init(msgs, "", opts...); err != nil {
		return genai.Result{}, err
	}
	if err := c.initPrompt(ctx, &rpcin, msgs, opts...); err != nil {
		return genai.Result{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.Result{}, fmt.Errorf("failed to get llama server response: %w", err)
	}
	return rpcout.ToResult()
}

// CompletionRaw provides raw access to the completion API.
func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	in.Stream = false
	return c.impl.DoRequest(ctx, "POST", c.completionsURL, in, out)
}

// CompletionStream sends a streaming completion request.
func (c *Client) CompletionStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (fragments iter.Seq[genai.Reply], finish func() (genai.Result, error)) {
	res := genai.Result{}
	var finalErr error

	fnFragments := func(yield func(genai.Reply) bool) {
		in := CompletionRequest{}
		if err := in.Init(msgs, "", opts...); err != nil {
			finalErr = err
			return
		}
		// Converts raw chunks into fragments.
		// Generate parsed chunks from the raw JSON SSE stream.
		chunks, finish := c.CompletionStreamRaw(ctx, &in)
		fragments, finish2 := ProcessCompletionStream(chunks)
		for f := range fragments {
			if f.IsZero() {
				continue
			}
			if err := f.Validate(); err != nil {
				// Catch provider implementation bugs.
				finalErr = &internal.BadError{Err: err}
				break
			}
			if err := res.Accumulate(&f); err != nil {
				finalErr = &internal.BadError{Err: err}
				return
			}
			if !yield(f) {
				break
			}
		}
		if err := finish(); finalErr == nil {
			finalErr = err
		}
		var err error
		res.Usage, res.Logprobs, err = finish2()
		if finalErr == nil {
			finalErr = err
		}
	}
	fnFinish := func() (genai.Result, error) {
		if finalErr != nil {
			return res, finalErr
		}
		if err := res.Validate(); err != nil {
			// Catch provider implementation bugs.
			return res, &internal.BadError{Err: err}
		}
		return res, nil
	}
	return fnFragments, fnFinish
}

// CompletionStreamRaw provides raw access to the streaming completion API.
func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest) (chunks iter.Seq[CompletionStreamChunkResponse], finish func() error) {
	in.Stream = true
	resp, err := c.impl.JSONRequest(ctx, "POST", c.completionsURL, in)
	if err != nil {
		return yieldNothing[CompletionStreamChunkResponse], func() error {
			return &internal.BadError{Err: fmt.Errorf("failed to get llama server response: %w", err)}
		}
	}
	if resp.StatusCode != http.StatusOK {
		defer func() { _ = resp.Body.Close() }()
		return yieldNothing[CompletionStreamChunkResponse], func() error {
			return &internal.BadError{Err: c.impl.DecodeError(c.completionsURL, resp)}
		}
	}
	// Process the stream in a separate goroutine to make sure that when the client iterate, there is already a
	// packet waiting for it. This reduces the overall latency.
	out := make(chan CompletionStreamChunkResponse, 16)
	ch := make(chan error)
	go func() {
		defer func() { _ = resp.Body.Close() }()
		er := ErrorResponse{}
		it, finish := sse.Process[CompletionStreamChunkResponse](resp.Body, &er, c.impl.Lenient)
		for pkt := range it {
			out <- pkt
		}
		err := finish()
		close(out)
		ch <- err
	}()

	return func(yield func(CompletionStreamChunkResponse) bool) {
			for pkt := range out {
				if !yield(pkt) {
					break
				}
			}
		}, func() error {
			return <-ch
		}
}

// GetHealth returns the health status string from the server.
func (c *Client) GetHealth(ctx context.Context) (string, error) {
	msg, err := c.GetHealthRaw(ctx)
	return msg.Status, err
}

// GetHealthRaw returns the raw health response from the server.
func (c *Client) GetHealthRaw(ctx context.Context) (HealthResponse, error) {
	msg := HealthResponse{}
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/health", nil, &msg); err != nil {
		return msg, fmt.Errorf("failed to get health response: %w", err)
	}
	return msg, nil
}

// Ping verifies the server is healthy and available.
func (c *Client) Ping(ctx context.Context) error {
	status, err := c.GetHealth(ctx)
	if err == nil && status != "ok" {
		err = fmt.Errorf("server unavailable. status: %q", status)
	}
	return err
}

// GetMetrics retrieves the performance statistics from the server.
func (c *Client) GetMetrics(ctx context.Context, m *Metrics) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/metrics", http.NoBody)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	// This is not a JSON response.
	resp, err := c.impl.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to get metrics response: %w", err)
	}
	b, err := io.ReadAll(resp.Body)
	if err2 := resp.Body.Close(); err == nil {
		err = err2
	}
	if err != nil {
		return fmt.Errorf("failed to get metrics response: %w", err)
	}
	// We hardcode things here since we know which server we are talking to. See
	// the commit history if you want the generic prometheus style data.
	for l := range strings.SplitSeq(strings.TrimSpace(string(b)), "\n") {
		if strings.HasPrefix(l, "#") {
			continue
		}
		parts := strings.Split(l, " ")
		if len(parts) != 2 {
			return fmt.Errorf("failed to parse line %q: %w", l, err)
		}
		// Search for these strings in
		// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp
		f := 0.0
		if parts[1] == "nan" || parts[1] == "-nan" {
			f = math.NaN()
		} else {
			if f, err = strconv.ParseFloat(parts[1], 64); err != nil {
				return fmt.Errorf("failed to parse line %q: %w", l, err)
			}
		}
		i, _ := strconv.Atoi(parts[1])
		switch parts[0] {
		case "llamacpp:prompt_tokens_total":
			m.Prompt.Count = i
		case "llamacpp:prompt_seconds_total":
			m.Prompt.Duration = time.Duration(f*1000) * time.Millisecond
		case "llamacpp:tokens_predicted_total":
			m.Generated.Count = i
		case "llamacpp:tokens_predicted_seconds_total":
			m.Generated.Duration = time.Duration(f*1000) * time.Millisecond
		case "llamacpp:prompt_tokens_seconds", "llamacpp:predicted_tokens_seconds":
			// Ignore.
		case "llamacpp:kv_cache_usage_ratio":
			m.KVCacheUsage = f
		case "llamacpp:kv_cache_tokens":
			m.KVCacheTokens = i
		case "llamacpp:requests_processing":
			m.RequestsProcessing = i
		case "llamacpp:requests_deferred":
			m.RequestedPending = i
		case "llamacpp:n_decode_total":
		case "llamacpp:n_busy_slots_per_decode":
		case "llamacpp:n_past_max":
		case "llamacpp:n_tokens_max":
		default:
			if !internal.BeLenient {
				panic(fmt.Sprintf("unknown metric %q", l))
			}
			return fmt.Errorf("unknown metric %q", l)
		}
	}
	return nil
}

func (c *Client) initPrompt(ctx context.Context, in *CompletionRequest, msgs genai.Messages, opts ...genai.GenOption) error {
	if c.encoding == nil {
		// Use the server to convert the OpenAI style format into a templated form.
		in2 := applyTemplateRequest{}
		if err := in2.Init(msgs, opts...); err != nil {
			return err
		}
		out := applyTemplateResponse{}
		if err := c.impl.DoRequest(ctx, "POST", c.baseURL+"/apply-template", &in2, &out); err != nil {
			return err
		}
		in.Prompt = out.Prompt
		return nil
	}

	in.Prompt = c.encoding.BeginOfText
	for _, m := range msgs {
		switch r := m.Role(); r {
		/* TODO
		case genai.AvailableTools:
			in.Prompt += c.encoding.ToolsAvailableTokenStart + m.Text + c.encoding.ToolsAvailableTokenEnd
		*/
		case "system":
			for _, b := range m.Requests {
				in.Prompt += c.encoding.SystemTokenStart + b.Text + c.encoding.SystemTokenEnd
			}
		case "user":
			for _, b := range m.Requests {
				in.Prompt += c.encoding.UserTokenStart + b.Text + c.encoding.UserTokenEnd
			}
		case "assistant":
			for _, b := range m.Requests {
				in.Prompt += c.encoding.AssistantTokenStart + b.Text + c.encoding.AssistantTokenEnd
			}
		default:
			return fmt.Errorf("unexpected role %q", r)
		}
	}
	return nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (fragments iter.Seq[genai.Reply], finish func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if pkt.Usage.PromptTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
				} else if pkt.Timings.PredictedN != 0 {
					// llama-server doesn't include usage in streaming responses
					// by default; fall back to the timings field which is always
					// present in the final chunk.
					u.InputTokens = pkt.Timings.PromptN
					u.InputCachedTokens = pkt.Timings.CacheN
					u.OutputTokens = pkt.Timings.PredictedN
				}
				if len(pkt.Choices) != 1 {
					continue
				}
				l = append(l, pkt.Choices[0].Logprobs.To()...)
				if pkt.Choices[0].FinishReason != "" {
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				f := genai.Reply{
					Text:      pkt.Choices[0].Delta.Content,
					Reasoning: pkt.Choices[0].Delta.ReasoningContent,
				}
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt)}
					return
				}
				if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
					if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID != "" {
						// A new call.
						if pendingToolCall.ID == "" {
							pendingToolCall = t
							if !f.IsZero() {
								finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
								return
							}
							continue
						}
						// Flush.
						pendingToolCall.To(&f.ToolCall)
						pendingToolCall = t
					} else if pendingToolCall.ID != "" {
						// Continuation.
						pendingToolCall.Function.Arguments += t.Function.Arguments
						if !f.IsZero() {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement tool call with metadata: %#v", pkt)}
							return
						}
						continue
					}
				} else if pendingToolCall.ID != "" {
					// Flush.
					pendingToolCall.To(&f.ToolCall)
					pendingToolCall = ToolCall{}
				}
				if !yield(f) {
					break
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
}

// ProcessCompletionStream converts the raw packets from the completion streaming API into Reply fragments.
func ProcessCompletionStream(chunks iter.Seq[CompletionStreamChunkResponse]) (fragments iter.Seq[genai.Reply], finish func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if pkt.Timings.PredictedN != 0 {
					u.InputTokens = pkt.Timings.PromptN
					u.OutputTokens = pkt.Timings.PredictedN
				}
				if pkt.StopType != "" {
					u.FinishReason = pkt.StopType.ToFinishReason()
				}
				if !yield(genai.Reply{Text: pkt.Content}) {
					break
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

func yieldNothing[T any](yield func(T) bool) {
}

var _ genai.Provider = &Client{}
