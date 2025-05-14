// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llamacpp implements a client for the llama-server native API, not
// the OpenAI compatible one.
//
// It is described at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
package llamacpp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// healthResponse is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
type healthResponse struct {
	Status          string
	SlotsIdle       int64 `json:"slots_idle"`
	SlotsProcessing int64 `json:"slots_processing"`
}

// https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
type CompletionRequest struct {
	// TODO: Prompt can be a string, a list of tokens or a mix.
	Prompt              string             `json:"prompt"`
	Temperature         float64            `json:"temperature,omitzero"`
	DynaTempRange       float64            `json:"dynatemp_range,omitzero"`
	DynaTempExponent    float64            `json:"dynatemp_exponent,omitzero"`
	TopK                int64              `json:"top_k,omitzero"`
	TopP                float64            `json:"top_p,omitzero"`
	MinP                float64            `json:"min_p,omitzero"`
	NPredict            int64              `json:"n_predict,omitzero"` // Maximum number of tokens to predict
	NIndent             int64              `json:"n_indent,omitzero"`
	NKeep               int64              `json:"n_keep,omitzero"`
	Stream              bool               `json:"stream"`
	Stop                []string           `json:"stop,omitzero"`
	TypicalP            float64            `json:"typical_p,omitzero"`
	RepeatPenalty       float64            `json:"repeat_penalty,omitzero"`
	RepeatLastN         int64              `json:"repeat_last_n,omitzero"`
	PresencePenalty     float64            `json:"presence_penalty,omitzero"`
	FrequencyPenalty    float64            `json:"frequency_penalty,omitzero"`
	DryMultiplier       float64            `json:"dry_multiplier,omitzero"`
	DryBase             float64            `json:"dry_base,omitzero"`
	DryAllowedLength    int64              `json:"dry_allowed_length,omitzero"`
	DryPenaltyLastN     int64              `json:"dry_penalty_last_n,omitzero"`
	DrySequenceBreakers []string           `json:"dry_sequence_breakers,omitzero"`
	XTCProbability      float64            `json:"xtc_probability,omitzero"`
	XTCThreshold        float64            `json:"xtc_threshold,omitzero"`
	Mirostat            int32              `json:"mirostat,omitzero"`
	MirostatTau         float64            `json:"mirostat_tau,omitzero"`
	MirostatEta         float64            `json:"mirostat_eta,omitzero"`
	Grammar             string             `json:"grammar,omitzero"`
	JSONSchema          *jsonschema.Schema `json:"json_schema,omitzero"`
	Seed                int64              `json:"seed,omitzero"`
	IgnoreEos           bool               `json:"ignore_eos,omitzero"`
	LogitBias           []any              `json:"logit_bias,omitzero"`
	Nprobs              int64              `json:"n_probs,omitzero"`
	MinKeep             int64              `json:"min_keep,omitzero"`
	TMaxPredictMS       int64              `json:"t_max_predict_ms,omitzero"`
	ImageData           []any              `json:"image_data,omitzero"`
	IDSlot              int64              `json:"id_slot,omitzero"`
	CachePrompt         bool               `json:"cache_prompt,omitzero"`
	ReturnTokens        bool               `json:"return_tokens,omitzero"`
	Samplers            []string           `json:"samplers,omitzero"`
	TimingsPerToken     bool               `json:"timings_per_token,omitzero"`
	PostSamplingProbs   bool               `json:"post_sampling_probs,omitzero"`
	ResponseFields      []string           `json:"response_fields,omitzero"`
	Lora                []any              `json:"lora,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *CompletionRequest) Init(opts genai.Validatable) error {
	var errs []error
	var unsupported []string
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.ChatOptions:
				c.NPredict = v.MaxTokens
				c.Seed = v.Seed
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				c.TopK = v.TopK
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					unsupported = append(unsupported, "ReplyAsJSON")
				}
				if v.DecodeAs != nil {
					unsupported = append(unsupported, "DecodeAs")
				}
				if len(v.Tools) != 0 {
					if v.ToolCallRequired {
						unsupported = append(unsupported, "ToolCallRequired")
					}
					// It's unclear how I'll implement this.
					unsupported = append(unsupported, "Tools")
				}
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

type CompletionResponse struct {
	Index              int64   `json:"index"`
	Content            string  `json:"content"`
	Tokens             []int64 `json:"tokens"`
	IDSlot             int64   `json:"id_slot"`
	Stop               bool    `json:"stop"`
	Model              string  `json:"model"`
	TokensPredicted    int64   `json:"tokens_predicted"`
	TokensEvaluated    int64   `json:"tokens_evaluated"`
	GenerationSettings struct {
		NPredict            int64    `json:"n_predict"`
		Seed                int64    `json:"seed"`
		Temperature         float64  `json:"temperature"`
		DynaTempRange       float64  `json:"dynatemp_range"`
		DynaTempExponent    float64  `json:"dynatemp_exponent"`
		TopK                int64    `json:"top_k"`
		TopP                float64  `json:"top_p"`
		MinP                float64  `json:"min_p"`
		XTCProbability      float64  `json:"xtc_probability"`
		XTCThreshold        float64  `json:"xtc_threshold"`
		TypicalP            float64  `json:"typical_p"`
		RepeatLastN         int64    `json:"repeat_last_n"`
		RepeatPenalty       float64  `json:"repeat_penalty"`
		PresencePenalty     float64  `json:"presence_penalty"`
		FrequencyPenalty    float64  `json:"frequency_penalty"`
		DryMultiplier       float64  `json:"dry_multiplier"`
		DryBase             float64  `json:"dry_base"`
		DryAllowedLength    int64    `json:"dry_allowed_length"`
		DryPenaltyLastN     int64    `json:"dry_penalty_last_n"`
		DrySequenceBreakers []string `json:"dry_sequence_breakers"`
		Mirostat            int32    `json:"mirostat"`
		MirostatTau         float64  `json:"mirostat_tau"`
		MirostatEta         float64  `json:"mirostat_eta"`
		Stop                []string `json:"stop"`
		MaxTokens           int64    `json:"max_tokens"`
		NKeep               int64    `json:"n_keep"`
		NDiscard            int64    `json:"n_discard"`
		IgnoreEos           bool     `json:"ignore_eos"`
		Stream              bool     `json:"stream"`
		LogitBias           []any    `json:"logit_bias"`
		NProbs              int64    `json:"n_probs"`
		MinKeep             int64    `json:"min_keep"`
		Grammar             string   `json:"grammar"`
		GrammarLazy         bool     `json:"grammar_lazy"`
		GrammarTriggers     []string `json:"grammar_triggers"`
		PreservedTokens     []string `json:"preserved_tokens"`
		ChatFormat          string   `json:"chat_format"`
		Samplers            []string `json:"samplers"`
		SpeculativeNMax     int64    `json:"speculative.n_max"`
		SpeculativeNMin     int64    `json:"speculative.n_min"`
		SpeculativePMin     float64  `json:"speculative.p_min"`
		TimingsPerToken     bool     `json:"timings_per_token"`
		PostSamplingProbs   bool     `json:"post_sampling_probs"`
		Lora                []any    `json:"lora"`
		TopNSigma           float64  `json:"top_n_sigma"`
	} `json:"generation_settings"`
	Prompt       string  `json:"prompt"`
	HasNewLine   bool    `json:"has_new_line"`
	Truncated    bool    `json:"truncated"`
	StopType     string  `json:"stop_type"`
	StoppingWord string  `json:"stopping_word"`
	TokensCached int64   `json:"tokens_cached"`
	Timings      Timings `json:"timings"`
}

func (c *CompletionResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		Message: genai.Message{
			Role: genai.Assistant,
			// Mistral Nemo really likes "▁".
			Contents: []genai.Content{{Text: strings.ReplaceAll(c.Content, "\u2581", " ")}},
		},
		Usage: genai.Usage{
			InputTokens:       c.TokensPredicted,
			InputCachedTokens: c.TokensCached,
			OutputTokens:      c.TokensEvaluated,
		},
	}
	// Set FinishReason based on Stop field
	if c.Stop {
		out.FinishReason = "stop"
	}
	return out, nil
}

type Timings struct {
	PromptN             int64   `json:"prompt_n"`
	PromptMS            float64 `json:"prompt_ms"`
	PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
	PromptPerSecond     float64 `json:"prompt_per_second"`
	PredictedN          int64   `json:"predicted_n"`
	PredictedMS         float64 `json:"predicted_ms"`
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	PredictedPerSecond  float64 `json:"predicted_per_second"`
}

type CompletionStreamChunkResponse struct {
	// Always
	Index           int64   `json:"index"`
	Content         string  `json:"content"`
	Tokens          []int64 `json:"tokens"`
	Stop            bool    `json:"stop"`
	IDSlot          int64   `json:"id_slot"`
	TokensPredicted int64   `json:"tokens_predicted"`
	TokensEvaluated int64   `json:"tokens_evaluated"`

	// Last message
	Model              string  `json:"model"`
	GenerationSettings any     `json:"generation_settings"`
	Prompt             string  `json:"prompt"`
	HasNewLine         bool    `json:"has_new_line"`
	Truncated          bool    `json:"truncated"`
	StopType           string  `json:"stop_type"`
	StoppingWord       string  `json:"stopping_word"`
	TokensCached       int64   `json:"tokens_cached"`
	Timings            Timings `json:"timings"`
}

type applyTemplateRequest struct {
	Messages []Message `json:"messages"`
}

func (a *applyTemplateRequest) Init(opts genai.Validatable, msgs genai.Messages) error {
	sp := ""
	if v, ok := opts.(*genai.ChatOptions); ok {
		sp = v.SystemPrompt
	}
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	unsupported := []string{}
	offset := 0
	if sp != "" {
		offset = 1
	}
	a.Messages = make([]Message, len(msgs)+offset)
	if sp != "" {
		a.Messages[0].Role = "system"
		a.Messages[0].Content = sp
	}
	for i := range msgs {
		if err := a.Messages[i+offset].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
		}
	}
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

func (msg *Message) From(m *genai.Message) error {
	// We don't filter the role here.
	msg.Role = string(m.Role)
	if len(m.Contents) != 1 {
		return fmt.Errorf("expected exactly one block, got %d", len(m.Contents))
	}
	if m.Contents[0].Text != "" {
		msg.Content = m.Contents[0].Text
	} else {
		return fmt.Errorf("unsupported content type %v", m.Contents[0])
	}
	return nil
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type applyTemplateResponse struct {
	Prompt string `json:"prompt"`
}

//

type errorResponse struct {
	Error struct {
		Code    int64
		Message string
		Type    string
	} `json:"error"`
}

//

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

// Client implements the REST JSON based API.
type Client struct {
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	baseURL  string
	encoding *PromptEncoding
}

// New creates a new client to talk to a llama-server instance.
//
// encoding is optional.
func New(baseURL string, encoding *PromptEncoding) (*Client, error) {
	if baseURL == "" {
		return nil, errors.New("baseURL is required")
	}
	return &Client{
		Client: httpjson.Client{
			Client: &http.Client{
				Transport: &roundtrippers.Retry{
					Transport: &roundtrippers.RequestID{
						Transport: http.DefaultTransport,
					},
				},
			},
			Lenient: internal.BeLenient,
		},
		baseURL:  baseURL,
		encoding: encoding,
	}, nil
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	// https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
	// Doc mentions Cache:true causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	for i, msg := range msgs {
		if len(msg.Opaque) != 0 {
			return genai.ChatResult{}, fmt.Errorf("message #%d: field Opaque not supported", i)
		}
	}
	rpcin := CompletionRequest{CachePrompt: true}
	if err := rpcin.Init(opts); err != nil {
		return genai.ChatResult{}, err
	}
	if err := c.initPrompt(ctx, &rpcin, opts, msgs); err != nil {
		return genai.ChatResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.ChatResult{}, fmt.Errorf("failed to get llama server response: %w", err)
	}
	slog.DebugContext(ctx, "llm", "prompt tok", rpcout.Timings.PromptN, "gen tok", rpcout.Timings.PredictedN, "prompt tok/ms", rpcout.Timings.PromptPerTokenMS, "gen tok/ms", rpcout.Timings.PredictedPerTokenMS)
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	// TODO: Distinguish between completion and chat. Chat is completion with the template applied.
	in.Stream = false
	return c.post(ctx, c.baseURL+"/completion", in, out)
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.Usage, error) {
	// start := time.Now()
	// Doc mentions Cache:true causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.

	// Check for non-empty Opaque field
	for _, msg := range msgs {
		if len(msg.Opaque) != 0 {
			return genai.Usage{}, fmt.Errorf("Opaque field not supported")
		}
	}

	in := CompletionRequest{CachePrompt: true}
	usage := genai.Usage{}
	var continuableErr error
	if err := in.Init(opts); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return usage, err
		}
	}
	if err := c.initPrompt(ctx, &in, opts, msgs); err != nil {
		return usage, err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	finalUsage := Timings{}
	go func() {
		for msg := range ch {
			if msg.Timings.PredictedN != 0 {
				finalUsage = msg.Timings
			}
			// slog.DebugContext(ctx, "llm", "word", word, "stop", msg.Stop, "prompt tok", msg.Timings.PromptN, "gen tok", msg.Timings.PredictedN, "prompt tok/ms", msg.Timings.PromptPerTokenMS, "gen tok/ms", msg.Timings.PredictedPerTokenMS, "duration", time.Since(start).Round(time.Millisecond))
			if word := msg.Content; word != "" {
				// Mistral Nemo really likes "▁".
				// word = strings.ReplaceAll(msg.Content, "\u2581", " ")
				fragment := genai.MessageFragment{TextFragment: word}
				// Include FinishReason if available (StopType is equivalent to FinishReason)
				if msg.Stop && msg.StopType != "" {
					fragment.FinishReason = msg.StopType
				}
				chunks <- fragment
			}
		}
		end <- nil
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := <-end; err2 != nil {
		err = err2
	}
	usage.InputTokens = finalUsage.PromptN
	usage.OutputTokens = finalUsage.PredictedN
	// usage.InputCachedTokens = finalUsage.TokensCached
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return usage, continuableErr
	}
	return usage, err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	in.Stream = true
	resp, err := c.Client.PostRequest(ctx, c.baseURL+"/completion", nil, in)
	if err != nil {
		return fmt.Errorf("failed to get llama server response: %w", err)
	}
	defer resp.Body.Close()
	for r := bufio.NewReader(resp.Body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) != 0 {
			if err := parseStreamLine(line, out); err != nil {
				return err
			}
		}
	}
}

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
	const prefix = "data: "
	if !bytes.HasPrefix(line, []byte(prefix)) {
		return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
	}
	d := json.NewDecoder(bytes.NewReader(line[len(prefix):]))
	d.DisallowUnknownFields()
	d.UseNumber()
	msg := CompletionStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		return fmt.Errorf("failed to decode llama server response %q: %w", string(line), err)
	}
	out <- msg
	if msg.Stop {
		return nil
	}
	return nil
}

func (c *Client) GetHealth(ctx context.Context) (string, error) {
	msg := healthResponse{}
	if err := c.Client.Get(ctx, c.baseURL+"/health", nil, &msg); err != nil {
		return "", fmt.Errorf("failed to get health response: %w", err)
	}
	return msg.Status, nil
}

// TokenPerformance is the performance for the metrics
type TokenPerformance struct {
	Count    int
	Duration time.Duration
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

// GetMetrics retrieves the performance statistics from the server.
func (c *Client) GetMetrics(ctx context.Context, m *Metrics) error {
	// TODO: Generalize.
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/metrics", nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	// This is not a JSON response.
	resp, err := c.Client.Client.Do(req)
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
		// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/server.cpp
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
		default:
			return fmt.Errorf("unknown metric %q", l)
		}
	}
	return nil
}

func (c *Client) initPrompt(ctx context.Context, in *CompletionRequest, opts genai.Validatable, msgs genai.Messages) error {
	if c.encoding == nil {
		// Use the server to convert the OpenAI style format into a templated form.
		in2 := applyTemplateRequest{}
		var continuableErr error
		if err := in2.Init(opts, msgs); err != nil {
			// If it's an UnsupportedContinuableError, we can continue
			if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
				// Store the error to return later if no other error occurs
				continuableErr = uce
				// Otherwise log the error but continue
			} else {
				return err
			}
		}
		out := applyTemplateResponse{}
		if err := c.post(ctx, c.baseURL+"/apply-template", &in2, &out); err != nil {
			return err
		}
		in.Prompt = out.Prompt
		// Return the continuable error if no other error occurred
		if continuableErr != nil {
			return continuableErr
		}
		return nil
	}

	in.Prompt = c.encoding.BeginOfText
	for _, m := range msgs {
		switch m.Role {
		/* TODO
		case genai.AvailableTools:
			in.Prompt += c.encoding.ToolsAvailableTokenStart + m.Text + c.encoding.ToolsAvailableTokenEnd
		*/
		case "system":
			for _, b := range m.Contents {
				in.Prompt += c.encoding.SystemTokenStart + b.Text + c.encoding.SystemTokenEnd
			}
		case genai.User:
			for _, b := range m.Contents {
				in.Prompt += c.encoding.UserTokenStart + b.Text + c.encoding.UserTokenEnd
			}
			// in.Prompt += c.encoding.ToolCallResultTokenStart + m.Text + c.encoding.ToolCallResultTokenEnd
		case genai.Assistant:
			for _, b := range m.Contents {
				in.Prompt += c.encoding.AssistantTokenStart + b.Text + c.encoding.AssistantTokenEnd
			}
			// in.Prompt += c.encoding.ToolCallTokenStart + m.Text + c.encoding.ToolCallTokenEnd
		default:
			return fmt.Errorf("unexpected role %q", m.Role)
		}
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	resp, err := c.Client.PostRequest(ctx, url, nil, in)
	if err != nil {
		return err
	}
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			return fmt.Errorf("%w: error %d (%s): %s", herr, er.Error.Code, er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %d (%s): %s", er.Error.Code, er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "llamacpp", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "llamacpp", "url", url, "err", err)
		}
		return err
	}
}

var _ genai.ChatProvider = &Client{}
