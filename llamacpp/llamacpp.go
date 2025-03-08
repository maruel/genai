// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

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

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// healthResponse is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
type healthResponse struct {
	Status          string
	SlotsIdle       int64 `json:"slots_idle"`
	SlotsProcessing int64 `json:"slots_processing"`
}

// https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion

type completionRequest struct {
	// TODO: Can be a string, a list of tokens or a mix.
	Prompt              string   `json:"prompt"`
	Temperature         float64  `json:"temperature,omitempty"`
	DynaTempRange       float64  `json:"dynatemp_range,omitempty"`
	DynaTempExponent    float64  `json:"dynatemp_exponent,omitempty"`
	TopK                int64    `json:"top_k,omitempty"`
	TopP                float64  `json:"top_p,omitempty"`
	MinP                float64  `json:"min_p,omitempty"`
	NPredict            int64    `json:"n_predict,omitempty"` // Maximum number of tokens to predict
	NIndent             int64    `json:"n_indent,omitempty"`
	NKeep               int64    `json:"n_keep,omitempty"`
	Stream              bool     `json:"stream"`
	Stop                []string `json:"stop,omitempty"`
	TypicalP            float64  `json:"typical_p,omitempty"`
	RepeatPenalty       float64  `json:"repeat_penalty,omitempty"`
	RepeatLastN         int64    `json:"repeat_last_n,omitempty"`
	PresencePenalty     float64  `json:"presence_penalty,omitempty"`
	FrequencyPenalty    float64  `json:"frequency_penalty,omitempty"`
	DryMultiplier       float64  `json:"dry_multiplier,omitempty"`
	DryBase             float64  `json:"dry_base,omitempty"`
	DryAllowedLength    int64    `json:"dry_allowed_length,omitempty"`
	DryPenaltyLastN     int64    `json:"dry_penalty_last_n,omitempty"`
	DrySequenceBreakers []string `json:"dry_sequence_breakers,omitempty"`
	XTCProbability      float64  `json:"xtc_probability,omitempty"`
	XTCThreshold        float64  `json:"xtc_threshold,omitempty"`
	Mirostat            int32    `json:"mirostat,omitempty"`
	MirostatTau         float64  `json:"mirostat_tau,omitempty"`
	MirostatEta         float64  `json:"mirostat_eta,omitempty"`
	Grammar             string   `json:"grammar,omitempty"`
	JSONSchema          any      `json:"json_schema,omitempty"`
	Seed                int64    `json:"seed,omitempty"`
	IgnoreEos           bool     `json:"ignore_eos,omitempty"`
	LogitBias           []any    `json:"logit_bias,omitempty"`
	Nprobs              int64    `json:"n_probs,omitempty"`
	MinKeep             int64    `json:"min_keep,omitempty"`
	TMaxPredictMS       int64    `json:"t_max_predict_ms,omitempty"`
	ImageData           []any    `json:"image_data,omitempty"`
	IDSlot              int64    `json:"id_slot,omitempty"`
	CachePrompt         bool     `json:"cache_prompt,omitempty"`
	ReturnTokens        bool     `json:"return_tokens,omitempty"`
	Samplers            []string `json:"samplers,omitempty"`
	TimingsPerToken     bool     `json:"timings_per_token,omitempty"`
	PostSamplingProbs   bool     `json:"post_sampling_probs,omitempty"`
	ResponseFields      []string `json:"response_fields,omitempty"`
	Lora                []any    `json:"lora,omitempty"`
}

/*
	{
		"n_predict": 10,
		"seed": 1,
		"temperature": 0.800000011920929,
		"dynatemp_range": 0.0,
		"dynatemp_exponent": 1.0,
		"top_k": 40,
		"top_p": 0.949999988079071,
		"min_p": 0.05000000074505806,
		"xtc_probability": 0.0,
		"xtc_threshold": 0.10000000149011612,
		"typical_p": 1.0,
		"repeat_last_n": 64,
		"repeat_penalty": 1.0,
		"presence_penalty": 0.0,
		"frequency_penalty": 0.0,
		"dry_multiplier": 0.0,
		"dry_base": 1.75,
		"dry_allowed_length": 2,
		"dry_penalty_last_n": 4096,
		"dry_sequence_breakers": [
			"\n",
			":",
			"\"",
			"*"
		],
		"mirostat": 0,
		"mirostat_tau": 5.0,
		"mirostat_eta": 0.10000000149011612,
		"stop": [],
		"max_tokens": 10,
		"n_keep": 0,
		"n_discard": 0,
		"ignore_eos": false,
		"stream": false,
		"logit_bias": [],
		"n_probs": 0,
		"min_keep": 0,
		"grammar": "",
		"grammar_lazy": false,
		"grammar_triggers": [],
		"preserved_tokens": [],
		"chat_format": "Content-only",
		"samplers": [
			"penalties",
			"dry",
			"top_k",
			"typ_p",
			"top_p",
			"min_p",
			"xtc",
			"temperature"
		],
		"speculative.n_max": 16,
		"speculative.n_min": 0,
		"speculative.p_min": 0.75,
		"timings_per_token": false,
		"post_sampling_probs": false,
		"lora": []
	}
*/
type generationSettings map[string]any

type completionResponse struct {
	Index              int64              `json:"index"`
	Content            string             `json:"content"`
	Tokens             []int64            `json:"tokens"`
	IDSlot             int64              `json:"id_slot"`
	Stop               bool               `json:"stop"`
	Model              string             `json:"model"`
	TokensPredicted    int64              `json:"tokens_predicted"`
	TokensEvaluated    int64              `json:"tokens_evaluated"`
	GenerationSettings generationSettings `json:"generation_settings"`
	Prompt             string             `json:"prompt"`
	HasNewLine         bool               `json:"has_new_line"`
	Truncated          bool               `json:"truncated"`
	StopType           string             `json:"stop_type"`
	StoppingWord       string             `json:"stopping_word"`
	TokensCached       int64              `json:"tokens_cached"`
	Timings            struct {
		PromptN             int64   `json:"prompt_n"`
		PromptMS            float64 `json:"prompt_ms"`
		PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
		PromptPerSecond     float64 `json:"prompt_per_second"`
		PredictedN          int64   `json:"predicted_n"`
		PredictedMS         float64 `json:"predicted_ms"`
		PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
		PredictedPerSecond  float64 `json:"predicted_per_second"`
	} `json:"timings"`
}

type completionStreamResponse struct {
	// Always
	Index           int64   `json:"index"`
	Content         string  `json:"content"`
	Tokens          []int64 `json:"tokens"`
	Stop            bool    `json:"stop"`
	IDSlot          int64   `json:"id_slot"`
	TokensPredicted int64   `json:"tokens_predicted"`
	TokensEvaluated int64   `json:"tokens_evaluated"`

	// Last message
	Model              string `json:"model"`
	GenerationSettings any    `json:"generation_settings"`
	Prompt             string `json:"prompt"`
	HasNewLine         bool   `json:"has_new_line"`
	Truncated          bool   `json:"truncated"`
	StopType           string `json:"stop_type"`
	StoppingWord       string `json:"stopping_word"`
	TokensCached       int64  `json:"tokens_cached"`
	Timings            struct {
		PromptN             int64   `json:"prompt_n"`
		PromptMS            float64 `json:"prompt_ms"`
		PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
		PromptPerSecond     float64 `json:"prompt_per_second"`
		PredictedN          int64   `json:"predicted_n"`
		PredictedMS         float64 `json:"predicted_ms"`
		PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
		PredictedPerSecond  float64 `json:"predicted_per_second"`
	} `json:"timings"`
}

type applyTemplateRequest struct {
	Messages []genaiapi.Message `json:"messages"`
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

type Client struct {
	BaseURL  string
	Encoding *PromptEncoding
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
	in := completionRequest{}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.NPredict = v.MaxTokens
		in.Seed = v.Seed
		in.Temperature = v.Temperature
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}
	// Doc mentions it causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	in.CachePrompt = true
	if err := c.initPrompt(ctx, &in, msgs); err != nil {
		return "", err
	}
	out := completionResponse{}
	if err := c.post(ctx, c.BaseURL+"/completion", in, &out); err != nil {
		return "", fmt.Errorf("failed to get llama server response: %w", err)
	}
	slog.DebugContext(ctx, "llm", "prompt tok", out.Timings.PromptN, "gen tok", out.Timings.PredictedN, "prompt tok/ms", out.Timings.PromptPerTokenMS, "gen tok/ms", out.Timings.PredictedPerTokenMS)
	// Mistral Nemo really likes "▁".
	return strings.ReplaceAll(out.Content, "\u2581", " "), nil
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) (string, error) {
	start := time.Now()
	in := completionRequest{Stream: true}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.NPredict = v.MaxTokens
		in.Seed = v.Seed
		in.Temperature = v.Temperature
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}
	// Doc mentions it causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	in.CachePrompt = true
	if err := c.initPrompt(ctx, &in, msgs); err != nil {
		return "", err
	}
	p := httpjson.DefaultClient
	p.Compress = ""
	resp, err := p.PostRequest(ctx, c.BaseURL+"/completion", nil, in)
	if err != nil {
		return "", fmt.Errorf("failed to get llama server response: %w", err)
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	reply := ""
	for {
		line, err := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if err == io.EOF {
			err = nil
			if len(line) == 0 {
				return reply, nil
			}
		}
		if err != nil {
			return reply, fmt.Errorf("failed to get llama server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		const prefix = "data: "
		if !bytes.HasPrefix(line, []byte(prefix)) {
			return reply, fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		d := json.NewDecoder(bytes.NewReader(line[len(prefix):]))
		d.DisallowUnknownFields()
		d.UseNumber()
		msg := completionStreamResponse{}
		if err = d.Decode(&msg); err != nil {
			return reply, fmt.Errorf("failed to decode llama server response %q: %w", string(line), err)
		}
		word := msg.Content
		slog.DebugContext(ctx, "llm", "word", word, "stop", msg.Stop, "prompt tok", msg.Timings.PromptN, "gen tok", msg.Timings.PredictedN, "prompt tok/ms", msg.Timings.PromptPerTokenMS, "gen tok/ms", msg.Timings.PredictedPerTokenMS, "duration", time.Since(start).Round(time.Millisecond))
		if word != "" {
			// Mistral Nemo really likes "▁".
			word = strings.ReplaceAll(msg.Content, "\u2581", " ")
			words <- word
			reply += word
		}
		if msg.Stop {
			return reply, nil
		}
	}
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) GetHealth(ctx context.Context) (string, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", c.BaseURL+"/health", nil)
	if err != nil {
		return "", fmt.Errorf("failed to create HTTP request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("failed to get health response: %w", err)
	}
	d := json.NewDecoder(resp.Body)
	d.DisallowUnknownFields()
	msg := healthResponse{}
	err = d.Decode(&msg)
	_ = resp.Body.Close()
	if err != nil {
		return msg.Status, fmt.Errorf("failed to decode health response: %w", err)
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
	req, err := http.NewRequestWithContext(ctx, "GET", c.BaseURL+"/metrics", nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	resp, err := http.DefaultClient.Do(req)
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

func (c *Client) initPrompt(ctx context.Context, in *completionRequest, msgs []genaiapi.Message) error {
	if c.Encoding == nil {
		// Use the server to convert the OpenAI style format into a templated form.
		in2 := applyTemplateRequest{Messages: msgs}
		out := applyTemplateResponse{}
		if err := c.post(ctx, c.BaseURL+"/apply-template", &in2, &out); err != nil {
			return err
		}
		in.Prompt = out.Prompt
		return nil
	}

	// Do a quick validation. 1 == available_tools, 2 = system, 3 = rest
	state := 0
	in.Prompt = c.Encoding.BeginOfText
	for i, m := range msgs {
		switch m.Role {
		case genaiapi.AvailableTools:
			if state != 0 || i != 0 {
				return fmt.Errorf("unexpected available_tools message at index %d; state %d", i, state)
			}
			state = 1
			in.Prompt += c.Encoding.ToolsAvailableTokenStart + m.Content + c.Encoding.ToolsAvailableTokenEnd
		case genaiapi.System:
			if state > 1 {
				return fmt.Errorf("unexpected system message at index %d; state %d", i, state)
			}
			state = 2
			in.Prompt += c.Encoding.SystemTokenStart + m.Content + c.Encoding.SystemTokenEnd
		case genaiapi.User:
			state = 3
			in.Prompt += c.Encoding.UserTokenStart + m.Content + c.Encoding.UserTokenEnd
		case genaiapi.Assistant:
			state = 3
			in.Prompt += c.Encoding.AssistantTokenStart + m.Content + c.Encoding.AssistantTokenEnd
		case genaiapi.ToolCall:
			state = 3
			in.Prompt += c.Encoding.ToolCallTokenStart + m.Content + c.Encoding.ToolCallTokenEnd
		case genaiapi.ToolCallResult:
			state = 3
			in.Prompt += c.Encoding.ToolCallResultTokenStart + m.Content + c.Encoding.ToolCallResultTokenEnd
		default:
			return fmt.Errorf("unexpected role %q", m.Role)
		}
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	p := httpjson.DefaultClient
	// Llama.cpp doesn't support compression. lol.
	p.Compress = ""
	resp, err := p.PostRequest(ctx, url, nil, in)
	if err != nil {
		return err
	}
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
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

var _ genaiapi.CompletionProvider = &Client{}
