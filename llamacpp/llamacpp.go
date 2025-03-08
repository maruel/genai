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
	"net/http"
	"strings"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// healthResponse is documented at
// https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md#api-endpoints
type healthResponse struct {
	Status          string
	SlotsIdle       int `json:"slots_idle"`
	SlotsProcessing int `json:"slots_processing"`
}

// https://github.com/ggml-org/llama.cpp/blob/master/examples/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion

type completionRequest struct {
	SystemPrompt     string      `json:"system_prompt,omitempty"`
	Prompt           string      `json:"prompt"`
	Grammar          string      `json:"grammar,omitempty"`
	JSONSchema       interface{} `json:"json_schema,omitempty"`
	Seed             int64       `json:"seed,omitempty"`
	Temperature      float64     `json:"temperature,omitempty"`
	DynaTempRange    float64     `json:"dynatemp_range,omitempty"`
	DynaTempExponent float64     `json:"dynatemp_exponent,omitempty"`
	CachePrompt      bool        `json:"cache_prompt,omitempty"`
	Stream           bool        `json:"stream"`
	// top_k             float64
	// top_p             float64
	// min_p             float64
	NPredict int64 `json:"n_predict,omitempty"` // Maximum number of tokens to predict
	// n_keep            int64
	// stop              []string
	// tfs_z             float64
	// typical_p         float64
	// repeat_penalty    float64
	// repeat_last_n     int64
	// penalize_nl       bool
	// presence_penalty  float64
	// frequency_penalty float64
	// penalty_prompt    *string
	// mirostat          int32
	// mirostat_tau      float64
	// mirostat_eta      float64
	// ignore_eos   bool
	// logit_bias   []interface{}
	// n_probs      int64
	// min_keep     int64
	// image_data   []byte
	// id_slot      int64
	// samplers     []string
}

type completionResponse struct {
	Content            string      `json:"content"`
	Stop               bool        `json:"stop"`
	GenerationSettings interface{} `json:"generation_settings"`
	Model              string      `json:"model"`
	Prompt             string      `json:"prompt"`
	StoppedEOS         bool        `json:"stopped_eos"`
	StoppedLimit       bool        `json:"stopped_limit"`
	StoppedWord        bool        `json:"stopped_word"`
	StoppingWord       string      `json:"stopping_word"`
	Timings            struct {
		// Undocumented:
		PromptN             int64   `json:"prompt_n"`
		PromptMS            float64 `json:"prompt_ms"`
		PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
		PromptPerSecond     float64 `json:"prompt_per_second"`
		PredictedN          int64   `json:"predicted_n"`
		PredictedMS         float64 `json:"predicted_ms"`
		PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
		PredictedPerSecond  float64 `json:"predicted_per_second"`
	}
	TokensCached            int64 `json:"tokens_cached"`
	TokensEvaluated         int64 `json:"tokens_evaluated"`
	Truncated               bool  `json:"truncated"`
	CompletionProbabilities []struct {
		Content string
		Probs   []struct {
			Prob   float64
			TokStr string `json:"tok_str"`
		}
	} `json:"completion_probabilities"`
	// Undocumented:
	HasNewLine      bool  `json:"has_new_line"`
	IDSlot          int64 `json:"id_slot"`
	Index           int64 `json:"index"`
	TokensPredicted int64 `json:"tokens_predicted"`
	Multimodal      bool  `json:"multimodal"`
	// Error case:
	Error errorResponse `json:"error"`
}

//

type errorResponse struct {
	Code    int
	Message string
	Type    string
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
	if err := c.initPrompt(&in, msgs); err != nil {
		return "", err
	}
	out := completionResponse{}
	if err := httpjson.DefaultClient.Post(ctx, c.BaseURL+"/completion", nil, in, &out); err != nil {
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
	if err := c.initPrompt(&in, msgs); err != nil {
		return "", err
	}
	resp, err := httpjson.DefaultClient.PostRequest(ctx, c.BaseURL+"/completion", nil, in)
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
		msg := completionResponse{}
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

func (c *Client) initPrompt(in *completionRequest, msgs []genaiapi.Message) error {
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

var _ genaiapi.CompletionProvider = &Client{}
