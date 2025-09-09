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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
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

// ChatRequest is not documented.
//
// Better take a look at oaicompat_chat_params_parse() in
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/utils.hpp
type ChatRequest struct {
	Stream         bool      `json:"stream,omitzero"`
	Model          string    `json:"model,omitzero"`
	MaxTokens      int64     `json:"max_tokens,omitzero"`
	Messages       []Message `json:"messages"`
	ResponseFormat struct {
		Type       string `json:"type,omitzero"` // Default: "text"; "json_object", "json_schema"
		JSONSchema struct {
			Schema *jsonschema.Schema `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Grammar         string `json:"grammar,omitzero"`
	TimingsPerToken bool   `json:"timings_per_token,omitzero"`

	Tools               []Tool   `json:"tools,omitzero"`
	ToolChoice          string   `json:"tool_choice,omitzero"` // Default: "auto"; "none", "required"
	Stop                []string `json:"stop,omitzero"`
	ParallelToolCalls   bool     `json:"parallel_tool_calls,omitzero"`
	AddGenerationPrompt bool     `json:"add_generation_prompt,omitzero"`
	// ReasoningFormat     struct{}   `json:"reasoning_format,omitzero"`
	// EnableThinking      bool       `json:"enable_thinking,omitzero"`
	ChatTemplateKWArgs map[string]string `json:"chat_template_kwargs,omitzero"`
	N                  int64             `json:"n,omitzero"` // Must be 1 anyway.
	Logprobs           bool              `json:"logprobs,omitzero"`
	TopLogprobs        int64             `json:"top_logprobs,omitzero"` // Requires Logprobs:true

	// Prompt              string             `json:"prompt"`
	Temperature         float64  `json:"temperature,omitzero"`
	DynaTempRange       float64  `json:"dynatemp_range,omitzero"`
	DynaTempExponent    float64  `json:"dynatemp_exponent,omitzero"`
	TopK                int64    `json:"top_k,omitzero"`
	TopP                float64  `json:"top_p,omitzero"`
	MinP                float64  `json:"min_p,omitzero"`
	NPredict            int64    `json:"n_predict,omitzero"` // Maximum number of tokens to predict
	NIndent             int64    `json:"n_indent,omitzero"`
	NKeep               int64    `json:"n_keep,omitzero"`
	TypicalP            float64  `json:"typical_p,omitzero"`
	RepeatPenalty       float64  `json:"repeat_penalty,omitzero"`
	RepeatLastN         int64    `json:"repeat_last_n,omitzero"`
	PresencePenalty     float64  `json:"presence_penalty,omitzero"`
	FrequencyPenalty    float64  `json:"frequency_penalty,omitzero"`
	DryMultiplier       float64  `json:"dry_multiplier,omitzero"`
	DryBase             float64  `json:"dry_base,omitzero"`
	DryAllowedLength    int64    `json:"dry_allowed_length,omitzero"`
	DryPenaltyLastN     int64    `json:"dry_penalty_last_n,omitzero"`
	DrySequenceBreakers []string `json:"dry_sequence_breakers,omitzero"`
	XTCProbability      float64  `json:"xtc_probability,omitzero"`
	XTCThreshold        float64  `json:"xtc_threshold,omitzero"`
	Mirostat            int32    `json:"mirostat,omitzero"`
	MirostatTau         float64  `json:"mirostat_tau,omitzero"`
	MirostatEta         float64  `json:"mirostat_eta,omitzero"`
	Seed                int64    `json:"seed,omitzero"`
	IgnoreEos           bool     `json:"ignore_eos,omitzero"`
	LogitBias           []any    `json:"logit_bias,omitzero"`
	Nprobs              int64    `json:"n_probs,omitzero"`
	MinKeep             int64    `json:"min_keep,omitzero"`
	TMaxPredictMS       int64    `json:"t_max_predict_ms,omitzero"`
	ImageData           []any    `json:"image_data,omitzero"`
	IDSlot              int64    `json:"id_slot,omitzero"`
	CachePrompt         bool     `json:"cache_prompt,omitzero"`
	ReturnTokens        bool     `json:"return_tokens,omitzero"`
	Samplers            []string `json:"samplers,omitzero"`
	PostSamplingProbs   bool     `json:"post_sampling_probs,omitzero"`
	ResponseFields      []string `json:"response_fields,omitzero"`
	Lora                []Lora   `json:"lora,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string
	sp := ""
	c.CachePrompt = true
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *genai.OptionsText:
			c.NPredict = v.MaxTokens
			c.Seed = v.Seed
			if v.TopLogprobs > 0 {
				c.TopLogprobs = v.TopLogprobs
				c.Logprobs = true
			}
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			c.TopK = v.TopK
			c.Stop = v.Stop
			if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
			}
		case *genai.OptionsTools:
			if len(v.Tools) != 0 {
				c.Tools = make([]Tool, len(v.Tools))
				for i := range c.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = v.Tools[i].Name
					c.Tools[i].Function.Description = v.Tools[i].Description
					if v.Tools[i].InputSchemaOverride != nil {
						c.Tools[i].Function.Parameters = v.Tools[i].InputSchemaOverride
					} else {
						c.Tools[i].Function.Parameters = v.Tools[i].GetInputSchema()
					}
				}
			}
			if v.WebSearch {
				errs = append(errs, errors.New("unsupported OptionsTools.WebSearch"))
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{
			Role:    "system",
			Content: Contents{{Type: "text", Text: sp}},
		})
	}
	for i := range msgs {
		if len(msgs[i].ToolCallResults) > 1 {
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message %d, tool result %d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		} else {
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			} else {
				c.Messages = append(c.Messages, newMsg)
			}
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

type ChatResponse struct {
	Created           base.Time `json:"created"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Object            string    `json:"object"` // "chat.completion"
	ID                string    `json:"id"`
	Timings           Timings   `json:"timings"`
	Usage             Usage     `json:"usage"`
	Choices           []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Model string `json:"model"` // "gpt-3.5-turbo"
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
			TotalTokens:  c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) == 1 {
		out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
		if err := c.Choices[0].Message.To(&out.Message); err != nil {
			return out, err
		}
		out.Logprobs = c.Choices[0].Logprobs.To()
	}
	return out, nil
}

type Logprobs struct {
	Content []struct {
		ID          int64   `json:"id"`
		Token       string  `json:"token"`
		Bytes       []byte  `json:"bytes"`
		Logprob     float64 `json:"logprob"`
		TopLogprobs []struct {
			ID      int64   `json:"id"`
			Token   string  `json:"token"`
			Bytes   []byte  `json:"bytes"`
			Logprob float64 `json:"logprob"`
		} `json:"top_logprobs"`
	} `json:"content"`
}

func (l *Logprobs) To() [][]genai.Logprob {
	if len(l.Content) == 0 {
		return nil
	}
	out := make([][]genai.Logprob, 0, len(l.Content))
	for _, p := range l.Content {
		lp := make([]genai.Logprob, 1, len(p.TopLogprobs)+1)
		// Intentionally discard Bytes.
		lp[0] = genai.Logprob{ID: p.ID, Text: p.Token, Logprob: p.Logprob}
		for _, tlp := range p.TopLogprobs {
			lp = append(lp, genai.Logprob{ID: tlp.ID, Text: tlp.Token, Logprob: tlp.Logprob})
		}
		out = append(out, lp)
	}
	return out
}

// Tool is not documented.
//
// It's purely handled by the chat templates, thus its real structure varies from model to model.
// See https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp
type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Name        string             `json:"name"`
		Description string             `json:"description"`
		Parameters  *jsonschema.Schema `json:"parameters"`
	} `json:"function"`
}

type Usage struct {
	CompletionTokens int64 `json:"completion_tokens"`
	PromptTokens     int64 `json:"prompt_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type FinishReason string

const (
	FinishedStop      FinishReason = "stop"
	FinishedLength    FinishReason = "length"
	FinishedToolCalls FinishReason = "tool_calls"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishedStop:
		return genai.FinishedStop
	case FinishedLength:
		return genai.FinishedLength
	case FinishedToolCalls:
		return genai.FinishedToolCalls
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

type ChatStreamChunkResponse struct {
	Created           base.Time `json:"created"`
	ID                string    `json:"id"`
	Model             string    `json:"model"` // "gpt-3.5-turbo"
	SystemFingerprint string    `json:"system_fingerprint"`
	Object            string    `json:"object"` // "chat.completion.chunk"
	Choices           []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Delta        struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Logprobs Logprobs `json:"logprobs"`
	} `json:"choices"`
	Usage   Usage   `json:"usage"`
	Timings Timings `json:"timings"`
}

// HealthResponse is documented at
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#get-health-returns-heath-check-result
type HealthResponse struct {
	Status          string `json:"status"`
	SlotsIdle       int64  `json:"slots_idle"`
	SlotsProcessing int64  `json:"slots_processing"`
}

// CompletionRequest is documented at
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
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
	Lora                []Lora             `json:"lora,omitzero"`
}

type Lora struct {
	ID    int64   `json:"id,omitzero"`
	Scale float64 `json:"scale,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *CompletionRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	var errs []error
	var unsupported []string
	c.CachePrompt = true
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *genai.OptionsText:
			c.NPredict = v.MaxTokens
			c.Seed = v.Seed
			if v.TopLogprobs > 0 {
				// TODO: This should be supported.
				unsupported = append(unsupported, "OptionsText.TopLogprobs")
			}
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			c.TopK = v.TopK
			c.Stop = v.Stop
			if v.ReplyAsJSON {
				errs = append(errs, errors.New("implement option ReplyAsJSON"))
			}
			if v.DecodeAs != nil {
				errs = append(errs, errors.New("implement option DecodeAs"))
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opt))
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
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
		ReasoningFormat     string   `json:"reasoning_format"`
		ReasoningInContent  bool     `json:"reasoning_in_content"`
		ThinkingForcedOpen  bool     `json:"thinking_forced_open"`
		Samplers            []string `json:"samplers"`
		SpeculativeNMax     int64    `json:"speculative.n_max"`
		SpeculativeNMin     int64    `json:"speculative.n_min"`
		SpeculativePMin     float64  `json:"speculative.p_min"`
		TimingsPerToken     bool     `json:"timings_per_token"`
		PostSamplingProbs   bool     `json:"post_sampling_probs"`
		Lora                []Lora   `json:"lora"`
		TopNSigma           float64  `json:"top_n_sigma"`
	} `json:"generation_settings"`
	Prompt       string   `json:"prompt"`
	HasNewLine   bool     `json:"has_new_line"`
	Truncated    bool     `json:"truncated"`
	StopType     StopType `json:"stop_type"`
	StoppingWord string   `json:"stopping_word"`
	TokensCached int64    `json:"tokens_cached"`
	Timings      Timings  `json:"timings"`
}

func (c *CompletionResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Message: genai.Message{Replies: []genai.Reply{{Text: c.Content}}},
		Usage: genai.Usage{
			InputTokens:       c.TokensPredicted,
			InputCachedTokens: c.TokensCached,
			OutputTokens:      c.TokensEvaluated,
			FinishReason:      c.StopType.ToFinishReason(),
		},
	}
	return out, nil
}

type StopType string

const (
	StopEOS   StopType = "eos"
	StopLimit StopType = "limit"
	StopWord  StopType = "word"
)

func (s StopType) ToFinishReason() genai.FinishReason {
	switch s {
	case StopEOS:
		return genai.FinishedStop
	case StopLimit:
		return genai.FinishedLength
	case StopWord:
		return genai.FinishedStopSequence
	default:
		if !internal.BeLenient {
			panic(s)
		}
		return genai.FinishReason(s)
	}
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
	Model              string   `json:"model"`
	GenerationSettings struct{} `json:"generation_settings"`
	Prompt             string   `json:"prompt"`
	HasNewLine         bool     `json:"has_new_line"`
	Truncated          bool     `json:"truncated"`
	StopType           StopType `json:"stop_type"`
	StoppingWord       string   `json:"stopping_word"`
	TokensCached       int64    `json:"tokens_cached"`
	Timings            Timings  `json:"timings"`
}

type applyTemplateRequest struct {
	Messages []Message `json:"messages"`
}

func (a *applyTemplateRequest) Init(msgs genai.Messages, opts ...genai.Options) error {
	sp := ""
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		if v, ok := opt.(*genai.OptionsText); ok {
			sp = v.SystemPrompt
		}
	}
	var errs []error
	var unsupported []string

	if sp != "" {
		a.Messages = append(a.Messages, Message{Role: "system", Content: Contents{{Type: "text", Text: sp}}})
	}
	for i := range msgs {
		if len(msgs[i].ToolCallResults) > 1 {
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message %d, tool result %d: %w", i, j, err))
				} else {
					a.Messages = append(a.Messages, newMsg)
				}
			}
		} else {
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			} else {
				a.Messages = append(a.Messages, newMsg)
			}
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// Message is not documented.
//
// You can look at how it's used in oaicompat_chat_params_parse() in
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/utils.hpp
// and common_chat_msgs_parse_oaicompat() in
// https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp
type Message struct {
	Role             string     `json:"role"` // "system", "assistant", "user", "tool"
	Content          Contents   `json:"content,omitzero"`
	ToolCalls        []ToolCall `json:"tool_calls,omitzero"`
	ReasoningContent string     `json:"reasoning_content,omitzero"`
	Name             string     `json:"name,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"`
}

// From must be called with at most one ToolCallResults.
func (m *Message) From(in *genai.Message) error {
	if len(in.ToolCallResults) > 1 {
		return errors.New("internal error")
	}
	switch r := in.Role(); r {
	case "assistant", "user":
		m.Role = r
	case "computer":
		m.Role = "tool"
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	if len(in.Requests) != 0 {
		for i := range in.Requests {
			c := Content{}
			if skip, err := c.FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request %d: %w", i, err)
			} else if !skip {
				m.Content = append(m.Content, c)
			}
		}
	}
	if len(in.Replies) != 0 {
		for i := range in.Replies {
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
					return err
				}
				continue
			}
			c := Content{}
			if skip, err := c.FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply %d: %w", i, err)
			} else if !skip {
				m.Content = append(m.Content, c)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Content = []Content{{Type: "text", Text: in.ToolCallResults[0].Result}}
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	for i := range m.Content {
		out.Replies = make([]genai.Reply, len(m.Content))
		if err := m.Content[i].To(&out.Replies[i]); err != nil {
			return fmt.Errorf("reply %d: %w", i, err)
		}
		if len(m.ReasoningContent) != 0 {
			out.Replies = append(out.Replies, genai.Reply{Reasoning: m.ReasoningContent})
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

type Contents []Content

// UnmarshalJSON implements custom unmarshalling for Contents type
// to handle cases where content could be a string or []Content.
func (c *Contents) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*c = nil
		return nil
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode((*[]Content)(c)); err == nil {
		return nil
	}

	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*c = Contents{{Type: "text", Text: s}}
	return nil
}

// Content is not documented.
//
// You can look at how it's used in oaicompat_chat_params_parse() in
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/utils.hpp
type Content struct {
	Type string `json:"type"` // "text", "image_url", "input_audio"

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`

	InputAudio struct {
		Data   []byte `json:"data,omitzero"`
		Format string `json:"format,omitzero"` // "mp3", "wav"
	} `json:"input_audio,omitzero"`
}

func (c *Content) FromRequest(in *genai.Request) (bool, error) {
	if in.Text != "" {
		c.Type = "text"
		c.Text = in.Text
		return false, nil
	}
	if !in.Doc.IsZero() {
		// Check if this is a text document
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return false, fmt.Errorf("failed to read document: %w", err)
		}
		switch {
		// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			if in.Doc.URL != "" {
				return false, fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Type = "text"
			c.Text = string(data)
		case strings.HasPrefix(mimeType, "audio/"):
			c.Type = "input_audio"
			if in.Doc.URL != "" {
				return false, errors.New("audio doesn't support URLs")
			}
			c.InputAudio.Data = data
			c.InputAudio.Format, _ = strings.CutPrefix(mimeType, "audio/")
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = "image_url"
			if in.Doc.URL != "" {
				c.ImageURL.URL = in.Doc.URL
			} else {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			}
		default:
			return false, fmt.Errorf("mime type %s is unsupported", mimeType)
		}
		return false, nil
	}
	return false, errors.New("unknown Request type")
}

func (c *Content) FromReply(in *genai.Reply) (bool, error) {
	if !in.Citation.IsZero() {
		return false, &internal.BadError{Err: errors.New("field Reply.Citation not supported")}
	}
	if len(in.Opaque) != 0 {
		return false, &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
	}
	if in.Reasoning != "" {
		return true, nil
	}
	if in.Text != "" {
		c.Type = "text"
		c.Text = in.Text
		return false, nil
	}
	if !in.Doc.IsZero() {
		// Check if this is a text document
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return false, fmt.Errorf("failed to read document: %w", err)
		}
		switch {
		// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			if in.Doc.URL != "" {
				return false, fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Type = "text"
			c.Text = string(data)
		case strings.HasPrefix(mimeType, "audio/"):
			c.Type = "input_audio"
			if in.Doc.URL != "" {
				return false, errors.New("audio doesn't support URLs")
			}
			c.InputAudio.Data = data
			c.InputAudio.Format, _ = strings.CutPrefix(mimeType, "audio/")
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = "image_url"
			if in.Doc.URL != "" {
				c.ImageURL.URL = in.Doc.URL
			} else {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			}
		default:
			return false, &internal.BadError{Err: fmt.Errorf("mime type %s is unsupported", mimeType)}
		}
		return false, nil
	}
	return false, &internal.BadError{Err: errors.New("unknown Reply type")}
}

func (c *Content) To(out *genai.Reply) error {
	switch c.Type {
	case "text":
		if c.Text == "" {
			return errors.New("text content is empty")
		}
		out.Text = c.Text
		return nil
	case "image_url":
		return errors.New("implement support for generated images")
	case "input_audio":
		return errors.New("implement support for generated audio")
	default:
		return fmt.Errorf("unexpected content type %q", c.Type)
	}
}

// ToolCall is not documented.
//
// You can look at how it's used in common_chat_msgs_parse_oaicompat() in
// https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp
type ToolCall struct {
	Type     string `json:"type"` // "function"
	Index    int64  `json:"index"`
	ID       string `json:"id,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function"`
}

func (t *ToolCall) From(in *genai.ToolCall) error {
	if len(in.Opaque) != 0 {
		return errors.New("field ToolCall.Opaque not supported")
	}
	t.Type = "function"
	t.ID = in.ID
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
	return nil
}

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

//

type ModelHF struct {
	Name         string   `json:"name"`         // Path to the file
	Model        string   `json:"model"`        // Path to the file
	ModifiedAt   string   `json:"modified_at"`  // Dummy
	Size         string   `json:"size"`         // Dummy
	Digest       string   `json:"digest"`       // Dummy
	Type         string   `json:"type"`         // "model"
	Description  string   `json:"description"`  // Dummy
	Tags         []string `json:"tags"`         // Dummy
	Capabilities []string `json:"capabilities"` // "completion" (hardcoded)
	Parameters   string   `json:"parameters"`   // Dummy
	Details      struct {
		ParentModel       string   `json:"parent_model"`       // Dummy
		Format            string   `json:"format"`             // "gguf" (hardcoded)
		Family            string   `json:"family"`             // Dummy
		Families          []string `json:"families"`           // Dummy
		ParameterSize     string   `json:"parameter_size"`     // Dummy
		QuantizationLevel string   `json:"quantization_level"` // Dummy
	} `json:"details"`
}

type ModelOpenAI struct {
	ID      string    `json:"id"`       // Path to the file
	Object  string    `json:"object"`   // "model"
	Created base.Time `json:"created"`  // Dummy
	OwnedBy string    `json:"owned_by"` // "llamacpp"
	Meta    struct {
		VocabType int64 `json:"vocab_type"` // 1
		NVocab    int64 `json:"n_vocab"`
		NCtxTrain int64 `json:"n_ctx_train"`
		NEmbd     int64 `json:"n_embd"`
		NParams   int64 `json:"n_params"`
		Size      int64 `json:"size"`
	} `json:"meta"`
}

// Model is a synthetic struct combining the information from both ModelHF and ModelOpenAI
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

func (m *Model) Context() int64 {
	return m.OpenAI.Meta.NCtxTrain
}

// ModelsResponse is not documented.
//
// See handle_models() in
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp
type ModelsResponse struct {
	Models []ModelHF     `json:"models"`
	Object string        `json:"object"` // "list"
	Data   []ModelOpenAI `json:"data"`
}

func (m *ModelsResponse) ToModels() []genai.Model {
	var out []genai.Model
	if len(m.Models) != len(m.Data) {
		panic(fmt.Errorf("unexpected response; got different list sizes for models: %d vs %d", len(m.Models), len(m.Data)).Error())
	}
	for i := range m.Models {
		out = append(out, &Model{HF: m.Models[i], OpenAI: m.Data[i]})
	}
	return out
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

// TokenPerformance is the performance for the metrics
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

type applyTemplateResponse struct {
	Prompt string `json:"prompt"`
}

type ErrorResponse struct {
	ErrorVal struct {
		Code    int64  `json:"code"`
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	return fmt.Sprintf("%d (%s): %s", er.ErrorVal.Code, er.ErrorVal.Type, er.ErrorVal.Message)
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

//

// Client implements genai.Provider.
type Client struct {
	impl           base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
	baseURL        string
	completionsURL string
	modelsURL      string
	encoding       *PromptEncoding
}

// New creates a new client to talk to a llama-server instance.
//
// Options Remote defaults to "http://localhost:8080".
//
// Automatic model selection via ModelCheap, ModelGood, ModelSOTA is not supported. It will ask llama-server
// to determine which model is already loaded.
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	var encoding *PromptEncoding
	apiKey := opts.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("LLAMA_API_KEY")
	}
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	baseURL := opts.Remote
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(opts.OutputModalities) != 0 && !slices.Equal(opts.OutputModalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	t := base.DefaultTransport
	if apiKey != "" {
		t = &roundtrippers.Header{
			Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
			Transport: t,
		}
	}
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      baseURL + "/chat/completions",
			ProcessStream:   ProcessStream,
			PreloadedModels: opts.PreloadedModels,
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
		encoding:       encoding,
	}
	var err error
	switch opts.Model {
	case genai.ModelNone:
	case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA, "":
		if c.impl.Model, err = c.selectBestTextModel(ctx); err == nil {
			c.impl.OutputModalities = mod
		}
	default:
		c.impl.Model = opts.Model
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
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
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

func (c *Client) Completion(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
	// Doc mentions Cache:true causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	if err := msgs.Validate(); err != nil {
		return genai.Result{}, err
	}
	for i, msg := range msgs {
		for j, content := range msg.Replies {
			if len(content.Opaque) != 0 {
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

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	in.Stream = false
	return c.impl.DoRequest(ctx, "POST", c.completionsURL, in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
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
			if err := res.Accumulate(f); err != nil {
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

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest) (iter.Seq[CompletionStreamChunkResponse], func() error) {
	in.Stream = true
	resp, err := c.impl.JSONRequest(ctx, "POST", c.completionsURL, in)
	if err != nil {
		return yieldNothing[CompletionStreamChunkResponse], func() error {
			return &internal.BadError{Err: fmt.Errorf("failed to get llama server response: %w", err)}
		}
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return yieldNothing[CompletionStreamChunkResponse], func() error {
			return &internal.BadError{Err: c.impl.DecodeError(c.completionsURL, resp)}
		}
	}
	// Process the stream in a separate goroutine to make sure that when the client iterate, there is already a
	// packet waiting for it. This reduces the overall latency.
	out := make(chan CompletionStreamChunkResponse, 16)
	ch := make(chan error)
	go func() {
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

func (c *Client) GetHealth(ctx context.Context) (string, error) {
	msg, err := c.GetHealthRaw(ctx)
	return msg.Status, err
}

func (c *Client) GetHealthRaw(ctx context.Context) (HealthResponse, error) {
	msg := HealthResponse{}
	if err := c.impl.DoRequest(ctx, "GET", c.baseURL+"/health", nil, &msg); err != nil {
		return msg, fmt.Errorf("failed to get health response: %w", err)
	}
	return msg, nil
}

func (c *Client) Ping(ctx context.Context) error {
	status, err := c.GetHealth(ctx)
	if err == nil && status != "ok" {
		err = fmt.Errorf("server unavailable. status: %q", status)
	}
	return err
}

// GetMetrics retrieves the performance statistics from the server.
func (c *Client) GetMetrics(ctx context.Context, m *Metrics) error {
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/metrics", nil)
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
		default:
			if !internal.BeLenient {
				panic(fmt.Sprintf("unknown metric %q", l))
			}
			return fmt.Errorf("unknown metric %q", l)
		}
	}
	return nil
}

func (c *Client) initPrompt(ctx context.Context, in *CompletionRequest, msgs genai.Messages, opts ...genai.Options) error {
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
			// in.Prompt += c.encoding.ToolCallResultTokenStart + m.Text + c.encoding.ToolCallResultTokenEnd
		case "assistant":
			for _, b := range m.Requests {
				in.Prompt += c.encoding.AssistantTokenStart + b.Text + c.encoding.AssistantTokenEnd
			}
			// in.Prompt += c.encoding.ToolCallTokenStart + m.Text + c.encoding.ToolCallTokenEnd
		default:
			return fmt.Errorf("unexpected role %q", r)
		}
	}
	return nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if pkt.Usage.PromptTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
					u.TotalTokens = pkt.Usage.TotalTokens
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
					Text: pkt.Choices[0].Delta.Content,
					// Reasoning: pkt.Choices[0].Delta.ReasoningContent,
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
func ProcessCompletionStream(chunks iter.Seq[CompletionStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
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
