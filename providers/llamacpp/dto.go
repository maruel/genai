// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the llama-server native API.
//
// Endpoint documentation:
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#api-endpoints
//
// Implementation:
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp

package llamacpp

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

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
			Schema genai.JSONSchema `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Grammar         string `json:"grammar,omitzero"`
	TimingsPerToken bool   `json:"timings_per_token,omitzero"`

	Tools               []Tool                     `json:"tools,omitzero"`
	ToolChoice          string                     `json:"tool_choice,omitzero"` // Default: "auto"; "none", "required"
	Stop                []string                   `json:"stop,omitzero"`
	ParallelToolCalls   bool                       `json:"parallel_tool_calls,omitzero"`
	AddGenerationPrompt bool                       `json:"add_generation_prompt,omitzero"`
	ReasoningFormat     ReasoningFormat            `json:"reasoning_format,omitzero"`
	ChatTemplateKWArgs  map[string]json.RawMessage `json:"chat_template_kwargs,omitzero"`
	N                   int64                      `json:"n,omitzero"` // Must be 1 anyway.
	Logprobs            bool                       `json:"logprobs,omitzero"`
	TopLogprobs         int64                      `json:"top_logprobs,omitzero"` // Requires Logprobs:true

	// Prompt              string             `json:"prompt"`
	Temperature         float64           `json:"temperature,omitzero"`
	DynaTempRange       float64           `json:"dynatemp_range,omitzero"`
	DynaTempExponent    float64           `json:"dynatemp_exponent,omitzero"`
	TopK                int64             `json:"top_k,omitzero"`
	TopP                float64           `json:"top_p,omitzero"`
	MinP                float64           `json:"min_p,omitzero"`
	NPredict            int64             `json:"n_predict,omitzero"` // Maximum number of tokens to predict
	NIndent             int64             `json:"n_indent,omitzero"`
	NKeep               int64             `json:"n_keep,omitzero"`
	TypicalP            float64           `json:"typical_p,omitzero"`
	RepeatPenalty       float64           `json:"repeat_penalty,omitzero"`
	RepeatLastN         int64             `json:"repeat_last_n,omitzero"`
	PresencePenalty     float64           `json:"presence_penalty,omitzero"`
	FrequencyPenalty    float64           `json:"frequency_penalty,omitzero"`
	DryMultiplier       float64           `json:"dry_multiplier,omitzero"`
	DryBase             float64           `json:"dry_base,omitzero"`
	DryAllowedLength    int64             `json:"dry_allowed_length,omitzero"`
	DryPenaltyLastN     int64             `json:"dry_penalty_last_n,omitzero"`
	DrySequenceBreakers []string          `json:"dry_sequence_breakers,omitzero"`
	XTCProbability      float64           `json:"xtc_probability,omitzero"`
	XTCThreshold        float64           `json:"xtc_threshold,omitzero"`
	Mirostat            int32             `json:"mirostat,omitzero"`
	MirostatTau         float64           `json:"mirostat_tau,omitzero"`
	MirostatEta         float64           `json:"mirostat_eta,omitzero"`
	AdaptiveTarget      float64           `json:"adaptive_target,omitzero"`
	AdaptiveDecay       float64           `json:"adaptive_decay,omitzero"`
	TopNSigma           float64           `json:"top_n_sigma,omitzero"`
	Seed                int64             `json:"seed,omitzero"`
	IgnoreEos           bool              `json:"ignore_eos,omitzero"`
	LogitBias           []json.RawMessage `json:"logit_bias,omitzero"`
	Nprobs              int64             `json:"n_probs,omitzero"`
	MinKeep             int64             `json:"min_keep,omitzero"`
	TMaxPredict         base.DurationMS   `json:"t_max_predict_ms,omitzero"`
	ImageData           []base.Unknown    `json:"image_data,omitzero"`
	IDSlot              int64             `json:"id_slot,omitzero"`
	CachePrompt         bool              `json:"cache_prompt,omitzero"`
	ReturnTokens        bool              `json:"return_tokens,omitzero"`
	ReturnProgress      bool              `json:"return_progress,omitzero"`
	Samplers            []string          `json:"samplers,omitzero"`
	PostSamplingProbs   bool              `json:"post_sampling_probs,omitzero"`
	ResponseFields      []string          `json:"response_fields,omitzero"`
	Lora                []Lora            `json:"lora,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
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
		case *genai.GenOptionText:
			sp = v.SystemPrompt
			c.NPredict = v.MaxTokens
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
				s, err := v.DecodeSchema()
				if err != nil {
					errs = append(errs, err)
				} else {
					c.ResponseFormat.JSONSchema.Schema = s
				}
			}
		case *genai.GenOptionTools:
			if len(v.Tools) != 0 {
				c.Tools = make([]Tool, len(v.Tools))
				c.ParallelToolCalls = true
				switch v.Force {
				case genai.ToolCallAny:
					c.ToolChoice = "auto"
				case genai.ToolCallRequired:
					c.ToolChoice = "required"
				case genai.ToolCallNone:
					c.ToolChoice = "none"
				}
				for i := range c.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = v.Tools[i].Name
					c.Tools[i].Function.Description = v.Tools[i].Description
					s, err := v.Tools[i].GetInputSchema()
					if err != nil {
						errs = append(errs, err)
					}
					c.Tools[i].Function.Parameters = s
				}
			}
		case genai.GenOptionSeed:
			c.Seed = int64(v)
		case *GenOption:
			c.ReasoningFormat = v.ReasoningFormat
			if c.ChatTemplateKWArgs == nil {
				c.ChatTemplateKWArgs = map[string]json.RawMessage{}
			}
			// llama-server defaults enable_thinking to true, so we must
			// explicitly disable it when not requested.
			if v.Thinking {
				c.ChatTemplateKWArgs["enable_thinking"] = json.RawMessage("true")
			} else {
				c.ChatTemplateKWArgs["enable_thinking"] = json.RawMessage("false")
			}
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
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

// SetStream sets the streaming mode on the request.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// ChatResponse is the response from the chat completions endpoint.
type ChatResponse struct {
	Created           base.TimeS `json:"created"`
	SystemFingerprint string     `json:"system_fingerprint"`
	Object            string     `json:"object"` // "chat.completion"
	ID                string     `json:"id"`
	Timings           Timings    `json:"timings"`
	Usage             Usage      `json:"usage"`
	Choices           []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Model string `json:"model"` // "gpt-3.5-turbo"
}

// ToResult converts the chat response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       c.Usage.TotalTokens,
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

// Logprobs contains per-token log-probability information.
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

// To converts Logprobs to the genai log-probability format.
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
		Name        string           `json:"name"`
		Description string           `json:"description"`
		Parameters  genai.JSONSchema `json:"parameters"`
	} `json:"function"`
}

// Usage contains token usage statistics.
type Usage struct {
	CompletionTokens    int64 `json:"completion_tokens"`
	PromptTokens        int64 `json:"prompt_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details,omitzero"`
}

// FinishReason describes why the model stopped generating tokens.
type FinishReason string

// ToFinishReason converts to the genai finish reason type.
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

// Valid FinishReason values.
const (
	FinishedStop      FinishReason = "stop"
	FinishedLength    FinishReason = "length"
	FinishedToolCalls FinishReason = "tool_calls"
)

// ReasoningFormat defines the reasoning format supported by llama.cpp.
//
// See https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
type ReasoningFormat string

// Valid ReasoningFormat values.
const (
	ReasoningFormatAuto           ReasoningFormat = "auto"
	ReasoningFormatDeepSeek       ReasoningFormat = "deepseek"
	ReasoningFormatDeepSeekLegacy ReasoningFormat = "deepseek-legacy"
	ReasoningFormatNone           ReasoningFormat = "none"
)

// ChatStreamChunkResponse is a single chunk in a streaming chat response.
type ChatStreamChunkResponse struct {
	Created           base.TimeS `json:"created"`
	ID                string     `json:"id"`
	Model             string     `json:"model"` // "gpt-3.5-turbo"
	SystemFingerprint string     `json:"system_fingerprint"`
	Object            string     `json:"object"` // "chat.completion.chunk"
	Choices           []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Delta        struct {
			Role             string     `json:"role"`
			Content          string     `json:"content"`
			ReasoningContent string     `json:"reasoning_content"`
			ToolCalls        []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Logprobs Logprobs `json:"logprobs"`
	} `json:"choices"`
	Usage          Usage          `json:"usage"`
	Timings        Timings        `json:"timings"`
	PromptProgress PromptProgress `json:"prompt_progress"`
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
	Prompt              string            `json:"prompt"`
	Temperature         float64           `json:"temperature,omitzero"`
	DynaTempRange       float64           `json:"dynatemp_range,omitzero"`
	DynaTempExponent    float64           `json:"dynatemp_exponent,omitzero"`
	TopK                int64             `json:"top_k,omitzero"`
	TopP                float64           `json:"top_p,omitzero"`
	MinP                float64           `json:"min_p,omitzero"`
	NPredict            int64             `json:"n_predict,omitzero"` // Maximum number of tokens to predict
	NIndent             int64             `json:"n_indent,omitzero"`
	NKeep               int64             `json:"n_keep,omitzero"`
	Stream              bool              `json:"stream"`
	Stop                []string          `json:"stop,omitzero"`
	TypicalP            float64           `json:"typical_p,omitzero"`
	RepeatPenalty       float64           `json:"repeat_penalty,omitzero"`
	RepeatLastN         int64             `json:"repeat_last_n,omitzero"`
	PresencePenalty     float64           `json:"presence_penalty,omitzero"`
	FrequencyPenalty    float64           `json:"frequency_penalty,omitzero"`
	DryMultiplier       float64           `json:"dry_multiplier,omitzero"`
	DryBase             float64           `json:"dry_base,omitzero"`
	DryAllowedLength    int64             `json:"dry_allowed_length,omitzero"`
	DryPenaltyLastN     int64             `json:"dry_penalty_last_n,omitzero"`
	DrySequenceBreakers []string          `json:"dry_sequence_breakers,omitzero"`
	XTCProbability      float64           `json:"xtc_probability,omitzero"`
	XTCThreshold        float64           `json:"xtc_threshold,omitzero"`
	Mirostat            int32             `json:"mirostat,omitzero"`
	MirostatTau         float64           `json:"mirostat_tau,omitzero"`
	MirostatEta         float64           `json:"mirostat_eta,omitzero"`
	AdaptiveTarget      float64           `json:"adaptive_target,omitzero"`
	AdaptiveDecay       float64           `json:"adaptive_decay,omitzero"`
	TopNSigma           float64           `json:"top_n_sigma,omitzero"`
	Grammar             string            `json:"grammar,omitzero"`
	JSONSchema          genai.JSONSchema  `json:"json_schema,omitzero"`
	Seed                int64             `json:"seed,omitzero"`
	IgnoreEos           bool              `json:"ignore_eos,omitzero"`
	LogitBias           []json.RawMessage `json:"logit_bias,omitzero"`
	Nprobs              int64             `json:"n_probs,omitzero"`
	MinKeep             int64             `json:"min_keep,omitzero"`
	TMaxPredict         base.DurationMS   `json:"t_max_predict_ms,omitzero"`
	ImageData           []base.Unknown    `json:"image_data,omitzero"`
	IDSlot              int64             `json:"id_slot,omitzero"`
	CachePrompt         bool              `json:"cache_prompt,omitzero"`
	ReturnTokens        bool              `json:"return_tokens,omitzero"`
	ReturnProgress      bool              `json:"return_progress,omitzero"`
	Samplers            []string          `json:"samplers,omitzero"`
	TimingsPerToken     bool              `json:"timings_per_token,omitzero"`
	PostSamplingProbs   bool              `json:"post_sampling_probs,omitzero"`
	ResponseFields      []string          `json:"response_fields,omitzero"`
	Lora                []Lora            `json:"lora,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *CompletionRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
	var errs []error
	var unsupported []string
	c.CachePrompt = true
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *genai.GenOptionText:
			c.NPredict = v.MaxTokens
			if v.TopLogprobs > 0 {
				// TODO: This should be supported.
				unsupported = append(unsupported, "GenOptionText.TopLogprobs")
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
		case genai.GenOptionSeed:
			c.Seed = int64(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// Lora is a LoRA adapter configuration.
type Lora struct {
	ID    int64   `json:"id,omitzero"`
	Scale float64 `json:"scale,omitzero"`
}

// GenerationSettings contains the generation settings returned by the server in completion responses.
type GenerationSettings struct {
	NPredict            int64             `json:"n_predict"`
	Seed                int64             `json:"seed"`
	Temperature         float64           `json:"temperature"`
	DynaTempRange       float64           `json:"dynatemp_range"`
	DynaTempExponent    float64           `json:"dynatemp_exponent"`
	TopK                int64             `json:"top_k"`
	TopP                float64           `json:"top_p"`
	MinP                float64           `json:"min_p"`
	XTCProbability      float64           `json:"xtc_probability"`
	XTCThreshold        float64           `json:"xtc_threshold"`
	TypicalP            float64           `json:"typical_p"`
	RepeatLastN         int64             `json:"repeat_last_n"`
	RepeatPenalty       float64           `json:"repeat_penalty"`
	PresencePenalty     float64           `json:"presence_penalty"`
	FrequencyPenalty    float64           `json:"frequency_penalty"`
	DryMultiplier       float64           `json:"dry_multiplier"`
	DryBase             float64           `json:"dry_base"`
	DryAllowedLength    int64             `json:"dry_allowed_length"`
	DryPenaltyLastN     int64             `json:"dry_penalty_last_n"`
	DrySequenceBreakers []string          `json:"dry_sequence_breakers"`
	Mirostat            int32             `json:"mirostat"`
	MirostatTau         float64           `json:"mirostat_tau"`
	MirostatEta         float64           `json:"mirostat_eta"`
	AdaptiveTarget      float64           `json:"adaptive_target"`
	AdaptiveDecay       float64           `json:"adaptive_decay"`
	Stop                []string          `json:"stop"`
	MaxTokens           int64             `json:"max_tokens"`
	NKeep               int64             `json:"n_keep"`
	NDiscard            int64             `json:"n_discard"`
	IgnoreEos           bool              `json:"ignore_eos"`
	Stream              bool              `json:"stream"`
	LogitBias           []json.RawMessage `json:"logit_bias"`
	NProbs              int64             `json:"n_probs"`
	MinKeep             int64             `json:"min_keep"`
	Grammar             string            `json:"grammar"`
	GrammarLazy         bool              `json:"grammar_lazy"`
	GrammarTriggers     []string          `json:"grammar_triggers"`
	PreservedTokens     []string          `json:"preserved_tokens"`
	ChatFormat          string            `json:"chat_format"`
	ReasoningFormat     string            `json:"reasoning_format"`
	ReasoningInContent  bool              `json:"reasoning_in_content"`
	GenerationPrompt    string            `json:"generation_prompt"`
	BackendSampling     bool              `json:"backend_sampling"`
	SpeculativeTypes    string            `json:"speculative.types"`
	ThinkingForcedOpen  bool              `json:"thinking_forced_open"`
	Samplers            []string          `json:"samplers"`
	SpeculativeNMax     int64             `json:"speculative.n_max"`
	SpeculativeNMin     int64             `json:"speculative.n_min"`
	SpeculativePMin     float64           `json:"speculative.p_min"`
	TimingsPerToken     bool              `json:"timings_per_token"`
	PostSamplingProbs   bool              `json:"post_sampling_probs"`
	Lora                []Lora            `json:"lora"`
	TopNSigma           float64           `json:"top_n_sigma"`
}

// CompletionResponse is the response from the completion endpoint.
type CompletionResponse struct {
	Index              int64              `json:"index"`
	Content            string             `json:"content"`
	Tokens             []int64            `json:"tokens"`
	IDSlot             int64              `json:"id_slot"`
	Stop               bool               `json:"stop"`
	Model              string             `json:"model"`
	TokensPredicted    int64              `json:"tokens_predicted"`
	TokensEvaluated    int64              `json:"tokens_evaluated"`
	GenerationSettings GenerationSettings `json:"generation_settings"`
	Prompt             string             `json:"prompt"`
	HasNewLine         bool               `json:"has_new_line"`
	Truncated          bool               `json:"truncated"`
	StopType           StopType           `json:"stop_type"`
	StoppingWord       string             `json:"stopping_word"`
	TokensCached       int64              `json:"tokens_cached"`
	Timings            Timings            `json:"timings"`
}

// ToResult converts the completion response to a genai.Result.
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

// StopType describes the reason a completion stopped.
type StopType string

// ToFinishReason converts to the genai finish reason type.
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

// Valid StopType values.
const (
	StopEOS   StopType = "eos"
	StopLimit StopType = "limit"
	StopWord  StopType = "word"
)

// Timings contains timing information for prompt processing and prediction.
type Timings struct {
	CacheN             int64           `json:"cache_n"`
	PromptN            int64           `json:"prompt_n"`
	Prompt             base.DurationMS `json:"prompt_ms"`
	PromptPerToken     base.DurationMS `json:"prompt_per_token_ms"`
	PromptPerSecond    float64         `json:"prompt_per_second"`
	PredictedN         int64           `json:"predicted_n"`
	Predicted          base.DurationMS `json:"predicted_ms"`
	PredictedPerToken  base.DurationMS `json:"predicted_per_token_ms"`
	PredictedPerSecond float64         `json:"predicted_per_second"`
	DraftN             int64           `json:"draft_n"`
	DraftNAccepted     int64           `json:"draft_n_accepted"`
}

// PromptProgress reports streaming prompt processing progress.
type PromptProgress struct {
	Total     int64           `json:"total"`
	Cache     int64           `json:"cache"`
	Processed int64           `json:"processed"`
	Time      base.DurationMS `json:"time_ms"`
}

// CompletionStreamChunkResponse is a single chunk in a streaming completion response.
type CompletionStreamChunkResponse struct {
	// Always
	Index           int64          `json:"index"`
	Content         string         `json:"content"`
	Tokens          []int64        `json:"tokens"`
	Stop            bool           `json:"stop"`
	IDSlot          int64          `json:"id_slot"`
	TokensPredicted int64          `json:"tokens_predicted"`
	TokensEvaluated int64          `json:"tokens_evaluated"`
	PromptProgress  PromptProgress `json:"prompt_progress"`

	// Last message
	Model              string       `json:"model"`
	GenerationSettings base.Unknown `json:"generation_settings"`
	Prompt             string       `json:"prompt"`
	HasNewLine         bool         `json:"has_new_line"`
	Truncated          bool         `json:"truncated"`
	StopType           StopType     `json:"stop_type"`
	StoppingWord       string       `json:"stopping_word"`
	TokensCached       int64        `json:"tokens_cached"`
	Timings            Timings      `json:"timings"`
}

type applyTemplateRequest struct {
	Messages []Message `json:"messages"`
}

func (a *applyTemplateRequest) Init(msgs genai.Messages, opts ...genai.GenOption) error {
	sp := ""
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		if v, ok := opt.(*genai.GenOptionText); ok {
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

// To converts a Message to a genai.Message.
func (m *Message) To(out *genai.Message) error {
	if m.ReasoningContent != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.ReasoningContent})
	}
	out.Replies = slices.Grow(out.Replies, len(m.Content))
	for i := range m.Content {
		out.Replies = append(out.Replies, genai.Reply{})
		if err := m.Content[i].To(&out.Replies[len(out.Replies)-1]); err != nil {
			return fmt.Errorf("reply %d: %w", i, err)
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

// Contents is a list of Content items that may be unmarshalled from a string or array.
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
	if s != "" {
		*c = Contents{{Type: "text", Text: s}}
	}
	return nil
}

// Content is not documented.
//
// You can look at how it's used in oaicompat_chat_params_parse() in
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/utils.hpp
type Content struct {
	Type string `json:"type"` // "text", "image_url", "input_audio", "input_video", "video_url"

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`

	InputVideo VideoInput `json:"input_video,omitzero"`
	VideoURL   VideoInput `json:"video_url,omitzero"`

	InputAudio struct {
		Data   []byte `json:"data,omitzero"`
		Format string `json:"format,omitzero"` // "mp3", "wav"
	} `json:"input_audio,omitzero"`
}

// FromRequest populates a Content from a genai.Request.
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
		case mimeType == "audio/mpeg":
			c.Type = "input_audio"
			if in.Doc.URL != "" {
				return false, errors.New("audio doesn't support URLs")
			}
			c.InputAudio.Data = data
			c.InputAudio.Format = "mp3"
		case mimeType == "audio/wav":
			c.Type = "input_audio"
			if in.Doc.URL != "" {
				return false, errors.New("audio doesn't support URLs")
			}
			c.InputAudio.Data = data
			c.InputAudio.Format = "wav"
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

// FromReply populates a Content from a genai.Reply.
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
		case mimeType == "audio/mpeg":
			c.Type = "input_audio"
			if in.Doc.URL != "" {
				return false, errors.New("audio doesn't support URLs")
			}
			c.InputAudio.Data = data
			c.InputAudio.Format = "mp3"
		case mimeType == "audio/wav":
			c.Type = "input_audio"
			if in.Doc.URL != "" {
				return false, errors.New("audio doesn't support URLs")
			}
			c.InputAudio.Data = data
			c.InputAudio.Format = "wav"
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

// To converts a Content to a genai.Reply.
func (c *Content) To(out *genai.Reply) error {
	switch c.Type {
	case "text":
		// Allow empty text content. Some models (e.g. Qwen thinking) return
		// empty text content alongside reasoning_content in non-streaming
		// mode; the reasoning is handled separately in Message.To().
		if c.Text == "" {
			return nil
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

// VideoInput identifies a video by URL or a data URI containing base64 media.
type VideoInput struct {
	URL  string `json:"url,omitzero"`
	Data string `json:"data,omitzero"`
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

// From populates a ToolCall from a genai.ToolCall.
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

// To converts a ToolCall to a genai.ToolCall.
func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

// ModelHF is the HuggingFace-style model metadata from the llama-server.
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

// ModelOpenAI is the OpenAI-compatible model metadata from the llama-server.
type ModelOpenAI struct {
	ID      string     `json:"id"`       // Path to the file
	Object  string     `json:"object"`   // "model"
	Created base.TimeS `json:"created"`  // Dummy
	OwnedBy string     `json:"owned_by"` // "llamacpp"
	Meta    struct {
		VocabType int64  `json:"vocab_type"` // 1
		NVocab    int64  `json:"n_vocab"`
		NCtx      int64  `json:"n_ctx"`
		NCtxTrain int64  `json:"n_ctx_train"`
		NEmbd     int64  `json:"n_embd"`
		NParams   int64  `json:"n_params"`
		FType     string `json:"ftype"` // "Q4_K - Medium"
		Size      int64  `json:"size"`
	} `json:"meta"`
	Aliases []string `json:"aliases,omitzero"`
	Tags    []string `json:"tags,omitzero"`
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

// ToModels converts the response to a list of genai.Model.
func (m *ModelsResponse) ToModels() []genai.Model {
	if len(m.Models) != len(m.Data) {
		panic(fmt.Errorf("unexpected response; got different list sizes for models: %d vs %d", len(m.Models), len(m.Data)).Error())
	}
	out := make([]genai.Model, 0, len(m.Models))
	for i := range m.Models {
		out = append(out, &Model{HF: m.Models[i], OpenAI: m.Data[i]})
	}
	return out
}

type applyTemplateResponse struct {
	Prompt string `json:"prompt"`
}

// ErrorResponse is the error response from the llama-server API.
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

// IsAPIError implements base.ErrAPI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// DecisionContent is text, a JSON object, or a JSON array.
//
// It is implemented by Text, Object and Array. It marshals to the value itself, not to an object with
// fields. Use nil when the field is optional and is left unset.
//
// It is used for state, instructions and criteria descriptions. It is meant to be sent; the API only
// returns it in Answer.Legend, decoded back by ScoreLegend, since an interface cannot be decoded on its
// own.
type DecisionContent interface {
	// Validate ensures the content is valid.
	Validate() error
	// content restricts DecisionContent to the types of this package.
	content()
}

// Text is DecisionContent that is a JSON string.
type Text string

// content implements DecisionContent.
func (Text) content() {}

// Validate implements internal.Validatable.
//
// Any text is valid, including the empty string, the API is the one that decides whether a specific field
// may be empty.
func (Text) Validate() error {
	return nil
}

// Object is DecisionContent that is a JSON object.
//
// The values are any, like genai.Reply.Opaque: they must be JSON encodable. Go has no recursive JSON
// value type that does not require wrapping every scalar, so a nested map[string]any, []any or a struct
// with JSON tags is passed as is. It is the equivalent of the SDKs' Mapping[str, JSONValue | None].
type Object map[string]any

// content implements DecisionContent.
func (Object) content() {}

// Validate implements internal.Validatable.
//
// Every value is validated so that the error names the offending key.
func (o Object) Validate() error {
	if o == nil {
		return errors.New("Object is nil, use nil to leave the field unset")
	}
	// TODO: Validate the values recursively to report the full path, e.g. `key "ticket": index 3: ...`.
	// This needs a depth limit because a map can contain itself, and anything that is not map[string]any,
	// []any, Object or Array must stay delegated to encoding/json.
	var errs []error
	for _, k := range slices.Sorted(maps.Keys(o)) {
		// The values are any, so encoding/json is the authority on what they may be.
		if _, err := json.Marshal(o[k]); err != nil {
			errs = append(errs, fmt.Errorf("key %q: %w", k, err))
		}
	}
	return errors.Join(errs...)
}

// Array is DecisionContent that is a JSON array.
//
// The values are any, like Object: they must be JSON encodable. They are not restricted to DecisionContent items.
type Array []any

// content implements DecisionContent.
func (Array) content() {}

// Validate implements internal.Validatable.
//
// Every item is validated so that the error names the offending index.
func (a Array) Validate() error {
	if a == nil {
		return errors.New("Array is nil, use nil to leave the field unset")
	}
	// TODO: Validate the items recursively, see Object.Validate for the constraints.
	var errs []error
	for i := range a {
		// The values are any, so encoding/json is the authority on what they may be.
		if _, err := json.Marshal(a[i]); err != nil {
			errs = append(errs, fmt.Errorf("index %d: %w", i, err))
		}
	}
	return errors.Join(errs...)
}

// QuestionType is the type of a Question.
type QuestionType string

// Question types.
const (
	// QuestionNoul is a yes/no question. The answer is the probability that the answer is yes.
	QuestionNoul QuestionType = "noul"
	// QuestionChoice picks one option from a set defined by the question.
	QuestionChoice QuestionType = "choice"
	// QuestionScore rates the state along an ordered rubric.
	QuestionScore QuestionType = "score"
)

// Question is a typed question about a state.
//
// Exactly one of Noul, Choice or Score can be set, the one matching Type. Instructions is shared by the
// three types.
//
// Type is explicit instead of inferred from the field that is set: a choice or score question whose
// criteria are nil at runtime, which append and conditional map building produce, would otherwise be
// silently asked as a noul question instead of being rejected.
//
// Recommended reading: https://docs.typesafe.ai/primitives/advanced
type Question struct {
	// Type is how the state is evaluated. It selects which of Noul, Choice or Score must be set.
	Type QuestionType
	// Instructions is what the model should decide or rate. It is required for every question.
	Instructions DecisionContent
	// Noul describes what the yes and the no outcomes mean. It can only be set for QuestionNoul, where it
	// is optional.
	//
	// Recommended reading: https://docs.typesafe.ai/primitives/noul
	Noul *NoulCriteria
	// Choice maps the options of a choice question to their descriptions. It can only be set for
	// QuestionChoice, where at least one option is required. Use nil for an option that needs no extra
	// detail.
	//
	// Recommended reading: https://docs.typesafe.ai/primitives/choice
	Choice map[string]DecisionContent
	// Score lists the levels of a score question rubric, in order, starting at level 0. It can only be set
	// for QuestionScore, where 2 to 10 levels are required. Use nil for an undescribed level.
	//
	// Recommended reading: https://docs.typesafe.ai/primitives/score
	Score []DecisionContent
}

// Validate implements internal.Validatable.
func (q Question) Validate() error {
	var errs []error
	if q.Instructions == nil {
		errs = append(errs, errors.New("field Instructions: is required"))
	} else {
		if err := validateContent(q.Instructions); err != nil {
			errs = append(errs, fmt.Errorf("field Instructions: %w", err))
		}
	}
	switch q.Type {
	case QuestionNoul:
		if q.Choice != nil || q.Score != nil {
			errs = append(errs, errors.New("fields Choice and Score: can't be set on a noul question"))
		}
		if q.Noul != nil {
			if err := q.Noul.Validate(); err != nil {
				errs = append(errs, err)
			}
		}
	case QuestionChoice:
		if q.Noul != nil || q.Score != nil {
			errs = append(errs, errors.New("fields Noul and Score: can't be set on a choice question"))
		}
		if len(q.Choice) == 0 {
			errs = append(errs, errors.New("field Choice: at least one option is required"))
		}
		for _, name := range slices.Sorted(maps.Keys(q.Choice)) {
			if v := q.Choice[name]; v != nil {
				if err := validateContent(v); err != nil {
					errs = append(errs, fmt.Errorf("field Choice[%s]: %w", name, err))
				}
			}
		}
	case QuestionScore:
		if q.Noul != nil || q.Choice != nil {
			errs = append(errs, errors.New("fields Noul and Choice: can't be set on a score question"))
		}
		if len(q.Score) < 2 || len(q.Score) > 10 {
			errs = append(errs, errors.New("field Score: 2 to 10 levels are required"))
		}
		for i, c := range q.Score {
			if c != nil {
				if err := validateContent(c); err != nil {
					errs = append(errs, fmt.Errorf("field Score[%d]: %w", i, err))
				}
			}
		}
	default:
		errs = append(errs, fmt.Errorf("field Type: must be %q, %q or %q, got %q", QuestionNoul, QuestionChoice, QuestionScore, q.Type))
	}
	return errors.Join(errs...)
}

// MarshalJSON implements json.Marshaler.
func (q Question) MarshalJSON() ([]byte, error) {
	// Criteria stays nil, and so is omitted, for a noul question without criteria.
	var criteria json.RawMessage
	var err error
	switch q.Type {
	case QuestionNoul:
		if q.Noul != nil {
			criteria, err = json.Marshal(q.Noul)
		}
	case QuestionChoice:
		criteria, err = json.Marshal(q.Choice)
	case QuestionScore:
		criteria, err = json.Marshal(q.Score)
	default:
		return nil, fmt.Errorf("unknown question type %q", q.Type)
	}
	if err != nil {
		return nil, err
	}
	return json.Marshal(questionJSON{Type: q.Type, Instructions: q.Instructions, Criteria: criteria})
}

// Questions is a set of questions to ask about one state, keyed by the name to report each answer under.
//
// The names are chosen by the caller. They are not sent to the model, they are only used to key the
// answers.
//
// It is passed as genai.GenOptionText.DecodeAs when the questions are not declared by a struct of Noul,
// Choice and Score fields. The answers then decode into Answers.
type Questions map[string]Question

// QuestionsFrom returns the questions declared by the fields of v.
//
// v must be a pointer to a struct whose fields are Noul, Choice or Score; the field name, or its `json`
// tag, is used as the question name, and a `json:"-"` field is skipped. It is what GenSync does with a
// struct passed as genai.GenOptionText.DecodeAs, exposed so the questions can be printed, reviewed, or
// tuned, and then passed as DecodeAs themselves.
func QuestionsFrom(v any) (Questions, error) {
	t := reflect.TypeOf(v)
	if t == nil || t.Kind() != reflect.Pointer || t.Elem().Kind() != reflect.Struct || reflect.ValueOf(v).IsNil() {
		return nil, fmt.Errorf("%T: must be a pointer to a struct of Noul, Choice or Score fields", v)
	}
	t = t.Elem()
	val := reflect.ValueOf(v).Elem()
	out := make(Questions, t.NumField())
	for i := range t.NumField() {
		f := t.Field(i)
		if name, skip, err := questionName(&f); err != nil {
			return nil, err
		} else if skip {
			continue
		} else {
			if _, ok := out[name]; ok {
				return nil, fmt.Errorf("field %s: duplicate question name %q", f.Name, name)
			}
			switch f.Type {
			case reflect.TypeFor[Noul]():
				n := val.Field(i).Addr().Interface().(*Noul)
				out[name] = Question{Type: QuestionNoul, Instructions: n.Instructions, Noul: n.Criteria}
			case reflect.TypeFor[Choice]():
				c := val.Field(i).Addr().Interface().(*Choice)
				out[name] = Question{Type: QuestionChoice, Instructions: c.Instructions, Choice: c.Criteria}
			case reflect.TypeFor[Score]():
				s := val.Field(i).Addr().Interface().(*Score)
				out[name] = Question{Type: QuestionScore, Instructions: s.Instructions, Score: s.Criteria}
			default:
				return nil, fmt.Errorf("field %s: must be a Noul, a Choice or a Score, got a %s", f.Name, f.Type)
			}
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("%T: no Noul, Choice or Score field to ask", v)
	}
	if err := out.Validate(); err != nil {
		return nil, err
	}
	return out, nil
}

// Validate ensures the questions are valid.
func (q Questions) Validate() error {
	if len(q) == 0 {
		return errors.New("at least one question is required")
	}
	var errs []error
	for _, name := range slices.Sorted(maps.Keys(q)) {
		if err := q[name].Validate(); err != nil {
			errs = append(errs, fmt.Errorf("question %q: %w", name, err))
		}
	}
	return errors.Join(errs...)
}

// NoulCriteria describes what the yes and the no answers of a noul question mean.
type NoulCriteria struct {
	// True describes what a yes (value near 1) means.
	True DecisionContent `json:"true,omitzero"`
	// False describes what a no (value near 0) means.
	False DecisionContent `json:"false,omitzero"`
}

// Validate ensures the criteria are valid.
func (c *NoulCriteria) Validate() error {
	if c == nil {
		return errors.New("field Criteria: is nil")
	}
	var errs []error
	if c.True != nil {
		if err := validateContent(c.True); err != nil {
			errs = append(errs, fmt.Errorf("field Criteria.True: %w", err))
		}
	}
	if c.False != nil {
		if err := validateContent(c.False); err != nil {
			errs = append(errs, fmt.Errorf("field Criteria.False: %w", err))
		}
	}
	return errors.Join(errs...)
}

// questionJSON is the wire representation of a Question.
//
// Criteria is the encoded form of whichever of the Noul, Choice and Score fields that Question.Type
// selects. The three do not share a Go type, so Question.MarshalJSON encodes the one that applies. It is
// left nil, and so omitted, for a noul question without criteria.
type questionJSON struct {
	Type         QuestionType    `json:"type"`
	Instructions DecisionContent `json:"instructions,omitzero"`
	Criteria     json.RawMessage `json:"criteria,omitzero"`
}

// Answers holds the answer to each question, keyed by the question name.
type Answers map[string]Answer

// Answer is the answer to a Question.
//
// It is a union discriminated by Type. Only the fields that match Type are set, and MarshalJSON writes
// only those fields, with the shape the API uses, so a probability or a score of 0 is not dropped.
//
// An answer type the server adds later is kept as is instead of making the whole reply unusable.
type Answer struct {
	// Type is the type of the question this is the answer to.
	Type QuestionType `json:"type"`
	// Noul is the probability that the answer is yes, from 0 to 1. It is set for QuestionNoul.
	Noul float64 `json:"noul,omitzero"`
	// Choice is the highest probability option. It is set for QuestionChoice.
	Choice string `json:"choice,omitzero"`
	// Score is the probability weighted value across the levels. It can land between levels. It is set for
	// QuestionScore.
	Score float64 `json:"score,omitzero"`
	// Confidence is how certain the model is, derived from Probabilities. It is set for QuestionChoice and
	// QuestionScore.
	Confidence float64 `json:"confidence,omitzero"`
	// Probabilities maps each option, or each level as a string key, to its probability. It is set for
	// QuestionChoice and QuestionScore.
	Probabilities map[string]float64 `json:"probabilities,omitzero"`
	// Legend maps each level as a string key to the description passed in Question.Score for that level.
	// It is set for QuestionScore.
	Legend ScoreLegend `json:"legend,omitzero"`

	// raw is the answer as returned by the API. It is only set when Type is not one of the known types, so
	// that a new answer type does not make the whole reply unusable.
	raw json.RawMessage
}

// UnmarshalJSON implements json.Unmarshaler.
func (a *Answer) UnmarshalJSON(b []byte) error {
	t := answerTypeJSON{}
	if err := json.Unmarshal(b, &t); err != nil {
		return err
	}
	switch t.Type {
	case QuestionNoul, QuestionChoice, QuestionScore:
	default:
		// Answer type added by the server after this client was written. Keep it as-is instead of
		// making the whole reply unusable.
		a.Type = t.Type
		a.raw = append(json.RawMessage(nil), b...)
		return nil
	}
	type alias Answer
	v := alias{}
	if err := internal.UnmarshalJSON(b, &v); err != nil {
		return err
	}
	*a = Answer(v)
	return nil
}

// answerTypeJSON is the wire representation of an answer read for its type only.
type answerTypeJSON struct {
	Type QuestionType `json:"type"`
}

// MarshalJSON implements json.Marshaler.
//
// The fields of the union that do not match Type are omitted, even when they are zero, so that a score
// of 0 or a probability of 0 is preserved.
//
//nolint:gocritic // hugeParam: a value receiver is required to marshal the values of an Answers map.
func (a Answer) MarshalJSON() ([]byte, error) {
	switch a.Type {
	case QuestionNoul:
		return json.Marshal(noulAnswerJSON{Type: a.Type, Noul: a.Noul})
	case QuestionChoice:
		return json.Marshal(choiceAnswerJSON{
			Type: a.Type, Choice: a.Choice, Confidence: a.Confidence, Probabilities: a.Probabilities,
		})
	case QuestionScore:
		return json.Marshal(scoreAnswerJSON{
			Type: a.Type, Score: a.Score, Confidence: a.Confidence, Legend: a.Legend, Probabilities: a.Probabilities,
		})
	default:
		if len(a.raw) != 0 {
			// Answer type added by the server after this client was written.
			return a.raw, nil
		}
		return nil, fmt.Errorf("unknown answer type %q", a.Type)
	}
}

// noulAnswerJSON is the wire representation of a Noul answer.
type noulAnswerJSON struct {
	Type QuestionType `json:"type"`
	Noul float64      `json:"noul"`
}

// choiceAnswerJSON is the wire representation of a Choice answer.
type choiceAnswerJSON struct {
	Type          QuestionType       `json:"type"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}

// scoreAnswerJSON is the wire representation of a Score answer.
type scoreAnswerJSON struct {
	Type          QuestionType               `json:"type"`
	Score         float64                    `json:"score"`
	Confidence    float64                    `json:"confidence"`
	Legend        map[string]DecisionContent `json:"legend"`
	Probabilities map[string]float64         `json:"probabilities"`
}

// ScoreLegend is the description of each level of a score answer, keyed by the level.
//
// It is the Question.Score criteria of the question, echoed back by the API. The values are decoded as
// Text, Object or Array; a nil value is a level the caller left undescribed.
type ScoreLegend map[string]DecisionContent

// UnmarshalJSON implements json.Unmarshaler.
//
// DecisionContent is an interface, so encoding/json cannot decode it on its own; each level is decoded into the
// variant matching its JSON token here.
func (l *ScoreLegend) UnmarshalJSON(b []byte) error {
	raw := map[string]json.RawMessage{}
	if err := internal.UnmarshalJSON(b, &raw); err != nil {
		return err
	}
	if raw == nil {
		*l = nil
		return nil
	}
	out := make(ScoreLegend, len(raw))
	for k, v := range raw {
		// A level the caller left undescribed is null.
		if bytes.Equal(v, []byte("null")) {
			out[k] = nil
			continue
		}
		c, err := contentFromJSON(v)
		if err != nil {
			return fmt.Errorf("level %q: %w", k, err)
		}
		out[k] = c
	}
	*l = out
	return nil
}

// contentFromJSON decodes one JSON value into its DecisionContent variant.
//
// The variant is picked from the JSON token, not from the Go value a decoder would produce for an any, so
// what it accepts is unambiguous.
func contentFromJSON(b []byte) (DecisionContent, error) {
	b = bytes.TrimSpace(b)
	if len(b) == 0 {
		return nil, errors.New("is empty")
	}
	switch b[0] {
	case '"':
		t := Text("")
		if err := internal.UnmarshalJSON(b, &t); err != nil {
			return nil, err
		}
		return t, nil
	case '{':
		o := Object{}
		if err := internal.UnmarshalJSON(b, &o); err != nil {
			return nil, err
		}
		return o, nil
	case '[':
		a := Array{}
		if err := internal.UnmarshalJSON(b, &a); err != nil {
			return nil, err
		}
		return a, nil
	default:
		return nil, fmt.Errorf("expected a string, a JSON object or a JSON array, got %s", b)
	}
}

// SystemOneRequest is a native /v1/systemone request.
// Questions and responses use the System One API types. llama.cpp additionally accepts images,
// requires instructions for every question, and restricts scores to 2 to 10 levels.
type SystemOneRequest struct {
	State DecisionContent `json:"state"`
	// Model is optional for a server with one loaded model; router mode uses it to select a model.
	Model     string    `json:"model,omitzero"`
	Questions Questions `json:"questions"`
	// Images contains inline data URLs. Remote image URLs are not supported.
	Images []string `json:"images,omitzero"`
}

// From sets the state and images from a single message.
// Text and JSON documents supply the state. Image documents must be inline.
// An image-only message supplies an empty text state.
func (r *SystemOneRequest) From(msg *genai.Message) error {
	if len(msg.Replies) != 0 || len(msg.ToolCallResults) != 0 {
		return errors.New("system one requires the full state instead of assistant replies or tool call results")
	}
	arr := make(Array, 0, len(msg.Requests))
	var images []string
	for i := range msg.Requests {
		in := &msg.Requests[i]
		if in.Doc.IsZero() {
			if in.Text == "" {
				return fmt.Errorf("request #%d: must contain text, JSON or an inline image", i)
			}
			arr = append(arr, Text(in.Text))
			continue
		}
		if err := in.Doc.Validate(); err != nil {
			return fmt.Errorf("request #%d: %w", i, err)
		}
		if in.Doc.URL != "" {
			return errors.New("system one documents must be inline")
		}
		mt, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return fmt.Errorf("request #%d: %w", i, err)
		}
		switch {
		case mt == "application/json":
			if in.Text != "" {
				return fmt.Errorf("request #%d: text and a JSON document cannot be combined in one request", i)
			}
			state, err := contentFromJSON(data)
			if err != nil {
				return fmt.Errorf("request #%d: invalid JSON state: %w", i, err)
			}
			arr = append(arr, state)
		case strings.HasPrefix(mt, "image/"):
			images = append(images, "data:"+mt+";base64,"+base64.StdEncoding.EncodeToString(data))
			if in.Text != "" {
				arr = append(arr, Text(in.Text))
			}
		default:
			return fmt.Errorf("request #%d: unsupported document type %q; use a .json document or an inline image", i, mt)
		}
	}
	switch len(arr) {
	case 0:
		if len(images) == 0 {
			return errors.New("the message must have the state as text, a JSON document or an inline image")
		}
		r.State = Text("")
	case 1:
		r.State = arr[0].(DecisionContent)
	default:
		r.State = arr
	}
	r.Images = images
	return nil
}

// FromOptions sets the questions to ask from the options.
//
// They come from genai.GenOptionText.DecodeAs, a pointer to a struct of Noul, Choice and Score fields
// or a Questions.
func (r *SystemOneRequest) FromOptions(opts ...genai.GenOption) error {
	for _, opt := range opts {
		switch v := opt.(type) {
		case *genai.GenOptionText:
			if err := v.Validate(); err != nil {
				return err
			}
			if unsupported := unsupportedTextOptions(v); len(unsupported) != 0 {
				return &base.ErrNotSupported{Options: unsupported}
			}
			switch d := v.DecodeAs.(type) {
			case nil:
				return errors.New("field DecodeAs: a pointer to a struct of Noul, Choice and Score fields, or a Questions, is required to declare the questions")
			case Questions:
				if err := d.Validate(); err != nil {
					return fmt.Errorf("field DecodeAs: %w", err)
				}
				r.Questions = d
			default:
				q, err := QuestionsFrom(v.DecodeAs)
				if err != nil {
					return fmt.Errorf("field DecodeAs: %w", err)
				}
				r.Questions = q
			}
		default:
			return &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", opt)}}
		}
	}
	if r.Questions == nil {
		return errors.New("the questions to ask are required, pass *genai.GenOptionText with DecodeAs")
	}
	return nil
}

// Validate checks the native endpoint's state, question and image requirements.
func (r *SystemOneRequest) Validate() error {
	if r.State == nil {
		return errors.New("state is required")
	}
	if err := r.State.Validate(); err != nil {
		return fmt.Errorf("state: %w", err)
	}
	if err := r.Questions.Validate(); err != nil {
		return err
	}
	for i, img := range r.Images {
		if !strings.HasPrefix(img, "data:image/") || !strings.Contains(img, ";base64,") {
			return fmt.Errorf("image #%d: must be an inline image data URL", i)
		}
	}
	return nil
}

// SystemOneResponse is the response of a POST /v1/systemone request.
type SystemOneResponse struct {
	// Model identifies the loaded decision model that answered.
	Model string `json:"model"`
	// Answers holds one answer per question, keyed by the question name.
	Answers Answers `json:"answers"`
	// Usage is the token usage of the request.
	Usage DecisionUsage `json:"usage"`
}

// ToResult converts the response to a genai.Result.
//
// The reply is the JSON object of the answers keyed by question name, which Result.Decode decodes into
// the questionnaire struct or into Answers.
func (r *SystemOneResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:  r.Usage.InputTokens,
			OutputTokens: r.Usage.OutputTokens,
			TotalTokens:  r.Usage.InputTokens + r.Usage.OutputTokens,
			FinishReason: genai.FinishedStop,
		},
	}
	if len(r.Answers) == 0 {
		return out, errors.New("no answer returned")
	}
	raw, err := marshalAnswers(r.Answers)
	if err != nil {
		return out, err
	}
	out.Replies = []genai.Reply{{Text: string(raw)}}
	return out, nil
}

// DecisionUsage reports the token usage of a request.
type DecisionUsage struct {
	// InputTokens counts the prompt tokens evaluated for all questions.
	InputTokens int64 `json:"input_tokens"`
	// OutputTokens is always zero: decision models evaluate probabilities without generating text.
	OutputTokens int64 `json:"output_tokens"`
}

// marshalAnswers marshals the answers for genai.Result.
//
// It does not escape HTML so that descriptions are returned verbatim, like the API does.
func marshalAnswers(a Answers) ([]byte, error) {
	buf := &bytes.Buffer{}
	enc := json.NewEncoder(buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(a); err != nil {
		return nil, err
	}
	// Encoder.Encode appends a newline.
	return bytes.TrimSuffix(buf.Bytes(), []byte("\n")), nil
}

// unsupportedTextOptions lists the genai.GenOptionText fields that are set but can't be honored, the
// questions being the only thing System One takes from it.
func unsupportedTextOptions(o *genai.GenOptionText) []string {
	var out []string
	if o.Temperature != 0 {
		out = append(out, "GenOptionText.Temperature")
	}
	if o.TopP != 0 {
		out = append(out, "GenOptionText.TopP")
	}
	if o.MaxTokens != 0 {
		out = append(out, "GenOptionText.MaxTokens")
	}
	if o.TopLogprobs != 0 {
		out = append(out, "GenOptionText.TopLogprobs")
	}
	if o.TopK != 0 {
		out = append(out, "GenOptionText.TopK")
	}
	if o.SystemPrompt != "" {
		out = append(out, "GenOptionText.SystemPrompt")
	}
	if len(o.Stop) != 0 {
		out = append(out, "GenOptionText.Stop")
	}
	if o.ReplyAsJSON {
		out = append(out, "GenOptionText.ReplyAsJSON")
	}
	return out
}

func validateContent(c DecisionContent) error {
	if c == nil {
		return errors.New("must not be nil")
	}
	return c.Validate()
}

// Noul is a yes/no question about a state, and holds its answer once asked.
//
// Declare the questions to ask as fields of a struct passed to genai.GenOptionText.DecodeAs; GenSync asks
// them and Decode fills the answers in. The field name, or its `json` tag, is the question name.
type Noul struct {
	// Instructions is the yes/no question to ask. Instructions is required.
	Instructions DecisionContent
	// Criteria optionally describes what the yes and the no outcomes mean.
	Criteria *NoulCriteria

	// Probability is the answer, the probability that the answer is yes, from 0 to 1. It is set by
	// Decode.
	Probability float64
}

// UnmarshalJSON implements json.Unmarshaler.
func (n *Noul) UnmarshalJSON(b []byte) error {
	a := Answer{}
	if err := internal.UnmarshalJSON(b, &a); err != nil {
		return err
	}
	if a.Type != QuestionNoul {
		return fmt.Errorf("expected a noul answer, got %q", a.Type)
	}
	n.Probability = a.Noul
	return nil
}

// Choice is a question that picks one option among a set, and holds its answer once asked.
//
// Declare the questions to ask as fields of a struct passed to genai.GenOptionText.DecodeAs; GenSync asks
// them and Decode fills the answers in. The field name, or its `json` tag, is the question name.
type Choice struct {
	// Instructions describes what the model should decide.
	Instructions DecisionContent
	// Criteria maps the options to their descriptions. Use nil for an option that needs no extra detail.
	// At least one option is required.
	Criteria map[string]DecisionContent

	// Label is the answer, the highest probability option. The other fields are derived from the
	// probabilities the model reported. They are set by Decode.
	Label         string
	Confidence    float64
	Probabilities map[string]float64
}

// UnmarshalJSON implements json.Unmarshaler.
func (c *Choice) UnmarshalJSON(b []byte) error {
	a := Answer{}
	if err := internal.UnmarshalJSON(b, &a); err != nil {
		return err
	}
	if a.Type != QuestionChoice {
		return fmt.Errorf("expected a choice answer, got %q", a.Type)
	}
	c.Label = a.Choice
	c.Confidence = a.Confidence
	c.Probabilities = a.Probabilities
	return nil
}

// Score is a question that rates the state along an ordered rubric, and holds its answer once asked.
//
// Declare the questions to ask as fields of a struct passed to genai.GenOptionText.DecodeAs; GenSync asks
// them and Decode fills the answers in. The field name, or its `json` tag, is the question name.
type Score struct {
	// Instructions describes what the model should rate.
	Instructions DecisionContent
	// Criteria lists the levels in order, starting at level 0. Between 2 and 10 levels are required.
	Criteria []DecisionContent

	// Value is the answer, the probability weighted value across the levels, which can land between
	// levels. The other fields are derived from the probabilities the model reported, and legend is the
	// rubric echoed back. They are set by Decode.
	Value         float64
	Confidence    float64
	Legend        ScoreLegend
	Probabilities map[string]float64
}

// UnmarshalJSON implements json.Unmarshaler.
func (s *Score) UnmarshalJSON(b []byte) error {
	a := Answer{}
	if err := internal.UnmarshalJSON(b, &a); err != nil {
		return err
	}
	if a.Type != QuestionScore {
		return fmt.Errorf("expected a score answer, got %q", a.Type)
	}
	s.Value = a.Score
	s.Confidence = a.Confidence
	s.Legend = a.Legend
	s.Probabilities = a.Probabilities
	return nil
}

// questionName returns the name to use for a question field.
func questionName(f *reflect.StructField) (string, bool, error) {
	name := f.Name
	if tag, ok := f.Tag.Lookup("json"); ok {
		n, _, _ := strings.Cut(tag, ",")
		if n == "-" {
			return "", true, nil
		}
		if n != "" {
			name = n
		}
	}
	if !f.IsExported() {
		return "", false, fmt.Errorf("field %s: must be exported to receive its answer", f.Name)
	}
	return name, false, nil
}

var (
	_ json.Unmarshaler     = (*Noul)(nil)
	_ json.Unmarshaler     = (*Choice)(nil)
	_ json.Unmarshaler     = (*Score)(nil)
	_ internal.Validatable = (*NoulCriteria)(nil)
)
