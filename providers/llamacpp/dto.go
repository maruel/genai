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
	"path/filepath"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// EmbeddingInput is one text, token sequence, or multimodal input to embed.
// Set exactly one field. Content uses the same parts as chat completions and
// requires a compatible model and projector loaded by llama-server.
type EmbeddingInput struct {
	Text    string
	Tokens  []int
	Content Contents
}

// Validate implements genai.Validatable.
func (e *EmbeddingInput) Validate() error {
	n := 0
	if e.Text != "" {
		n++
	}
	if len(e.Tokens) != 0 {
		n++
		for _, t := range e.Tokens {
			if t < 0 {
				return errors.New("embedding token IDs must be non-negative")
			}
		}
	}
	if len(e.Content) != 0 {
		n++
	}
	if n != 1 {
		return errors.New("embedding input must set exactly one of Text, Tokens or Content")
	}
	return nil
}

// MarshalJSON encodes the native input union as a string, token array or content object.
func (e EmbeddingInput) MarshalJSON() ([]byte, error) {
	if err := e.Validate(); err != nil {
		return nil, err
	}
	if e.Text != "" {
		return json.Marshal(e.Text)
	}
	if len(e.Tokens) != 0 {
		return json.Marshal(e.Tokens)
	}
	return json.Marshal(embeddingInputContent{Content: e.Content})
}

// EmbeddingRequest supplies pooled inputs to /v1/embeddings.
// Task prefixes are passed verbatim; llama-server does not apply chat templates.
type EmbeddingRequest struct {
	Input []EmbeddingInput `json:"input"`
	Model string           `json:"model,omitzero"`
	// Normalize overrides the server's normalization: -1 disables it, 0 uses
	// the maximum absolute value scaled to int16 range, 1 uses L1, 2 uses L2,
	// and larger values use p-norm.
	// Nil uses the server's --embd-normalize setting (default 2, L2).

	Normalize *int `json:"embd_normalize,omitzero"`
}

// Validate implements genai.Validatable.
func (e *EmbeddingRequest) Validate() error {
	if len(e.Input) == 0 {
		return errors.New("embedding input must not be empty")
	}
	if e.Normalize != nil && *e.Normalize < -1 {
		return errors.New("embedding normalization must be at least -1")
	}
	for i := range e.Input {
		if err := e.Input[i].Validate(); err != nil {
			return fmt.Errorf("embedding input #%d: %w", i, err)
		}
	}
	return nil
}

// embeddingRequest fixes the response encoding without modifying caller inputs.
type embeddingRequest struct {
	Model          string           `json:"model,omitzero"`
	Input          []EmbeddingInput `json:"input"`
	EncodingFormat string           `json:"encoding_format"`
	Normalize      *int             `json:"embd_normalize,omitzero"`
}

// EmbeddingResponse is the pooled /v1/embeddings response.
type EmbeddingResponse struct {
	Model  string         `json:"model"`
	Object string         `json:"object"`
	Usage  EmbeddingUsage `json:"usage"`
	Data   []Embedding    `json:"data"`
}

// EmbeddingUsage reports input token consumption.
type EmbeddingUsage struct {
	PromptTokens int64 `json:"prompt_tokens"`
	TotalTokens  int64 `json:"total_tokens"`
}

// Embedding is one indexed pooled vector.
type Embedding struct {
	Object         string               `json:"object"`
	Index          int                  `json:"index"`
	EncodingFormat string               `json:"encoding_format,omitzero"`
	Embedding      base.EmbeddingVector `json:"embedding"`
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

// SystemOneResponse is the native /v1/systemone response.
type SystemOneResponse struct {
	// Model identifies the loaded decision model that answered.
	Model string `json:"model"`
	genai.SystemOneResponse
}

// To converts native decision output to shared decision output.
func (r *SystemOneResponse) To(out *genai.SystemOneResponse) error {
	*out = r.SystemOneResponse
	return nil
}

// SystemOneRequest is a native /v1/systemone request.
// llama.cpp requires instructions and 2 to 10 score levels.
type SystemOneRequest struct {
	State     genai.DecisionContent `json:"state"`
	Model     string                `json:"model,omitzero"`
	Questions genai.Questions       `json:"questions"`
	// Images contains inline data URLs, as required by the native endpoint.
	Images []string `json:"images,omitzero"`
}

// From converts shared decision input, preserving Model.
// One JSON document can supply the state. Images are encoded as data URLs.
// Each document read is bounded to 10 MiB.
func (r *SystemOneRequest) From(in *genai.SystemOneRequest) error {
	state, docs, err := in.ReadState(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	var images []string
	if len(docs) != 0 {
		images = make([]string, len(docs))
	}
	for j := range docs {
		d := docs[j]
		if err := d.Validate(); err != nil {
			return fmt.Errorf("document #%d: %w", j, err)
		}
		if d.URL != "" || d.Src == nil {
			return fmt.Errorf("document #%d: must be an inline document", j)
		}
		mt := internal.MimeByExt(filepath.Ext(d.GetFilename()))
		if !strings.HasPrefix(mt, "image/") {
			return fmt.Errorf("document #%d: unsupported document type %q", j, mt)
		}
	}
	for j := range docs {
		d := docs[j]
		mt, data, err := d.Read(10 * 1024 * 1024)
		if err != nil {
			return fmt.Errorf("image #%d: %w", j, err)
		}
		images[j] = "data:" + mt + ";base64," + base64.StdEncoding.EncodeToString(data)
	}
	r.State, r.Questions, r.Images = state, in.Questions, images
	return nil
}

// Validate checks the native endpoint's state, question and image requirements.
func (r *SystemOneRequest) Validate() error {
	var errs []error
	in := genai.SystemOneRequest{State: r.State, Questions: r.Questions}
	if err := in.Validate(); err != nil {
		errs = append(errs, err)
	}
	for j, img := range r.Images {
		if !strings.HasPrefix(img, "data:image/") || !strings.Contains(img, ";base64,") {
			errs = append(errs, fmt.Errorf("image #%d: must be an inline image data URL", j))
		}
	}
	for _, id := range slices.Sorted(maps.Keys(r.Questions)) {
		q := r.Questions[id]
		if q == nil {
			continue
		}
		if q.Instructions == nil {
			errs = append(errs, fmt.Errorf("question %q: field Instructions: is required", id))
		}
		if q.Type == genai.QuestionScore && (len(q.Score) < 2 || len(q.Score) > 10) {
			errs = append(errs, fmt.Errorf("question %q: field Score: 2 to 10 levels are required", id))
		}
	}
	return errors.Join(errs...)
}

type embeddingInputContent struct {
	Content Contents `json:"content"`
}
