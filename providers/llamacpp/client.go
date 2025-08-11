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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
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
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// Scoreboard for llama.cpp.
//
// # Warnings
//
//   - The multi-modal file is referred to with "#" character. I was initially using ";" but it caused
//     "go get" to fail and frankly it was annoying to use when copy-pasting paths in bash.
var Scoreboard = genai.Scoreboard{
	Country:      "Local",
	DashboardURL: "https://github.com/ggml-org/llama.cpp",
	Scenarios: []genai.Scenario{
		// https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/tree/main
		{
			// Models: []string{"ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q8_0.gguf#mmproj-model-f16.gguf"},
			Models: []string{"ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf#mmproj-model-f16.gguf"},
			// TODO: It supports genai.ModalityImage
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png"},
				},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:       genai.True,
				BiasedTool:  genai.True,
				Seed:        true,
				TopLogprobs: true,
				JSON:        true,
				JSONSchema:  true,
			},
			GenStream: &genai.FunctionalityText{
				Tools:       genai.True,
				BiasedTool:  genai.True,
				Seed:        true,
				TopLogprobs: true,
				JSON:        true,
				JSONSchema:  true,
			},
		},
	},
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
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	var errs []error
	var unsupported []string
	sp := ""
	c.CachePrompt = true
	if opts != nil {
		switch v := opts.(type) {
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
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}
	offset := 0
	if sp != "" {
		offset = 1
	}
	c.Messages = make([]Message, len(msgs)+offset)
	if sp != "" {
		c.Messages[0].Role = "system"
		c.Messages[0].Content = Contents{{Type: "text", Text: sp}}
	}
	for i := range msgs {
		if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
		}
	}
	// If we have unsupported features but no other errors, return a continuable error
	if len(unsupported) > 0 && len(errs) == 0 {
		return &genai.UnsupportedContinuableError{Unsupported: unsupported}
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
		out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
		if err := c.Choices[0].Message.To(&out.Message); err != nil {
			return out, err
		}
		if len(c.Choices[0].Logprobs.Content) != 0 {
			out.Logprobs = &genai.Logprobs{
				Content: make([]genai.LogprobsContent, len(c.Choices[0].Logprobs.Content)),
			}
			for i, lp := range c.Choices[0].Logprobs.Content {
				out.Logprobs.Content[i].Token = lp.Token
				out.Logprobs.Content[i].Logprob = lp.Logprob
				out.Logprobs.Content[i].Bytes = lp.Bytes
				out.Logprobs.Content[i].TopLogprobs = make([]genai.TopLogprob, len(lp.TopLogprobs))
				for j, tlp := range lp.TopLogprobs {
					out.Logprobs.Content[i].TopLogprobs[j].Token = tlp.Token
					out.Logprobs.Content[i].TopLogprobs[j].Logprob = tlp.Logprob
					out.Logprobs.Content[i].TopLogprobs[j].Bytes = tlp.Bytes
				}
			}
		}
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
func (c *CompletionRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	var errs []error
	var unsupported []string
	c.CachePrompt = true
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsText:
			c.NPredict = v.MaxTokens
			c.Seed = v.Seed
			if v.TopLogprobs > 0 {
				// TODO: This should be supported.
				unsupported = append(unsupported, "TopLogprobs")
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
			if len(v.Tools) != 0 {
				errs = append(errs, errors.New("implement option Tools"))
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}
	// If we have unsupported features but no other errors, return a continuable error
	if len(unsupported) > 0 && len(errs) == 0 {
		return &genai.UnsupportedContinuableError{Unsupported: unsupported}
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
		Message: genai.Message{
			Role: genai.Assistant,
			// Mistral Nemo really likes "▁".
			Contents: []genai.Content{{Text: strings.ReplaceAll(c.Content, "\u2581", " ")}},
		},
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

func (a *applyTemplateRequest) Init(opts genai.Options, msgs genai.Messages) error {
	sp := ""
	if v, ok := opts.(*genai.OptionsText); ok {
		sp = v.SystemPrompt
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
		a.Messages[0].Content = Contents{{Type: "text", Text: sp}}
	}
	for i := range msgs {
		if err := a.Messages[i+offset].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
		}
	}
	// If we have unsupported features but no other errors, return a continuable error
	if len(unsupported) > 0 && len(errs) == 0 {
		return &genai.UnsupportedContinuableError{Unsupported: unsupported}
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
	ReasoningContent struct{}   `json:"reasoning_content,omitzero"`
	Name             string     `json:"name,omitzero"`
	ToolCallID       string     `json:"tool_call_id,omitzero"`
}

func (m *Message) From(in *genai.Message) error {
	// We intentionally do not filter the role here.
	m.Role = string(in.Role)
	if len(in.Contents) != 0 {
		for i := range in.Contents {
			c := Content{}
			if skip, err := c.From(&in.Contents[i]); err != nil {
				return err
			} else if !skip {
				m.Content = append(m.Content, c)
			}
		}
	}
	if len(in.ToolCalls) != 0 {
		m.ToolCalls = make([]ToolCall, len(in.ToolCalls))
		for i := range m.ToolCalls {
			if err := m.ToolCalls[i].From(&in.ToolCalls[i]); err != nil {
				return err
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		if len(in.Contents) != 0 || len(in.ToolCalls) != 0 {
			// This could be worked around.
			return fmt.Errorf("can't have tool call result along content or tool calls")
		}
		if len(in.ToolCallResults) != 1 {
			// This could be worked around.
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
		m.Role = "tool"
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Content = []Content{{Type: "text", Text: in.ToolCallResults[0].Result}}
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	out.Role = genai.Assistant
	for i := range m.Content {
		out.Contents = make([]genai.Content, len(m.Content))
		if err := m.Content[i].To(&out.Contents[i]); err != nil {
			return err
		}
	}
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			m.ToolCalls[i].To(&out.ToolCalls[i])
		}
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

func (c *Content) From(in *genai.Content) (bool, error) {
	if len(in.Citations) != 0 {
		return false, errors.New("citations are not supported")
	}
	if len(in.Opaque) != 0 {
		return false, errors.New("opaque data is not supported")
	}
	if in.Thinking != "" {
		return true, nil
	}
	if in.Text != "" {
		c.Type = "text"
		c.Text = in.Text
		return false, nil
	}
	// Check if this is a text/plain document
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return false, fmt.Errorf("failed to read document: %w", err)
	}
	switch {
	case strings.HasPrefix(mimeType, "text/plain"):
		if in.URL != "" {
			return false, errors.New("text/plain documents must be provided inline, not as a URL")
		}
		c.Type = "text"
		c.Text = string(data)
	case strings.HasPrefix(mimeType, "audio/"):
		c.Type = "input_audio"
		if in.URL != "" {
			return false, errors.New("audio doesn't support URLs")
		}
		c.InputAudio.Data = data
		c.InputAudio.Format, _ = strings.CutPrefix(mimeType, "audio/")
		return false, nil
	case strings.HasPrefix(mimeType, "image/"):
		c.Type = "image_url"
		if in.URL != "" {
			c.ImageURL.URL = in.URL
		} else {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		}
		return false, nil
	default:
		return false, fmt.Errorf("mime type %s is unsupported", mimeType)
	}
	return false, fmt.Errorf("unsupported content type %v", in)
}

func (c *Content) To(out *genai.Content) error {
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
		return errors.New("unsupported opaque in tool call")
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

// Client implements genai.ProviderGen.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]

	baseURL        string
	completionsURL string
	modelsURL      string
	encoding       *PromptEncoding
}

// New creates a new client to talk to a llama-server instance.
//
// Options Remote defaults to "http://localhost:8080".
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	var encoding *PromptEncoding
	if opts.APIKey != "" {
		return nil, errors.New("option APIKey is not yet implemented")
	}
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	model := opts.Model
	switch model {
	case "", base.NoModel, base.PreferredCheap, base.PreferredGood, base.PreferredSOTA:
		model = ""
	default:
	}
	baseURL := opts.Remote
	if baseURL == "" {
		baseURL = "http://localhost:8080"
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	return &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:           baseURL + "/chat/completions",
			ProcessStreamPackets: processChatStreamPackets,
			ModelOptional:        true,
			Model:                model,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "llamacpp",
				ClientJSON: httpjson.Client{
					Lenient: internal.BeLenient,
					Client: &http.Client{
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
		baseURL:        baseURL,
		completionsURL: baseURL + "/completions",
		modelsURL:      baseURL + "/v1/models",
		encoding:       encoding,
	}, nil
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ModelID() string {
	if c.Model != "" {
		return c.Model
	}
	m, _ := c.ListModels(context.Background())
	if len(m) > 0 {
		return m[0].GetID()
	}
	return ""
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, c.modelsURL)
}

func (c *Client) Completions(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
	// Doc mentions Cache:true causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	if err := msgs.Validate(); err != nil {
		return genai.Result{}, err
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return genai.Result{}, err
		}
		if supported := opts.Modalities(); !slices.Contains(supported, genai.ModalityText) {
			return genai.Result{}, fmt.Errorf("modality text not supported, supported: %s", supported)
		}
	}
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return genai.Result{}, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
			}
		}
		for j, tool := range msg.ToolCalls {
			if len(tool.Opaque) != 0 {
				return genai.Result{}, fmt.Errorf("message #%d tool call #%d: field Opaque not supported", i, j)
			}
		}
	}
	rpcin := CompletionRequest{CachePrompt: true}
	if err := rpcin.Init(msgs, opts, ""); err != nil {
		return genai.Result{}, err
	}
	if err := c.initPrompt(ctx, &rpcin, opts, msgs); err != nil {
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
	return c.DoRequest(ctx, "POST", c.completionsURL, in, out)
}

func (c *Client) CompletionsStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	result := genai.Result{}
	if err := msgs.Validate(); err != nil {
		return result, err
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return result, err
		}
		if supported := opts.Modalities(); !slices.Contains(supported, genai.ModalityText) {
			return genai.Result{}, fmt.Errorf("modality text not supported, supported: %s", supported)
		}
	}

	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return result, fmt.Errorf("message #%d content #%d: Opaque field not supported", i, j)
			}
		}
		for j, tool := range msg.ToolCalls {
			if len(tool.Opaque) != 0 {
				return result, fmt.Errorf("message #%d tool call #%d: field Opaque not supported", i, j)
			}
		}
	}

	in := CompletionRequest{}
	var continuableErr error
	if err := in.Init(msgs, opts, ""); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return result, err
		}
	}
	if err := c.initPrompt(ctx, &in, opts, msgs); err != nil {
		return result, err
	}
	ch := make(chan CompletionStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processCompletionsStreamPackets(ch, chunks, &result)
	})
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return result, continuableErr
	}
	return result, err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	in.Stream = true
	resp, err := c.ClientJSON.Request(ctx, "POST", c.completionsURL, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get llama server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.DecodeError(c.completionsURL, resp)
	}
	return sse.Process(resp.Body, out, nil, c.ClientJSON.Lenient)
}

func (c *Client) GetHealth(ctx context.Context) (string, error) {
	msg, err := c.GetHealthRaw(ctx)
	return msg.Status, err
}

func (c *Client) GetHealthRaw(ctx context.Context) (HealthResponse, error) {
	msg := HealthResponse{}
	if err := c.DoRequest(ctx, "GET", c.baseURL+"/health", nil, &msg); err != nil {
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
	resp, err := c.ClientJSON.Client.Do(req)
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
		default:
			if !internal.BeLenient {
				panic(fmt.Sprintf("unknown metric %q", l))
			}
			return fmt.Errorf("unknown metric %q", l)
		}
	}
	return nil
}

func (c *Client) initPrompt(ctx context.Context, in *CompletionRequest, opts genai.Options, msgs genai.Messages) error {
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
		if err := c.DoRequest(ctx, "POST", c.baseURL+"/apply-template", &in2, &out); err != nil {
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
		case genai.Computer:
			fallthrough
		default:
			return fmt.Errorf("unexpected role %q", m.Role)
		}
	}
	return nil
}

func processChatStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Choices[0].FinishReason != "" {
			result.InputTokens = pkt.Usage.PromptTokens
			// result.InputCachedTokens = pkt.Usage.TokensCached
			result.OutputTokens = pkt.Usage.CompletionTokens
			result.TotalTokens = pkt.Usage.TotalTokens
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		f := genai.ContentFragment{
			TextFragment: pkt.Choices[0].Delta.Content,
			// ThinkingFragment: pkt.Choices[0].Delta.ReasoningContent,
		}
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt)
		}
		if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
			if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID != "" {
				// A new call.
				if pendingCall.ID == "" {
					pendingCall = t
					if !f.IsZero() {
						return fmt.Errorf("implement tool call with metadata: %#v", pkt)
					}
					continue
				}
				// Flush.
				pendingCall.To(&f.ToolCall)
				pendingCall = t
			} else if pendingCall.ID != "" {
				// Continuation.
				pendingCall.Function.Arguments += t.Function.Arguments
				if !f.IsZero() {
					return fmt.Errorf("implement tool call with metadata: %#v", pkt)
				}
				continue
			}
		} else if pendingCall.ID != "" {
			// Flush.
			pendingCall.To(&f.ToolCall)
			pendingCall = ToolCall{}
		}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		if len(pkt.Choices[0].Logprobs.Content) != 0 {
			if result.Logprobs == nil {
				result.Logprobs = &genai.Logprobs{}
			}
			for _, lp := range pkt.Choices[0].Logprobs.Content {
				genaiLp := genai.LogprobsContent{
					Token:   lp.Token,
					Logprob: lp.Logprob,
					Bytes:   lp.Bytes,
				}
				genaiLp.TopLogprobs = make([]genai.TopLogprob, len(lp.TopLogprobs))
				for j, tlp := range lp.TopLogprobs {
					genaiLp.TopLogprobs[j].Token = tlp.Token
					genaiLp.TopLogprobs[j].Logprob = tlp.Logprob
					genaiLp.TopLogprobs[j].Bytes = tlp.Bytes
				}
				result.Logprobs.Content = append(result.Logprobs.Content, genaiLp)
			}
		}
	}
	return nil
}

func processCompletionsStreamPackets(ch <-chan CompletionStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	for msg := range ch {
		if msg.Timings.PredictedN != 0 {
			result.InputTokens = msg.Timings.PromptN
			result.OutputTokens = msg.Timings.PredictedN
		}
		if msg.StopType != "" {
			result.FinishReason = msg.StopType.ToFinishReason()
		}
		f := genai.ContentFragment{TextFragment: msg.Content}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	return nil
}

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
