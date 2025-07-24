// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package togetherai implements a client for the Together.ai API.
//
// It is described at https://docs.together.ai/docs/
package togetherai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Official python client library at https://github.com/togethercomputer/together-python/tree/main/src/together

// Scoreboard for TogetherAI.
//
// # Warnings
//
//   - No model supports "required" tool calling, thus it's flaky everywhere.
//   - Tool calling is solid with llama 3.3 70B quantized in FP8 (-Turbo) but is flaky in more recent models.
//
// # Models
//
//   - Suffix "-Turbo" means FP8 quantization.
//   - Suffix "-Lite" means INT4 quantization.
//   - Suffix "-Free" has lower rate limits.
//
// See https://docs.together.ai/docs/serverless-models and https://api.together.ai/models
var Scoreboard = genai.Scoreboard{
	Country:      "US",
	DashboardURL: "https://api.together.ai/settings/billing",
	Scenarios: []genai.Scenario{
		{
			// Tool calling is flaky on llama-4 because it only support tool_choice auto, not required.
			Models: []string{"meta-llama/Llama-4-Scout-17B-16E-Instruct"},
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:          genai.Flaky,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
				Seed:           true,
			},
			GenStream: &genai.FunctionalityText{
				Tools:          genai.Flaky,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
		},
		{
			// Tool calling is flaky on llama-4 because it only support tool_choice auto, not required.
			Models: []string{"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:              genai.Flaky,
				BiasedTool:         genai.True,
				JSON:               true,
				JSONSchema:         true,
				Seed:               true,
				BrokenFinishReason: true,
				NoMaxTokens:        true,
			},
			GenStream: &genai.FunctionalityText{
				Tools:              genai.Flaky,
				BiasedTool:         genai.True,
				JSON:               true,
				JSONSchema:         true,
				Seed:               true,
				BrokenFinishReason: true,
				NoMaxTokens:        true,
			},
		},
		{
			Models: []string{"meta-llama/Llama-Vision-Free"},
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:              genai.Flaky,
				Seed:               true,
				BrokenFinishReason: true,
				NoMaxTokens:        true,
			},
			GenStream: &genai.FunctionalityText{
				Tools: genai.Flaky,
			},
		},
		{
			Models: []string{"moonshotai/Kimi-K2-Instruct"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
			GenStream: &genai.FunctionalityText{JSON: true, JSONSchema: true, Seed: true},
		},
		{
			Models: []string{"mistralai/Mistral-Small-24B-Instruct-2501"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:              genai.Flaky,
				BiasedTool:         genai.True,
				IndecisiveTool:     genai.True,
				JSON:               true,
				Seed:               true,
				BrokenFinishReason: true,
			},
			GenStream: &genai.FunctionalityText{
				Tools:              genai.Flaky,
				JSON:               true,
				Seed:               true,
				BrokenFinishReason: true,
			},
		},
		{
			Models: []string{"black-forest-labs/FLUX.1-schnell"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out: map[genai.Modality]genai.ModalCapability{
				genai.ModalityImage: {
					URL:              true,
					SupportedFormats: []string{"image/jpeg"},
				},
			},
			GenDoc: &genai.FunctionalityDoc{
				BrokenTokenUsage:   genai.True,
				BrokenFinishReason: true,
			},
		},
		// Skipped
		{
			Models: []string{
				"Alibaba-NLP/gte-modernbert-base",
				"Gryphe/MythoMax-L2-13b",
				"Gryphe/MythoMax-L2-13b-Lite",
				"NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
				"Qwen/QwQ-32B",
				"Qwen/Qwen2-72B-Instruct",
				"Qwen/Qwen2-VL-72B-Instruct",
				"Qwen/Qwen2.5-72B-Instruct-Turbo",
				"Qwen/Qwen2.5-7B-Instruct-Turbo",
				"Qwen/Qwen2.5-Coder-32B-Instruct",
				"Qwen/Qwen2.5-VL-72B-Instruct",
				"Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
				"Qwen/Qwen3-235B-A22B-fp8-tput",
				"Qwen/Qwen3-32B-FP8",
				"Salesforce/Llama-Rank-V1",
				"arcee-ai/AFM-4.5B-Preview",
				"arcee-ai/arcee-blitz",
				"arcee-ai/caller",
				"arcee-ai/coder-large",
				"arcee-ai/maestro-reasoning",
				"arcee-ai/virtuoso-large",
				"arcee-ai/virtuoso-medium-v2",
				"arcee_ai/arcee-spotlight",
				"black-forest-labs/FLUX.1-canny",
				"black-forest-labs/FLUX.1-depth",
				"black-forest-labs/FLUX.1-dev",
				"black-forest-labs/FLUX.1-dev-lora",
				"black-forest-labs/FLUX.1-kontext-dev",
				"black-forest-labs/FLUX.1-kontext-max",
				"black-forest-labs/FLUX.1-kontext-pro",
				"black-forest-labs/FLUX.1-pro",
				"black-forest-labs/FLUX.1-redux",
				"black-forest-labs/FLUX.1-schnell-Free",
				"black-forest-labs/FLUX.1-schnell-fixedres",
				"black-forest-labs/FLUX.1.1-pro",
				"cartesia/sonic",
				"cartesia/sonic-2",
				"deepseek-ai/DeepSeek-R1",
				"deepseek-ai/DeepSeek-R1-0528-tput",
				"deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
				"deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
				"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
				"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
				"deepseek-ai/DeepSeek-V3",
				"eddiehou/meta-llama/Llama-3.1-405B",
				"google/gemma-2-27b-it",
				"google/gemma-3-27b-it",
				"google/gemma-3n-E4B-it",
				"intfloat/multilingual-e5-large-instruct",
				"lgai/exaone-3-5-32b-instruct",
				"lgai/exaone-deep-32b",
				"marin-community/marin-8b-instruct",
				"meta-llama/Llama-2-70b-hf",
				"meta-llama/Llama-3-70b-chat-hf",
				"meta-llama/Llama-3-8b-chat-hf",
				"meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo",
				"meta-llama/Llama-3.2-3B-Instruct-Turbo",
				"meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
				"meta-llama/Llama-3.3-70B-Instruct-Turbo",
				"meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
				"meta-llama/Llama-Guard-3-11B-Vision-Turbo",
				"meta-llama/Llama-Guard-4-12B",
				"meta-llama/LlamaGuard-2-8b",
				"meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
				"meta-llama/Meta-Llama-3-8B-Instruct-Lite",
				"meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
				"meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
				"meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
				"meta-llama/Meta-Llama-Guard-3-8B",
				"mistralai/Mistral-7B-Instruct-v0.1",
				"mistralai/Mistral-7B-Instruct-v0.2",
				"mistralai/Mistral-7B-Instruct-v0.3",
				"mistralai/Mixtral-8x7B-Instruct-v0.1",
				"mixedbread-ai/Mxbai-Rerank-Large-V2",
				"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
				"openai/whisper-large-v3",
				"perplexity-ai/r1-1776",
				"scb10x/scb10x-llama3-1-typhoon2-70b-instruct",
				"scb10x/scb10x-typhoon-2-1-gemma3-12b",
				"togethercomputer/MoA-1",
				"togethercomputer/MoA-1-Turbo",
				"togethercomputer/Refuel-Llm-V2",
				"togethercomputer/Refuel-Llm-V2-Small",
				"togethercomputer/m2-bert-80M-32k-retrieval",
				"yan/deepseek-ai-deepseek-v3",
			},
		},
	},
}

// ChatRequest is documented at https://docs.together.ai/reference/chat-completions-1
//
// https://docs.together.ai/docs/chat-overview
type ChatRequest struct {
	Model                         string             `json:"model"`
	Stream                        bool               `json:"stream"`
	Messages                      []Message          `json:"messages"`
	MaxTokens                     int64              `json:"max_tokens,omitzero"`
	Stop                          []string           `json:"stop,omitzero"`
	Temperature                   float64            `json:"temperature,omitzero"` // [0, 1]
	TopP                          float64            `json:"top_p,omitzero"`       // [0, 1]
	TopK                          int64              `json:"top_k,omitzero"`
	ContextLengthExceededBehavior string             `json:"context_length_exceeded_behavior,omitzero"` // "error", "truncate"
	RepetitionPenalty             float64            `json:"repetition_penalty,omitzero"`
	Logprobs                      int32              `json:"logprobs,omitzero"` // bool as 0 or 1
	Echo                          bool               `json:"echo,omitzero"`
	N                             int32              `json:"n,omitzero"`                 // Number of completions to generate
	PresencePenalty               float64            `json:"presence_penalty,omitzero"`  // [-2.0, 2.0]
	FrequencyPenalty              float64            `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	LogitBias                     map[string]float64 `json:"logit_bias,omitzero"`
	Seed                          int64              `json:"seed,omitzero"`
	ResponseFormat                struct {
		Type   string             `json:"type,omitzero"` // "json_object", "json_schema" according to python library.
		Schema *jsonschema.Schema `json:"schema,omitzero"`
	} `json:"response_format,omitzero"`
	Tools       []Tool `json:"tools,omitzero"`
	ToolChoice  string `json:"tool_choice,omitzero"`  // "auto" or a []Tool
	SafetyModel string `json:"safety_model,omitzero"` // https://docs.together.ai/docs/inference-models#moderation-models
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			c.Seed = v.Seed
			c.TopK = v.TopK
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				// Warning: using a model small may fail.
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.Schema = jsonschema.Reflect(v.DecodeAs)
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if len(v.Tools) != 0 {
				switch v.ToolCallRequest {
				case genai.ToolCallAny:
					c.ToolChoice = "auto"
				case genai.ToolCallRequired:
					// Interestingly, https://docs.together.ai/reference/chat-completions-1 doesn't document anything
					// beside "auto" but https://docs.livekit.io/agents/integrations/llm/together/ says that
					// "required" works. I'll have to confirm.
					c.ToolChoice = "required"
				case genai.ToolCallNone:
					c.ToolChoice = "none"
				}
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					if c.Tools[i].Function.Parameters = t.InputSchemaOverride; c.Tools[i].Function.Parameters == nil {
						c.Tools[i].Function.Parameters = t.GetInputSchema()
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
		c.Messages[0].Content = []Content{{Type: "text", Text: sp}}
	}
	for i := range msgs {
		if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
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

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Message is documented at https://docs.together.ai/reference/chat-completions-1
type Message struct {
	Role    string   `json:"role,omitzero"` // "system", "assistant", "user"
	Content Contents `json:"content,omitzero"`
	// Warning: using a small model may fail.
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) != 0 {
		m.Content = make([]Content, 0, len(in.Contents))
		for i := range in.Contents {
			if in.Contents[i].Thinking != "" {
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].From(&in.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		}
	}
	if len(in.ToolCalls) != 0 {
		m.ToolCalls = make([]ToolCall, len(in.ToolCalls))
		for i := range in.ToolCalls {
			m.ToolCalls[i].From(&in.ToolCalls[i])
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
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant", "user":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			m.ToolCalls[i].To(&out.ToolCalls[i])
		}
	}
	if len(m.Content) != 0 {
		out.Contents = make([]genai.Content, len(m.Content))
		for i := range m.Content {
			if err := m.Content[i].To(&out.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		}
	}
	return nil
}

type Contents []Content

// UnmarshalJSON implements json.Unmarshaler.
//
// Together.AI replies with content as a string.
func (c *Contents) UnmarshalJSON(data []byte) error {
	var v []Content
	if err := json.Unmarshal(data, &v); err != nil {
		s := ""
		if err = json.Unmarshal(data, &s); err != nil {
			return err
		}
		*c = []Content{{Type: "text", Text: s}}
		return nil
	}
	*c = Contents(v)
	return nil
}

// MarshalJSON implements json.Marshaler.
//
// Together.AI really prefer simple strings.
func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal(([]Content)(*c))
}

type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == ContentText
	Text string `json:"text,omitzero"`

	// Type == ContentImageURL
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`

	// Type == ContentVideoURL
	VideoURL struct {
		URL string `json:"url,omitzero"`
	} `json:"video_url,omitzero"`
}

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		c.Type = ContentImageURL
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	case strings.HasPrefix(mimeType, "video/"):
		c.Type = ContentVideoURL
		if in.URL == "" {
			c.VideoURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.VideoURL.URL = in.URL
		}
	case strings.HasPrefix(mimeType, "text/plain"):
		c.Type = ContentText
		if in.URL != "" {
			return errors.New("text/plain documents must be provided inline, not as a URL")
		}
		c.Text = string(data)
	default:
		return fmt.Errorf("unsupported mime type %s", mimeType)
	}
	return nil
}

func (c *Content) To(out *genai.Content) error {
	switch c.Type {
	case ContentText:
		out.Text = c.Text
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

type ContentType string

const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
	ContentVideoURL ContentType = "video_url"
)

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

type ToolCall struct {
	Index    int64  `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"` // function
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func (t *ToolCall) From(in *genai.ToolCall) {
	t.Index = 0 // Unsure
	t.Type = "function"
	t.ID = in.ID
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
}

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

type ChatResponse struct {
	ID      string   `json:"id"`
	Prompt  []string `json:"prompt"`
	Choices []struct {
		// Text  string `json:"text"`
		Index int64 `json:"index"`
		// The seed is returned as a int128.
		Seed         big.Int      `json:"seed"`
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
		Logprobs     struct {
			TokenIDs      []int64   `json:"token_ids"`
			Tokens        []string  `json:"tokens"`
			TokenLogprobs []float64 `json:"token_logprobs"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage    Usage     `json:"usage"`
	Created  base.Time `json:"created"`
	Model    string    `json:"model"`
	Object   string    `json:"object"` // "chat.completion"
	Warnings []struct {
		Message string `json:"message"`
	} `json:"warnings"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.CachedTokens,
			OutputTokens:      c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type FinishReason string

const (
	FinishStop         FinishReason = "stop"
	FinishEOS          FinishReason = "eos"
	FinishLength       FinishReason = "length"
	FinishFunctionCall FinishReason = "function_call"
	FinishToolCalls    FinishReason = "tool_calls"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishEOS:
		return genai.FinishedStopSequence
	case FinishLength:
		return genai.FinishedLength
	case FinishToolCalls, FinishFunctionCall:
		return genai.FinishedToolCalls
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

type Usage struct {
	PromptTokens        int64    `json:"prompt_tokens"`
	CompletionTokens    int64    `json:"completion_tokens"`
	TotalTokens         int64    `json:"total_tokens"`
	CachedTokens        int64    `json:"cached_tokens"`
	PromptTokensDetails struct{} `json:"prompt_tokens_details"`
}

type ChatStreamChunkResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"` // "chat.completion.chunk"
	Created base.Time `json:"created"`
	Model   string    `json:"model"`
	Choices []struct {
		Index int64   `json:"index"`
		Text  string  `json:"text"` // Duplicated to Delta.Text
		Seed  big.Int `json:"seed"`
		Delta struct {
			TokenID   int64      `json:"token_id"`
			Role      genai.Role `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Logprobs     struct{}     `json:"logprobs"`
		FinishReason FinishReason `json:"finish_reason"`
		StopReason   int64        `json:"stop_reason"` // Seems to be a token
		ToolCalls    []ToolCall   `json:"tool_calls"`  // TODO: Implement.
	} `json:"choices"`
	// SystemFingerprint string `json:"system_fingerprint"`
	Usage    Usage `json:"usage"`
	Warnings []struct {
		Message string `json:"message"`
	} `json:"warnings"`
}

type Model struct {
	ID            string    `json:"id"`
	Object        string    `json:"object"`
	Created       base.Time `json:"created"`
	Type          string    `json:"type"` // "chat", "moderation", "image"
	Running       bool      `json:"running"`
	DisplayName   string    `json:"display_name"`
	Organization  string    `json:"organization"`
	Link          string    `json:"link"`
	License       string    `json:"license"`
	ContextLength int64     `json:"context_length"`
	Config        struct {
		ChatTemplate    string   `json:"chat_template"`
		Stop            []string `json:"stop"`
		BosToken        string   `json:"bos_token"`
		EosToken        string   `json:"eos_token"`
		MaxOutputLength int64    `json:"max_output_length"`
	} `json:"config"`
	Pricing struct {
		Hourly   float64 `json:"hourly"`
		Input    float64 `json:"input"`
		Output   float64 `json:"output"`
		Base     float64 `json:"base"`
		Finetune float64 `json:"finetune"`
	} `json:"pricing"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	c := ""
	if m.Config.MaxOutputLength != 0 {
		c = fmt.Sprintf("%d/%d", m.ContextLength, m.Config.MaxOutputLength)
	} else {
		c = fmt.Sprintf("%d", m.ContextLength)
	}
	return fmt.Sprintf("%s (%s): %s Context: %s; in: %.2f$/Mt out: %.2f$/Mt", m.ID, m.Created.AsTime().Format("2006-01-02"), m.Type, c, m.Pricing.Input, m.Pricing.Output)
}

func (m *Model) Context() int64 {
	return m.ContextLength
}

// ModelsResponse represents the response structure for TogetherAI models listing
type ModelsResponse []Model

// ToModels converts TogetherAI models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(*r))
	for i := range *r {
		models[i] = &(*r)[i]
	}
	return models
}

// ImageRequest doesn't have a formal documentation.
//
// https://github.com/togethercomputer/together-python/blob/main/src/together/resources/images.py is the
// closest.
type ImageRequest struct {
	Prompt         string `json:"prompt"`
	Model          string `json:"model,omitzero"`
	Steps          int64  `json:"steps,omitzero"`  // Default 20
	Seed           int64  `json:"seed,omitzero"`   //
	N              int64  `json:"n,omitzero"`      // Default 1
	Height         int64  `json:"height,omitzero"` // Default 1024
	Width          int64  `json:"width,omitzero"`  // Default 1024
	NegativePrompt string `json:"negative_prompt,omitzero"`
	ImageURL       string `json:"image_url,omitzero"`
	Image          []byte `json:"image_base64,omitzero"`
}

// ImageResponse doesn't have a formal documentation.
//
// https://github.com/togethercomputer/together-python/blob/main/src/together/types/images.py is the
// closest.
type ImageResponse struct {
	ID     string            `json:"id"`
	Model  string            `json:"model"`
	Object string            `json:"object"` // "list"
	Data   []ImageChoiceData `json:"data"`
}

type ImageChoiceData struct {
	Index   int64  `json:"index"`
	B64JSON []byte `json:"b64_json"`
	URL     string `json:"url"`
	Timings struct {
		Inference float64 `json:"inference"`
	} `json:"timings"`
}

//

type ErrorResponse struct {
	ID    string `json:"id"`
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
		Param   string `json:"param"`
	} `json:"error"`
}

func (er *ErrorResponse) String() string {
	if er.Error.Code != "" {
		return fmt.Sprintf("error %s (%s): %s", er.Error.Code, er.Error.Type, er.Error.Message)
	}
	return fmt.Sprintf("error (%s): %s", er.Error.Type, er.Error.Message)
}

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Together.AI platform API.
//
// If apiKey is not provided, it tries to load it from the TOGETHER_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://api.together.ai/settings/api-keys
//
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.together.ai/docs/serverless-models
//
// Pass model base.PreferredCheap to use a good cheap model, base.PreferredGood for a good model or
// base.PreferredSOTA to use its SOTA model. Keep in mind that as providers cycle through new models, it's
// possible the model is not available anymore.
//
// wrapper can be used to throttle outgoing requests, record calls, etc. It defaults to base.DefaultTransport.
//
// # Vision
//
// We must select a model that supports video.
// https://docs.together.ai/docs/serverless-models#vision-models
func New(apiKey, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://api.together.xyz/settings/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("TOGETHER_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "TOGETHER_API_KEY", URL: apiKeyURL}
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.together.xyz/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "togetherai",
				APIKeyURL:    apiKeyURL,
				ClientJSON: httpjson.Client{
					Lenient: internal.BeLenient,
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
							Transport: &roundtrippers.RequestID{Transport: t},
						},
					},
				},
			},
		},
	}
	if err == nil && (model == base.PreferredCheap || model == base.PreferredGood || model == base.PreferredSOTA) {
		mdls, err2 := c.ListModels(context.Background())
		if err2 != nil {
			return nil, err2
		}
		cheap := model == base.PreferredCheap
		good := model == base.PreferredGood
		c.Model = ""
		price := 0.
		cutoff := time.Now().Add(-365 * 25 * time.Hour)
		for _, mdl := range mdls {
			m := mdl.(*Model)
			if m.Type != "chat" || m.Created.AsTime().Before(cutoff) || strings.Contains(m.ID, "-VL-") || strings.Contains(m.ID, "-Vision-") {
				continue
			}
			if cheap {
				if strings.HasPrefix(m.ID, "meta-llama/") && (price == 0 || price > m.Pricing.Output) {
					price = m.Pricing.Output
					// date = m.Created
					c.Model = m.ID
				}
			} else if good {
				if strings.HasPrefix(m.ID, "Qwen/Qwen") && (price == 0 || price < m.Pricing.Output) {
					// Take the most expensive
					price = m.Pricing.Output
					c.Model = m.ID
				}
			} else {
				if strings.HasPrefix(m.ID, "meta-llama/") && (price == 0 || price < m.Pricing.Output) {
					price = m.Pricing.Output
					c.Model = m.ID
				}
			}
		}
		if c.Model == "" {
			return nil, errors.New("failed to find a model automatically")
		}
	}
	return c, err
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	if c.isAudio(opts) || c.isImage(opts) {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.GenDoc(ctx, msgs[0], opts)
	}
	return c.ProviderGen.GenSync(ctx, msgs, opts)
}

func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if c.isAudio(opts) || c.isImage(opts) {
		return base.SimulateStream(ctx, c, msgs, chunks, opts)
	}
	return c.ProviderGen.GenStream(ctx, msgs, chunks, opts)
}

func (c *Client) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	if c.isAudio(opts) {
		// https://docs.together.ai/reference/audio-speech
		return genai.Result{}, errors.New("audio not implemented yet")
	}
	if c.isImage(opts) {
		// https://docs.together.ai/reference/post-images-generations
		res := genai.Result{}
		if err := c.Validate(); err != nil {
			return genai.Result{}, err
		}
		if err := msg.Validate(); err != nil {
			return res, err
		}
		if opts != nil {
			if err := opts.Validate(); err != nil {
				return res, err
			}
		}
		for i := range msg.Contents {
			if msg.Contents[i].Text == "" {
				return res, errors.New("only text can be passed as input")
			}
		}
		req := ImageRequest{
			Prompt: msg.AsText(),
			Model:  c.Model,
		}
		if opts != nil {
			switch v := opts.(type) {
			case *genai.OptionsImage:
				req.Height = int64(v.Height)
				req.Width = int64(v.Width)
				req.Seed = v.Seed
			case *genai.OptionsText:
				req.Seed = v.Seed
			}
		}
		resp := ImageResponse{}
		if err := c.DoRequest(ctx, "POST", "https://api.together.xyz/v1/images/generations", &req, &resp); err != nil {
			return res, err
		}
		res.Role = genai.Assistant
		res.Contents = make([]genai.Content, len(resp.Data))
		for i := range resp.Data {
			n := "content.jpg"
			if len(resp.Data) > 1 {
				n = fmt.Sprintf("content%d.jpg", i+1)
			}
			if url := resp.Data[i].URL; url != "" {
				res.Contents[i].Filename = n
				res.Contents[i].URL = url
			} else if d := resp.Data[i].B64JSON; len(d) != 0 {
				res.Contents[i].Filename = n
				res.Contents[i].Document = &bb.BytesBuffer{D: resp.Data[i].B64JSON}
			} else {
				return res, errors.New("internal error")
			}
		}
		return res, nil
	}
	return genai.Result{}, errors.New("can only generate audio and images")
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.together.ai/reference/models-1
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, "https://api.together.xyz/v1/models")
}

func (c *Client) isAudio(opts genai.Options) bool {
	// TODO: Use Scoreboard list. The problem is that it's recursive while recreating the scoreboard, and that
	// the server HTTP 500 on onsupported models.
	if strings.HasPrefix(c.Model, "cartesia/") {
		return true
	}
	return opts != nil && opts.Modality() == genai.ModalityAudio
}

func (c *Client) isImage(opts genai.Options) bool {
	// TODO: Use Scoreboard list. The problem is that it's recursive while recreating the scoreboard, and that
	// the server HTTP 500 on onsupported models.
	if strings.HasPrefix(c.Model, "black-forest-labs/") {
		return true
	}
	return opts != nil && opts.Modality() == genai.ModalityImage
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		if pkt.Usage.TotalTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.InputCachedTokens = pkt.Usage.CachedTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
		}
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Choices[0].FinishReason != "" {
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		// There's only one at a time ever.
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt.Choices[0].Delta.ToolCalls)
		}
		// TODO: It's new, I'm not sure it's used.
		if len(pkt.Choices[0].ToolCalls) > 0 {
			return fmt.Errorf("implement tool calls: %#v", pkt.Choices[0].ToolCalls)
		}
		// TogetherAI streams the arguments. Buffer the arguments to send the fragment as a
		// whole tool call.
		if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
			t := pkt.Choices[0].Delta.ToolCalls[0]
			if t.ID != "" {
				// A new call.
				if pendingCall.ID != "" {
					// Flush.
					f := genai.ContentFragment{ToolCall: genai.ToolCall{
						ID:        pendingCall.ID,
						Name:      pendingCall.Function.Name,
						Arguments: pendingCall.Function.Arguments,
					}}
					if err := result.Accumulate(f); err != nil {
						return err
					}
					chunks <- f
				}
				pendingCall = t
				continue
			}
			if pendingCall.ID != "" {
				// Continuation.
				pendingCall.Function.Arguments += t.Function.Arguments
				continue
			}
		} else {
			if pendingCall.ID != "" {
				// Flush.
				f := genai.ContentFragment{ToolCall: genai.ToolCall{
					ID:        pendingCall.ID,
					Name:      pendingCall.Function.Name,
					Arguments: pendingCall.Function.Arguments,
				}}
				if err := result.Accumulate(f); err != nil {
					return err
				}
				chunks <- f
			}
		}
		f := genai.ContentFragment{TextFragment: pkt.Choices[0].Delta.Content}
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
	_ genai.ProviderGenDoc     = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
