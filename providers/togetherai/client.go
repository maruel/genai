// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package togetherai implements a client for the Together.ai API.
//
// It is described at https://docs.together.ai/docs/
package togetherai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"math"
	"math/big"
	"net/http"
	"os"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

// Official python client library at https://github.com/togethercomputer/together-python/tree/main/src/together

// Scoreboard for TogetherAI.
//
// See https://docs.together.ai/docs/serverless-models and https://api.together.ai/models
var Scoreboard = scoreboard.Score{
	Warnings: []string{
		"No model supports \"required\" tool calling, thus it's marked as \"flaky\" everywhere.",
		"Tool calling is solid with llama 3.3 70B quantized in FP8 (-Turbo) but is flaky in more recent models.",
		"Suffix \"-Turbo\" means FP8 quantization.",
		"Suffix \"-Lite\" means INT4 quantization.",
		"Suffix \"-Free\" has lower rate limits.",
	},
	Country:      "US",
	DashboardURL: "https://api.together.ai/settings/billing",
	Scenarios: []scoreboard.Scenario{
		{
			Comments: "Tool calling is flaky because Together.AI only supports tool_choice auto, not required.",
			Models:   []string{"meta-llama/Llama-4-Scout-17B-16E-Instruct"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.Flaky,
				IndecisiveTool:   scoreboard.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.Flaky,
				IndecisiveTool:   scoreboard.Flaky,
				JSON:             true,
				JSONSchema:       true,
				TopLogprobs:      true,
			},
		},
		{
			Comments: "Tool calling is flaky because Together.AI only supports tool_choice auto, not required.",
			Models:   []string{"meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
				NoMaxTokens:      true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
				NoMaxTokens:      true,
			},
		},
		{
			Models: []string{"meta-llama/Llama-Vision-Free"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.Flaky,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
				NoMaxTokens:      true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.Flaky,
				JSONSchema:       true,
				TopLogprobs:      true,
			},
		},
		{
			Models: []string{"moonshotai/Kimi-K2-Instruct"},
			In:     map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.Flaky,
				BiasedTool:       scoreboard.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
		},
		{
			Comments: "FinishReason is only broken with JSON.",
			Models:   []string{"mistralai/Mistral-Small-24B-Instruct-2501"},
			In:       map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:      map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits:   true,
				Tools:              scoreboard.Flaky,
				BiasedTool:         scoreboard.Flaky,
				IndecisiveTool:     scoreboard.Flaky,
				JSON:               true,
				Seed:               true,
				TopLogprobs:        true,
				BrokenFinishReason: true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits:   true,
				Tools:              scoreboard.Flaky,
				BiasedTool:         scoreboard.Flaky,
				IndecisiveTool:     scoreboard.Flaky,
				JSON:               true,
				Seed:               true,
				TopLogprobs:        true,
				BrokenFinishReason: true,
			},
		},
		{
			Models: []string{"black-forest-labs/FLUX.1-schnell"},
			In:     map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			Out: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityImage: {
					URL:              true,
					SupportedFormats: []string{"image/jpeg"},
				},
			},
			GenDoc: &scoreboard.FunctionalityDoc{
				Seed:               true,
				BrokenTokenUsage:   scoreboard.True,
				BrokenFinishReason: true,
			},
		},
		{
			Comments: "Untested",
			Models: []string{
				"Alibaba-NLP/gte-modernbert-base",
				"NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
				"Qwen/QwQ-32B",
				"Qwen/Qwen2-72B-Instruct",
				"Qwen/Qwen2.5-VL-72B-Instruct",
				"Qwen/Qwen2.5-72B-Instruct-Turbo",
				"Qwen/Qwen2.5-7B-Instruct-Turbo",
				"Qwen/Qwen2.5-Coder-32B-Instruct",
				"Qwen/Qwen3-235B-A22B-Instruct-2507-tput",
				"Qwen/Qwen3-235B-A22B-Thinking-2507",
				"Qwen/Qwen3-235B-A22B-fp8-tput",
				"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
				"Salesforce/Llama-Rank-V1",
				"Virtue-AI/VirtueGuard-Text-Lite",
				"arcee-ai/coder-large",
				"arize-ai/qwen-2-1.5b-instruct",
				"togethercomputer/m2-bert-80M-32k-retrieval",
				"deepcogito/cogito-v2-preview-llama-70B",
				"arcee-ai/AFM-4.5B",
				"deepcogito/cogito-v2-preview-llama-405B",
				"arcee-ai/maestro-reasoning",
				"arcee-ai/virtuoso-large",
				"arcee_ai/arcee-spotlight",
				"black-forest-labs/FLUX.1-canny",
				"black-forest-labs/FLUX.1-depth",
				"black-forest-labs/FLUX.1-dev",
				"black-forest-labs/FLUX.1-dev-lora",
				"black-forest-labs/FLUX.1-kontext-dev",
				"black-forest-labs/FLUX.1-kontext-max",
				"black-forest-labs/FLUX.1-kontext-pro",
				"black-forest-labs/FLUX.1-krea-dev",
				"black-forest-labs/FLUX.1-pro",
				"black-forest-labs/FLUX.1-redux",
				"black-forest-labs/FLUX.1-schnell-Free",
				"black-forest-labs/FLUX.1.1-pro",
				"cartesia/sonic",
				"cartesia/sonic-2",
				"deepcogito/cogito-v2-preview-deepseek-671b",
				"deepcogito/cogito-v2-preview-llama-109B-MoE",
				"deepseek-ai/DeepSeek-R1-0528-tput",
				"deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
				"deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
				"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
				"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
				"deepseek-ai/DeepSeek-V3",
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
				"mistralai/Mistral-7B-Instruct-v0.3",
				"mistralai/Mixtral-8x7B-Instruct-v0.1",
				"mixedbread-ai/Mxbai-Rerank-Large-V2",
				"nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
				"openai/gpt-oss-20b",
				"openai/whisper-large-v3",
				"perplexity-ai/r1-1776",
				"scb10x/scb10x-llama3-1-typhoon2-70b-instruct",
				"scb10x/scb10x-typhoon-2-1-gemma3-12b",
				"togethercomputer/MoA-1",
				"togethercomputer/MoA-1-Turbo",
				"togethercomputer/Refuel-Llm-V2",
				"togethercomputer/Refuel-Llm-V2-Small",
				"zai-org/GLM-4.5-Air-FP8",
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
	Logprobs                      int64              `json:"logprobs,omitzero"` // Actually toplogprobs; [0, 20]
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
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	c.Model = model
	// Validate messages
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string
	sp := ""
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *genai.OptionsText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			c.Seed = v.Seed
			c.Logprobs = v.TopLogprobs
			// TODO: Toplogprobs are not returned unless streaming. lol. Sadly we do not know yet here if streaming
			// is enabled.
			// if v.TopLogprobs > 1 && !Stream {
			// 	unsupported = append(unsupported, "TopLogprobs")
			// }
			c.TopK = v.TopK
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				// Warning: using a model small may fail.
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
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

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: "text", Text: sp}}})
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
					errs = append(errs, fmt.Errorf("message #%d: tool call results #%d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		} else {
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
			} else {
				c.Messages = append(c.Messages, newMsg)
			}
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

// Message is documented at https://docs.together.ai/reference/chat-completions-1
type Message struct {
	Role      string   `json:"role,omitzero"` // "system", "assistant", "user"
	Content   Contents `json:"content,omitzero"`
	Reasoning struct{} `json:"reasoning,omitzero"`
	// Warning: using a small model may fail.
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
}

// From must be called with at most one ToolCallResults.
func (m *Message) From(in *genai.Message) error {
	if len(in.ToolCallResults) > 1 {
		return errors.New("internal error")
	}
	switch r := in.Role(); r {
	case "user", "assistant":
		m.Role = r
	case "computer":
		m.Role = "tool"
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	if len(in.Requests) != 0 {
		m.Content = make([]Content, 0, len(in.Requests))
		for i := range in.Requests {
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request #%d: %w", i, err)
			}
		}
	}
	if len(in.Replies) != 0 {
		for i := range in.Replies {
			if in.Replies[i].Thinking != "" {
				continue
			}
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
					return fmt.Errorf("reply #%d: %w", i, err)
				}
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	if len(m.Content) != 0 {
		out.Replies = make([]genai.Reply, len(m.Content))
		for i := range m.Content {
			if err := m.Content[i].To(&out.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

type Contents []Content

// UnmarshalJSON implements json.Unmarshaler.
//
// Together.AI replies with content as a string.
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
	*c = []Content{{Type: "text", Text: s}}
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

func (c *Content) FromRequest(in *genai.Request) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case strings.HasPrefix(mimeType, "video/"):
			c.Type = ContentVideoURL
			if in.Doc.URL == "" {
				c.VideoURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.VideoURL.URL = in.Doc.URL
			}
		case strings.HasPrefix(mimeType, "text/plain"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			c.Text = string(data)
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Request type")
}

func (c *Content) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return errors.New("field ToolCall.Opaque not supported")
	}
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case strings.HasPrefix(mimeType, "video/"):
			c.Type = ContentVideoURL
			if in.Doc.URL == "" {
				c.VideoURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.VideoURL.URL = in.Doc.URL
			}
		case strings.HasPrefix(mimeType, "text/plain"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			c.Text = string(data)
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Reply type")
}

func (c *Content) To(out *genai.Reply) error {
	switch c.Type {
	case ContentText:
		out.Text = c.Text
	case ContentImageURL, ContentVideoURL:
		fallthrough
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
		Logprobs     Logprobs     `json:"logprobs"`
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
			TotalTokens:       c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	out.Logprobs = c.Choices[0].Logprobs.To()
	err := c.Choices[0].Message.To(&out.Message)
	if err == nil && len(c.Warnings) != 0 {
		uce := &genai.UnsupportedContinuableError{}
		for _, w := range c.Warnings {
			if strings.Contains(w.Message, "tool_choice") {
				uce.Unsupported = append(uce.Unsupported, "ToolCallRequest")
			} else {
				uce.Unsupported = append(uce.Unsupported, w.Message)
			}
		}
		return out, uce
	}
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

type Logprobs struct {
	Tokens        []string   `json:"tokens"`
	TokenLogprobs []float64  `json:"token_logprobs"`
	TokenIDs      []struct{} `json:"token_ids"` // Not set.
}

func (l *Logprobs) To() []genai.Logprobs {
	if len(l.Tokens) == 0 {
		return nil
	}
	// Toplogprobs are not returned when not streaming (!!)
	out := make([]genai.Logprobs, 0, len(l.Tokens))
	for i := range l.Tokens {
		out = append(out, genai.Logprobs{Text: l.Tokens[i], Logprob: l.TokenLogprobs[i]})
	}
	return out
}

type LogprobsChunk struct {
	Tokens        []string  `json:"tokens"`
	TokenLogprobs []float64 `json:"token_logprobs"`
	TopLogprobs   [][]struct {
		Token   string  `json:"token"`
		Logprob float64 `json:"logprob"`
	} `json:"top_logprobs"`
}

func (l *LogprobsChunk) ToLogprobs() []genai.Logprobs {
	if len(l.Tokens) == 0 {
		return nil
	}
	out := make([]genai.Logprobs, len(l.Tokens))
	for i := range l.Tokens {
		out[i] = genai.Logprobs{Text: l.Tokens[i], Logprob: l.TokenLogprobs[i], TopLogprobs: make([]genai.TopLogprob, 0, len(l.TopLogprobs[i]))}
		for _, tlp := range l.TopLogprobs[i] {
			out[i].TopLogprobs = append(out[i].TopLogprobs, genai.TopLogprob{Text: tlp.Token, Logprob: tlp.Logprob})
		}
	}
	return out
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
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
			Reasoning struct{}   `json:"reasoning"`
		} `json:"delta"`
		Logprobs     LogprobsChunk `json:"logprobs"`
		FinishReason FinishReason  `json:"finish_reason"`
		MatchedStop  int64         `json:"matched_stop"`
		StopReason   int64         `json:"stop_reason"` // Seems to be a token
		ToolCalls    []ToolCall    `json:"tool_calls"`  // TODO: Implement.
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

func (i *ImageRequest) Init(msg genai.Message, model string, opts ...genai.Options) error {
	if err := msg.Validate(); err != nil {
		return err
	}
	for i := range msg.Requests {
		if msg.Requests[i].Text == "" {
			return errors.New("only text can be passed as input")
		}
	}
	i.Prompt = msg.String()
	i.Model = model
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *genai.OptionsImage:
			i.Height = int64(v.Height)
			i.Width = int64(v.Width)
			i.Seed = v.Seed
		case *genai.OptionsText:
			i.Seed = v.Seed
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
	}
	return nil
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
	ID       string `json:"id"`
	ErrorVal struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
		Param   string `json:"param"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	if er.ErrorVal.Code != "" {
		return fmt.Sprintf("%s (%s): %s", er.ErrorVal.Code, er.ErrorVal.Type, er.ErrorVal.Message)
	}
	if er.ErrorVal.Type != "" {
		return fmt.Sprintf("%s: %s", er.ErrorVal.Type, er.ErrorVal.Message)
	}
	return er.ErrorVal.Message
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider.
type Client struct {
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Together.AI platform API.
//
// If apiKey is not provided, it tries to load it from the TOGETHER_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://api.together.ai/settings/api-keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.together.ai/docs/serverless-models
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
//
// # Vision
//
// We must select a model that supports video.
// https://docs.together.ai/docs/serverless-models#vision-models
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://api.together.ai/settings/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("TOGETHER_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "TOGETHER_API_KEY", URL: apiKeyURL}
		}
	}
	switch len(opts.OutputModalities) {
	case 0:
	case 1:
		switch opts.OutputModalities[0] {
		case genai.ModalityImage, genai.ModalityText:
		case genai.ModalityAudio:
			// TODO: Add support for audio.
			return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are implemented (send PR to add support)", opts.OutputModalities)
		case genai.ModalityDocument, genai.ModalityVideo:
			fallthrough
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are implemented", opts.OutputModalities)
		}
	default:
		return nil, fmt.Errorf("unexpected option Modalities %s, only image or text are implemented (send PR to add support)", opts.OutputModalities)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:           "https://api.together.xyz/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			ProcessHeaders:       processHeaders,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch opts.Model {
		case genai.ModelNone:
		case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA, "":
			if len(opts.OutputModalities) == 0 || opts.OutputModalities[0] == genai.ModalityText {
				if c.impl.Model, err = c.selectBestTextModel(ctx, opts.Model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{genai.ModalityText}
			} else {
				if c.impl.Model, err = c.selectBestImageModel(ctx, opts.Model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{genai.ModalityImage}
			}
		default:
			c.impl.Model = opts.Model
			if len(opts.OutputModalities) == 0 {
				c.impl.OutputModalities, err = c.detectModelModalities(ctx, opts.Model)
			} else {
				c.impl.OutputModalities = opts.OutputModalities
			}
		}
	}
	return c, err
}

// detectModelModalities tries its best to figure out the modality of a model
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) detectModelModalities(ctx context.Context, model string) (genai.Modalities, error) {
	// TODO: Detect if it is an image model.
	// Temporarily disabled to ease with the transition away from GenDoc / ProviderGenDoc.
	// I must not forget to enable this back once I removed GenDoc!
	if false {
		mdls, err2 := c.ListModels(ctx)
		if err2 != nil {
			return nil, fmt.Errorf("failed to detect the output modality for the model %s: %w", model, err2)
		}
		for _, mdl := range mdls {
			if m := mdl.(*Model); m.ID == model {
				switch m.Type {
				case "chat":
					return genai.Modalities{genai.ModalityText}, nil
				case "image":
					return genai.Modalities{genai.ModalityImage}, nil
				default:
					return nil, fmt.Errorf("failed to detect the output modality for the model %s: found type %s", model, m.Type)
				}
			}
		}
		return nil, fmt.Errorf("failed to automatically detect the model modality: model %s not found", model)
	}
	if strings.HasPrefix(model, "black-forest-labs/") {
		return genai.Modalities{genai.ModalityImage}, nil
	} else {
		return genai.Modalities{genai.ModalityText}, nil
	}
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == genai.ModelCheap
	good := preference == genai.ModelGood || preference == ""
	selectedModel := ""
	price := 0.
	if cheap || good {
		price = math.Inf(1)
	}
	cutoff := time.Now().Add(-365 * 25 * time.Hour)
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if m.Type != "chat" || m.Created.AsTime().Before(cutoff) || strings.Contains(m.ID, "-VL-") || strings.Contains(m.ID, "-Vision-") {
			continue
		}
		if cheap {
			if strings.HasPrefix(m.ID, "meta-llama/") && (m.Pricing.Output == 0 || (price > m.Pricing.Output)) {
				price = m.Pricing.Output
				// date = m.Created
				selectedModel = m.ID
			}
		} else if good {
			if strings.HasPrefix(m.ID, "Qwen/Qwen") && price > m.Pricing.Output {
				// Take the most expensive
				price = m.Pricing.Output
				selectedModel = m.ID
			}
		} else {
			if strings.HasPrefix(m.ID, "Qwen/Qwen") && price < m.Pricing.Output {
				price = m.Pricing.Output
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestImageModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestImageModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == genai.ModelCheap
	good := preference == genai.ModelGood || preference == ""
	selectedModel := ""
	// As of August 2025, price, created date are not set. This greatly limits the automatic model selection.
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if m.Type != "image" {
			continue
		}
		if cheap {
			if strings.HasSuffix(m.ID, "-schnell") && (selectedModel == "" || m.ID > selectedModel) {
				selectedModel = m.ID
			}
		} else if good {
			if strings.HasSuffix(m.ID, "-dev") && (selectedModel == "" || m.ID > selectedModel) {
				selectedModel = m.ID
			}
		} else {
			if strings.HasSuffix(m.ID, "-pro") && !strings.Contains(m.ID, "kontext") && (selectedModel == "" || m.ID > selectedModel) {
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "togetherai"
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

// Scoreboard implements scoreboard.ProviderScore.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	if c.impl.OutputModalities[0] == genai.ModalityText {
		return c.impl.GenSync(ctx, msgs, opts...)
	}
	if len(msgs) != 1 {
		return genai.Result{}, errors.New("must pass exactly one Message")
	}
	return c.genImage(ctx, msgs[0], opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	if c.impl.OutputModalities[0] == genai.ModalityText {
		return c.impl.GenStream(ctx, msgs, opts...)
	}
	return base.SimulateStream(ctx, c, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	return c.impl.GenStreamRaw(ctx, in, out)
}

// genImage generates images.
func (c *Client) genImage(ctx context.Context, msg genai.Message, opts ...genai.Options) (genai.Result, error) {
	if c.isAudio() {
		// https://docs.together.ai/reference/audio-speech
		return genai.Result{}, errors.New("audio not implemented yet")
	}
	if c.isImage() {
		// https://docs.together.ai/reference/post-images-generations
		res := genai.Result{}
		if err := c.impl.Validate(); err != nil {
			return genai.Result{}, err
		}
		req := ImageRequest{}
		if err := req.Init(msg, c.impl.Model, opts...); err != nil {
			return res, err
		}
		resp := ImageResponse{}
		if err := c.impl.DoRequest(ctx, "POST", "https://api.together.xyz/v1/images/generations", &req, &resp); err != nil {
			return res, err
		}
		res.Replies = make([]genai.Reply, len(resp.Data))
		for i := range resp.Data {
			n := "content.jpg"
			if len(resp.Data) > 1 {
				n = fmt.Sprintf("content%d.jpg", i+1)
			}
			if url := resp.Data[i].URL; url != "" {
				res.Replies[i].Doc = genai.Doc{Filename: n, URL: url}
			} else if d := resp.Data[i].B64JSON; len(d) != 0 {
				res.Replies[i].Doc = genai.Doc{Filename: n, Src: &bb.BytesBuffer{D: resp.Data[i].B64JSON}}
			} else {
				return res, errors.New("internal error")
			}
		}
		return res, res.Validate()
	}
	return genai.Result{}, errors.New("can only generate audio and images")
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.together.ai/reference/models-1
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.together.xyz/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

func (c *Client) isAudio() bool {
	// TODO: Use Scoreboard list. The problem is that it's recursive while recreating the scoreboard, and that
	// the server HTTP 500 on onsupported models.
	return strings.HasPrefix(c.impl.Model, "cartesia/")
}

func (c *Client) isImage() bool {
	// TODO: Use Scoreboard list. The problem is that it's recursive while recreating the scoreboard, and that
	// the server HTTP 500 on onsupported models.
	return strings.HasPrefix(c.impl.Model, "black-forest-labs/")
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	var warnings []string
	for pkt := range ch {
		if pkt.Usage.TotalTokens != 0 {
			result.Usage.InputTokens = pkt.Usage.PromptTokens
			result.Usage.InputCachedTokens = pkt.Usage.CachedTokens
			result.Usage.OutputTokens = pkt.Usage.CompletionTokens
			result.Usage.TotalTokens = pkt.Usage.TotalTokens
		}
		for _, w := range pkt.Warnings {
			warnings = append(warnings, w.Message)
		}
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Choices[0].FinishReason != "" {
			result.Usage.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
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
					f := genai.ReplyFragment{ToolCall: genai.ToolCall{
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
				f := genai.ReplyFragment{ToolCall: genai.ToolCall{
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
		f := genai.ReplyFragment{TextFragment: pkt.Choices[0].Delta.Content}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		if len(pkt.Choices[0].Logprobs.Tokens) != 0 {
			result.Logprobs = append(result.Logprobs, pkt.Choices[0].Logprobs.ToLogprobs()...)
		}
	}
	if len(warnings) != 0 {
		uce := &genai.UnsupportedContinuableError{}
		for _, w := range warnings {
			if strings.Contains(w, "tool_choice") {
				uce.Unsupported = append(uce.Unsupported, "ToolCallRequest")
			} else {
				uce.Unsupported = append(uce.Unsupported, w)
			}
		}
		return uce
	}
	return nil
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	limitReq, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit"), 10, 64)
	remainingReq, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining"), 10, 64)

	limitTok, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	remainingTok, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)

	reset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset"))

	if limitReq > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     limitReq,
			Remaining: remainingReq,
			Reset:     time.Now().Add(reset * time.Second).Round(10 * time.Millisecond), // Just guessing.
		})
	}
	if limitTok > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     limitTok,
			Remaining: remainingTok,
			Reset:     time.Now().Add(reset * time.Second).Round(10 * time.Millisecond), // Just guessing.
		})
	}
	return limits
}

var (
	_ genai.Provider           = &Client{}
	_ scoreboard.ProviderScore = &Client{}
)
