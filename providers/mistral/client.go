// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package mistral implements a client for the Mistral API.
//
// It is described at https://docs.mistral.ai/api/
package mistral

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Mistral.
var Scoreboard = scoreboard.Score{
	Warnings: []string{
		"Mistral supports more than what the client currently supports.",
		"Tool calling is excellent and unbiased for non \"mini\" models.",
		"PDF doesn't support inline document while images do.",
		"Rate limit depends on your tier: https://docs.mistral.ai/deployment/laplateforme/tier/",
	},
	Country:      "FR",
	DashboardURL: "https://console.mistral.ai/usage",
	Scenarios: []scoreboard.Scenario{
		{
			Models: []string{"ministral-3b-latest"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityDocument: {
					URL:              true,
					SupportedFormats: []string{"application/pdf"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
			},
			GenStream: &scoreboard.FunctionalityText{
				Tools:      scoreboard.True,
				BiasedTool: scoreboard.True,
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
		},
		{
			Models: []string{"mistral-small-latest"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityDocument: {
					URL:              true,
					SupportedFormats: []string{"application/pdf"},
				},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
			},
			GenStream: &scoreboard.FunctionalityText{
				Tools:      scoreboard.True,
				BiasedTool: scoreboard.True,
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
		},
		{
			Models: []string{"pixtral-12b-latest"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityDocument: {
					URL:              true,
					SupportedFormats: []string{"application/pdf"},
				},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
			},
			GenStream: &scoreboard.FunctionalityText{
				Tools:      scoreboard.True,
				BiasedTool: scoreboard.True,
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
		},
		{
			Models: []string{"voxtral-small-latest"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityAudio: {
					Inline:           true,
					SupportedFormats: []string{"audio/flac", "audio/mp3", "audio/ogg", "audio/wav"},
				},
				genai.ModalityDocument: {
					URL:              true,
					SupportedFormats: []string{"application/pdf"},
				},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
			},
			GenStream: &scoreboard.FunctionalityText{
				Tools:      scoreboard.True,
				BiasedTool: scoreboard.True,
				JSON:       true,
				JSONSchema: true,
				Seed:       true,
			},
		},
		{
			Comments: "Untested",
			Models: []string{
				"codestral-2411-rc5",
				"codestral-2412",
				"codestral-2501",
				"codestral-2508",
				"codestral-latest",
				"devstral-medium-2507",
				"devstral-medium-latest",
				"devstral-small-2505",
				"devstral-small-2507",
				"devstral-small-latest",
				"magistral-medium-2506",
				"magistral-medium-latest",
				"magistral-small-2506",
				"magistral-small-latest",
				"ministral-3b-2410",
				"ministral-8b-2410",
				"ministral-8b-latest",
				"mistral-large-2407",
				"mistral-large-2411",
				"mistral-large-latest",
				"mistral-large-pixtral-2411",
				"mistral-medium",
				"mistral-medium-2505",
				"mistral-medium-2508",
				"mistral-medium-latest",
				"mistral-ocr-2503",
				"mistral-ocr-2505",
				"mistral-ocr-latest",
				"mistral-saba-2502",
				"mistral-saba-latest",
				"mistral-small",
				"mistral-small-2312",
				"mistral-small-2409",
				"mistral-small-2501",
				"mistral-small-2503",
				"mistral-small-2506",
				"mistral-tiny",
				"mistral-tiny-2312",
				"mistral-tiny-2407",
				"mistral-tiny-latest",
				"open-mistral-7b",
				"open-mistral-nemo",
				"open-mistral-nemo-2407",
				"open-mixtral-8x22b",
				"open-mixtral-8x22b-2404",
				"open-mixtral-8x7b",
				"pixtral-12b",
				"pixtral-12b-2409",
				"pixtral-large-2411",
				"pixtral-large-latest",
				"voxtral-mini-2507",
				"voxtral-mini-latest",
				"voxtral-mini-transcribe-2507",
				"voxtral-small-2507",
			},
		},
		{
			Comments: "Unsupported",
			Models: []string{
				"codestral-embed",
				"codestral-embed-2505",
				"magistral-medium-2507",
				"magistral-small-2507",
				"mistral-embed",
				"mistral-moderation-2411",
				"mistral-moderation-latest",
			},
		},
	},
}

// ChatRequest is documented at https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
type ChatRequest struct {
	Model          string    `json:"model"`
	Temperature    float64   `json:"temperature,omitzero"` // [0, 2]
	TopP           float64   `json:"top_p,omitzero"`       // [0, 1]
	MaxTokens      int64     `json:"max_tokens,omitzero"`
	Stream         bool      `json:"stream"`
	Stop           []string  `json:"stop,omitzero"` // keywords to stop completion
	RandomSeed     int64     `json:"random_seed,omitzero"`
	Messages       []Message `json:"messages"`
	ResponseFormat struct {
		Type       string `json:"type,omitzero"` // "text", "json_object", "json_schema"
		JSONSchema struct {
			Name        string             `json:"name,omitzero"`
			Description string             `json:"description,omitzero"`
			Strict      bool               `json:"strict,omitzero"`
			Schema      *jsonschema.Schema `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Tools []Tool `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice       string  `json:"tool_choice,omitzero"`       // "auto", "none", "any", "required"
	PresencePenalty  float64 `json:"presence_penalty,omitzero"`  // [-2.0, 2.0]
	FrequencyPenalty float64 `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	N                int64   `json:"n,omitzero"`                 // Number of choices
	Prediction       struct {
		// Enable users to specify expected results, optimizing response times by
		// leveraging known or predictable content. This approach is especially
		// effective for updating text documents or code files with minimal
		// changes, reducing latency while maintaining high-quality results.
		Type    string `json:"type,omitzero"` // "content"
		Content string `json:"content,omitzero"`
	} `json:"prediction,omitzero"`
	SafePrompt bool `json:"safe_prompt,omitzero"`

	// See https://docs.mistral.ai/capabilities/document/
	DocumentImageLimit int64 `json:"document_image_limit,omitzero"`
	DocumentPageLimit  int64 `json:"document_page_limit,omitzero"`
	IncludeImageBase64 bool  `json:"include_image_base64,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	c.Model = model
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
			c.RandomSeed = v.Seed
			if v.TopK != 0 {
				unsupported = append(unsupported, "TopK")
			}
			if v.TopLogprobs > 0 {
				unsupported = append(unsupported, "TopLogprobs")
			}
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json_schema"
				// Mistral requires a name.
				c.ResponseFormat.JSONSchema.Name = "response"
				c.ResponseFormat.JSONSchema.Strict = true
				c.ResponseFormat.JSONSchema.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
			if len(v.Tools) != 0 {
				switch v.ToolCallRequest {
				case genai.ToolCallAny:
					c.ToolChoice = "auto"
				case genai.ToolCallRequired:
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
					// This costs a lot more.
					c.Tools[i].Function.Strict = true
				}
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: ContentText, Text: sp}}})
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

// Message is documented at https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
//
// See the python implementation at
// https://github.com/mistralai/mistral-common/blob/main/src/mistral_common/protocol/instruct/messages.py
type Message struct {
	Role       string     `json:"role"`             // "system", "assistant", "user", "tool"
	Content    []Content  `json:"content,omitzero"` // For system and assistant, must be at most a single string.
	Prefix     bool       `json:"prefix,omitzero"`  // Whether the message is a prefix
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
	Name       string     `json:"name,omitzero"`
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
		m.Content = make([]Content, len(in.Requests))
		for i := range in.Requests {
			if err := m.Content[i].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request #%d: %w", i, err)
			}
		}
	}
	if len(in.Replies) != 0 {
		for i := range in.Replies {
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[i].From(&in.Replies[i].ToolCall); err != nil {
					return fmt.Errorf("reply #%d: %w", i, err)
				}
				continue
			}
			if err := m.Content[i].FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Name = in.ToolCallResults[0].Name
		// Mistral supports images urls!!
		m.Content = []Content{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
	}
	return nil
}

// Content is a piece of information sent back and forth.
//
// Only the user can send non-textual information.
type Content struct {
	Type ContentType `json:"type"`

	// Type == ContentText
	Text string `json:"text,omitzero"`

	// Type == ContentImageURL
	ImageURL struct {
		URL    string `json:"url,omitzero"`    // Can be inline.
		Detail string `json:"detail,omitzero"` // undocumented, likely "auto" like OpenAI
	} `json:"image_url,omitzero"`
	ModelConfig struct{} `json:"model_config,omitzero"`

	// Type == ContentDocumentURL
	DocumentURL  string `json:"document_url,omitzero"`
	DocumentName string `json:"document_name,omitzero"`

	// Type == ContentReference
	ReferenceIDs []int64 `json:"reference_ids,omitzero"`

	// Type == ContentInputAudio
	InputAudio []byte `json:"input_audio,omitzero"`
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
		case strings.HasPrefix(mimeType, "audio/"):
			if in.Doc.URL != "" {
				return errors.New("unsupported URL audio reference")
			}
			c.Type = ContentInputAudio
			c.InputAudio = data
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case mimeType == "application/pdf":
			c.Type = ContentDocumentURL
			if in.Doc.URL == "" {
				// Inexplicably, Mistral supports inline images but not PDF.
				return errors.New("unsupported inline document")
			}
			c.DocumentName = in.Doc.GetFilename()
			c.DocumentURL = in.Doc.URL
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
		return errors.New("field Reply.Opaque not supported")
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
		case strings.HasPrefix(mimeType, "audio/"):
			if in.Doc.URL != "" {
				return errors.New("unsupported URL audio reference")
			}
			c.Type = ContentInputAudio
			c.InputAudio = data
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case mimeType == "application/pdf":
			c.Type = ContentDocumentURL
			if in.Doc.URL == "" {
				// Inexplicably, Mistral supports inline images but not PDF.
				return errors.New("unsupported inline document")
			}
			c.DocumentName = in.Doc.GetFilename()
			c.DocumentURL = in.Doc.URL
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

// ContentType is the type of a content block in a message.
//
// I got the whole list from an error message by sending a bad content type.
type ContentType string

const (
	ContentText        ContentType = "text"
	ContentImageURL    ContentType = "image_url"
	ContentDocumentURL ContentType = "document_url"
	ContentReference   ContentType = "reference"
	ContentBBox        ContentType = "bbox"
	ContentFileURL     ContentType = "file_url"
	ContentInputAudio  ContentType = "input_audio"
	ContentFile        ContentType = "file"
	ContentThinking    ContentType = "thinking"
	ContentAudioURL    ContentType = "audio_url"
)

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Strict      bool               `json:"strict,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

type ChatResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"` // "chat.completion"
	Model   string    `json:"model"`
	Created base.Time `json:"created"`
	Choices []struct {
		FinishReason FinishReason    `json:"finish_reason"`
		Index        int64           `json:"index"`
		Message      MessageResponse `json:"message"`
		Logprobs     struct{}        `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishContentFilter FinishReason = "content_filter"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishLength:
		return genai.FinishedLength
	case FinishToolCalls:
		return genai.FinishedToolCalls
	case FinishContentFilter:
		return genai.FinishedContentFilter
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

type Usage struct {
	PromptTokens       int64 `json:"prompt_tokens"`
	CompletionTokens   int64 `json:"completion_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	PromptAudioSeconds int64 `json:"prompt_audio_seconds"`
}

type MessageResponse struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Prefix    bool       `json:"prefix"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	if m.Content != "" {
		out.Replies = []genai.Reply{{Text: m.Content}}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	return nil
}

type ToolCall struct {
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
	Index int64 `json:"index,omitzero"`
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

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Mistral doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
			TotalTokens:  c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type ChatStreamChunkResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"` // "chat.completion.chunk"
	Created base.Time `json:"created"`
	Model   string    `json:"model"`
	Choices []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason FinishReason `json:"finish_reason"`
		Logprobs     struct{}     `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// Model is documented at https://docs.mistral.ai/api/#tag/models/operation/retrieve_model_v1_models__model_id__get
type Model struct {
	ID           string    `json:"id"`
	Object       string    `json:"object"`
	Created      base.Time `json:"created"`
	OwnedBy      string    `json:"owned_by"`
	Capabilities struct {
		Audio           bool `json:"audio"`
		CompletionChat  bool `json:"completion_chat"`
		CompletionFim   bool `json:"completion_fim"`
		FunctionCalling bool `json:"function_calling"`
		FineTuning      bool `json:"fine_tuning"`
		Moderation      bool `json:"moderation"`
		OCR             bool `json:"ocr"`
		Vision          bool `json:"vision"`
		Classification  bool `json:"classification"`
	} `json:"capabilities"`
	Name                        string   `json:"name"`
	Description                 string   `json:"description"`
	MaxContextLength            int64    `json:"max_context_length"`
	Aliases                     []string `json:"aliases"`
	Deprecation                 string   `json:"deprecation"`
	DeprecationReplacementModel struct{} `json:"deprecation_replacement_model"`
	DefaultModelTemperature     float64  `json:"default_model_temperature"`
	Type                        string   `json:"type"` // "base"
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	var caps []string
	if m.Capabilities.CompletionChat {
		caps = append(caps, "chat")
	}
	if m.Capabilities.CompletionFim {
		caps = append(caps, "fim")
	}
	if m.Capabilities.FunctionCalling {
		caps = append(caps, "function")
	}
	if m.Capabilities.FineTuning {
		caps = append(caps, "fine-tuning")
	}
	if m.Capabilities.Vision {
		caps = append(caps, "vision")
	}
	suffix := ""
	if m.Deprecation != "" {
		suffix += " (deprecated)"
	}
	prefix := m.ID
	if m.ID != m.Name {
		prefix += " (" + m.Name + ")"
	}
	// Not including Created and Description because Created is not set and Description is not useful.
	return fmt.Sprintf("%s: %s Context: %d%s", prefix, strings.Join(caps, "/"), m.MaxContextLength, suffix)
}

func (m *Model) Context() int64 {
	return m.MaxContextLength
}

// ModelsResponse represents the response structure for Mistral models listing
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts Mistral models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	// As of 2025-08, Mistral returns duplicate models voxtral-mini-latest and voxtral-mini-2507. Filter them
	// out so the client is not confused.
	seen := make(map[string]struct{}, len(r.Data))
	models := make([]genai.Model, 0, len(r.Data))
	for i := range r.Data {
		id := r.Data[i].ID
		if _, ok := seen[id]; ok {
			continue
		}
		seen[id] = struct{}{}
		models = append(models, &r.Data[i])
	}
	return models
}

//

// ErrorResponse is the most goddam unstructured way to process errors. Basically what happens is that any
// point in the Mistral stack can return an error and each python library generates a different structure.
type ErrorResponse struct {
	// When simple issue like auth failure.
	// Message   string `json:"message"`
	RequestID string `json:"request_id"`

	// First error type
	Object string `json:"object"` // "error"
	// Message string `json:"message"`
	Type  string      `json:"type"`
	Param string      `json:"param"`
	Code  json.Number `json:"code"` // Sometimes a string, sometimes a int64.

	// Second error type
	Detail ErrorDetails `json:"detail"`

	// Third error type
	// Object  string `json:"object"` // error
	Message ErrorMessage `json:"message"`
	// Type  string `json:"type"`
	// Param string `json:"param"`
	// Code  int64  `json:"code"`
}

func (er *ErrorResponse) Error() string {
	out := er.Type
	if out != "" {
		out += ": "
	}
	if s := er.Detail.String(); s != "" {
		return out + s
	}
	return out + er.Message.Detail.String()
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// ErrorDetail can be either a struct or a string. When a string, it decodes into Msg.
type ErrorDetail struct {
	Type string `json:"type"` // "string_type", "missing"
	Msg  string `json:"msg"`
	Loc  []any  `json:"loc"` // to be joined, a mix of string and number
	// Input is either a list or an instance of struct { Type string `json:"type"` }.
	Input any    `json:"input"`
	Ctx   any    `json:"ctx"`
	URL   string `json:"url"`
}

func (ed *ErrorDetail) String() string {
	if ed.Type == "" && len(ed.Loc) == 0 {
		// This was actually a string
		return ed.Msg
	}
	return fmt.Sprintf("%s: %s at %s", ed.Type, ed.Msg, ed.Loc)
}

type ErrorDetails []ErrorDetail

func (ed *ErrorDetails) String() string {
	out := ""
	for _, e := range *ed {
		out += e.String()
	}
	return out
}

func (ed *ErrorDetails) UnmarshalJSON(b []byte) error {
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode((*[]ErrorDetail)(ed)); err == nil {
		return nil
	}
	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*ed = []ErrorDetail{{Msg: s}}
	return nil
}

type ErrorMessage struct {
	Detail ErrorDetails `json:"detail"`
}

func (er *ErrorMessage) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err == nil {
		er.Detail = ErrorDetails{{Msg: s}}
		return nil
	}
	var x struct {
		Detail ErrorDetails `json:"detail"`
	}
	d := json.NewDecoder(bytes.NewReader(b))
	if !internal.BeLenient {
		d.DisallowUnknownFields()
	}
	if err := d.Decode(&x); err != nil {
		return err
	}
	er.Detail = x.Detail
	return nil
}

// Client implements genai.Provider.
type Client struct {
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// TODO:
// https://codestral.mistral.ai/v1/fim/completions
// https://codestral.mistral.ai/v1/chat/completions

// New creates a new client to talk to the Mistral platform API.
//
// If opts.APIKey is not provided, it tries to load it from the MISTRAL_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://console.mistral.ai/api-keys or https://console.mistral.ai/codestral
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.mistral.ai/getting-started/models/models_overview/
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
//
// # PDF understanding
//
// PDF understanding requires a model which has the "OCR" or the "Document understanding" capability. There's
// a subtle difference between the two; from what I understand, the document understanding will only parse the
// text, while the OCR will try to understand the pictures.
//
// https://docs.mistral.ai/capabilities/document/
// https://docs.mistral.ai/capabilities/vision/
//
// # Tool use
//
// Tool use requires a model which has the tool capability. See
// https://docs.mistral.ai/capabilities/function_calling/
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://console.mistral.ai/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("MISTRAL_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "MISTRAL_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(opts.OutputModalities) != 0 && !slices.Equal(opts.OutputModalities, mod) {
		// https://docs.mistral.ai/agents/connectors/image_generation/
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is implemented (send PR to add support)", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:           "https://api.mistral.ai/v1/chat/completions",
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
			if c.impl.Model, err = c.selectBestTextModel(ctx, opts.Model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = opts.Model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
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
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// TODO: Support magistral.
		if !strings.HasSuffix(m.ID, "latest") || strings.HasPrefix(m.ID, "devstral") || strings.HasPrefix(m.ID, "magistral") || strings.HasPrefix(m.ID, "pixtral") {
			continue
		}
		// This is not great. To improve.
		if cheap {
			if strings.Contains(m.ID, "tiny") {
				selectedModel = m.ID
			}
		} else if good {
			if strings.Contains(m.ID, "medium") {
				selectedModel = m.ID
			}
		} else {
			if strings.Contains(m.ID, "large") {
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
	return "mistral"
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
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	return c.impl.GenStreamRaw(ctx, in, out)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.mistral.ai/api/#tag/models
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.mistral.ai/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Usage.PromptTokens != 0 {
			result.Usage.InputTokens = pkt.Usage.PromptTokens
			result.Usage.OutputTokens = pkt.Usage.CompletionTokens
			result.Usage.TotalTokens = pkt.Usage.TotalTokens
			result.Usage.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		f := genai.ReplyFragment{TextFragment: pkt.Choices[0].Delta.Content}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		// Mistral is one of the rare provider that can stream multiple tool calls all at once. It's probably
		// because it's buffering server-side.
		for i := range pkt.Choices[0].Delta.ToolCalls {
			f := genai.ReplyFragment{}
			pkt.Choices[0].Delta.ToolCalls[i].To(&f.ToolCall)
			if !f.IsZero() {
				if err := result.Accumulate(f); err != nil {
					return err
				}
				chunks <- f
			}
		}
	}
	return nil
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Req-10-Second"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Req-10-Second"), 10, 64)

	tokensPerMinLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens-Minute"), 10, 64)
	tokensPerMinRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens-Minute"), 10, 64)

	tokensPerMonthLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens-Month"), 10, 64)
	tokensPerMonthRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens-Month"), 10, 64)

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther, // 10 seconds is not a standard period
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(10 * time.Second).Round(10 * time.Millisecond),
		})
	}
	if tokensPerMinLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerMinute,
			Limit:     tokensPerMinLimit,
			Remaining: tokensPerMinRemaining,
			Reset:     time.Now().Add(time.Minute).Round(10 * time.Millisecond),
		})
	}
	if tokensPerMonthLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerMonth,
			Limit:     tokensPerMonthLimit,
			Remaining: tokensPerMonthRemaining,
			// This is not accurate, but there's no reset header.
			Reset: time.Now().Add(30 * 24 * time.Hour).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var (
	_ genai.Provider           = &Client{}
	_ scoreboard.ProviderScore = &Client{}
)
