// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openaichat implements a client for the OpenAI Chat Completion API.
//
// It is described at https://platform.openai.com/docs/api-reference/
package openaichat

// See official client at https://github.com/openai/openai-go

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
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
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for OpenAI.
//
// # Warnings
//
//   - Thinking is not returned.
//   - OpenAI supports more than what the client supports.
//   - Tool calling works very well but is biased; the model is lazy and when it's unsure, it will use the
//     tool's first argument.
//   - Rate limit is based on how much you spend per month: https://platform.openai.com/docs/guides/rate-limits
var Scoreboard = scoreboard.Score{
	Country:      "US",
	DashboardURL: "https://platform.openai.com/usage",
	Scenarios: []scoreboard.Scenario{
		{
			Models: []string{"gpt-4.1"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityDocument: {Inline: true, SupportedFormats: []string{"application/pdf"}},
				genai.ModalityText:     {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				IndecisiveTool:   scoreboard.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				Tools:            scoreboard.True,
				IndecisiveTool:   scoreboard.True,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
				TopLogprobs:      true,
			},
		},
		{
			Models: []string{"gpt-4o-audio-preview"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityAudio: {
					Inline:           true,
					SupportedFormats: []string{"audio/mp3", "audio/wav"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
			},
		},
		{
			Models:   []string{"o4-mini"},
			Thinking: true,
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityDocument: {Inline: true, SupportedFormats: []string{"application/pdf"}},
				genai.ModalityText:     {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				NoStopSequence:   true,
				NoMaxTokens:      true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
			},
			GenStream: &scoreboard.FunctionalityText{
				ReportRateLimits: true,
				NoStopSequence:   true,
				NoMaxTokens:      true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.Flaky,
				JSON:             true,
				JSONSchema:       true,
				Seed:             true,
			},
		},
		{
			Models: []string{"gpt-image-1"},
			In:     map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			Out: map[genai.Modality]scoreboard.ModalCapability{
				// TODO: Expose other supported image formats.
				genai.ModalityImage: {
					Inline:           true,
					SupportedFormats: []string{"image/jpeg"},
				},
			},
			GenDoc: &scoreboard.FunctionalityDoc{
				BrokenTokenUsage:   scoreboard.True,
				BrokenFinishReason: true,
				Seed:               true,
			},
		},
		{
			Models: []string{
				"o1",
				"o1-2024-12-17",
				"o1-mini",
				"o1-mini-2024-09-12",
				"o1-pro",
				"o1-pro-2025-03-19",
				"o3",
				"o3-2025-04-16",
				"o3-deep-research",
				"o3-deep-research-2025-06-26",
				"o3-mini",
				"o3-mini-2025-01-31",
				"o3-pro",
				"o3-pro-2025-06-10",
				"o4-mini-2025-04-16",
				"o4-mini-deep-research",
				"o4-mini-deep-research-2025-06-26",
			},
			Thinking: true,
		},
		{
			Models: []string{
				"chatgpt-4o-latest",
				"codex-mini-latest",
				"dall-e-2",
				"dall-e-3",
				"davinci-002",
				"gpt-3.5-turbo",
				"gpt-3.5-turbo-0125",
				"gpt-3.5-turbo-1106",
				"gpt-3.5-turbo-16k",
				"gpt-3.5-turbo-instruct",
				"gpt-3.5-turbo-instruct-0914",
				"gpt-4",
				"gpt-4-0125-preview",
				"gpt-4-0613",
				"gpt-4-1106-preview",
				"gpt-4-turbo",
				"gpt-4-turbo-2024-04-09",
				"gpt-4-turbo-preview",
				"gpt-4.1-2025-04-14",
				"gpt-4.1-mini",
				"gpt-4.1-mini-2025-04-14",
				"gpt-4.1-nano",
				"gpt-4.1-nano-2025-04-14",
				"gpt-4o",
				"gpt-4o-2024-05-13",
				"gpt-4o-2024-08-06",
				"gpt-4o-2024-11-20",
				"gpt-4o-audio-preview-2024-10-01",
				"gpt-4o-audio-preview-2024-12-17",
				"gpt-4o-audio-preview-2025-06-03",
				"gpt-4o-mini",
				"gpt-4o-mini-2024-07-18",
				"gpt-4o-mini-audio-preview", // This model fails the smoke test.
				"gpt-4o-mini-audio-preview-2024-12-17",
				"gpt-4o-mini-realtime-preview",
				"gpt-4o-mini-realtime-preview-2024-12-17",
				"gpt-4o-mini-search-preview",
				"gpt-4o-mini-search-preview-2025-03-11",
				"gpt-4o-mini-transcribe",
				"gpt-4o-mini-tts",
				"gpt-4o-realtime-preview",
				"gpt-4o-realtime-preview-2024-10-01",
				"gpt-4o-realtime-preview-2024-12-17",
				"gpt-4o-realtime-preview-2025-06-03",
				"gpt-4o-search-preview",
				"gpt-4o-search-preview-2025-03-11",
				"gpt-4o-transcribe",
				"gpt-5",
				"gpt-5-2025-08-07",
				"gpt-5-chat-latest",
				"gpt-5-mini",
				"gpt-5-mini-2025-08-07",
				"gpt-5-nano",
				"gpt-5-nano-2025-08-07",
				"omni-moderation-2024-09-26",
				"omni-moderation-latest",
				"text-embedding-3-large",
				"text-embedding-3-small",
				"text-embedding-ada-002",
				"tts-1",
				"tts-1-1106",
				"tts-1-hd",
				"tts-1-hd-1106",
				"whisper-1",
			},
		},
	},
}

// OptionsText includes OpenAI specific options.
type OptionsText struct {
	genai.OptionsText

	// ReasoningEffort is the amount of effort (number of tokens) the LLM can use to think about the answer.
	//
	// When unspecified, defaults to medium.
	ReasoningEffort ReasoningEffort
	// ServiceTier specify the priority.
	ServiceTier ServiceTier
}

// ServiceTier is the quality of service to determine the request's priority.
type ServiceTier string

const (
	// ServiceTierAuto will utilize scale tier credits until they are exhausted if the Project is Scale tier
	// enabled, else the request will be processed using the default service tier with a lower uptime SLA and no
	// latency guarantee.
	//
	// https://openai.com/api-scale-tier/
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierDefault has the request be processed using the default service tier with a lower uptime SLA
	// and no latency guarantee.
	ServiceTierDefault ServiceTier = "default"
	// ServiceTierFlex has the request be processed with the Flex Processing service tier.
	//
	// Flex processing is in beta, and currently only available for o3 and o4-mini models.
	//
	// https://platform.openai.com/docs/guides/flex-processing
	ServiceTierFlex ServiceTier = "flex"
)

// OptionsImage includes OpenAI specific options.
type OptionsImage struct {
	genai.OptionsImage

	// Background is only supported on gpt-image-1.
	Background Background
}

// ChatRequest is documented at https://platform.openai.com/docs/api-reference/chat/create
type ChatRequest struct {
	Model            string             `json:"model"`
	MaxTokens        int64              `json:"max_tokens,omitzero"` // Deprecated
	MaxChatTokens    int64              `json:"max_completion_tokens,omitzero"`
	Stream           bool               `json:"stream"`
	Messages         []Message          `json:"messages"`
	Seed             int64              `json:"seed,omitzero"`
	Temperature      float64            `json:"temperature,omitzero"` // [0, 2]
	Store            bool               `json:"store,omitzero"`
	ReasoningEffort  ReasoningEffort    `json:"reasoning_effort,omitzero"`
	Metadata         map[string]string  `json:"metadata,omitzero"`
	FrequencyPenalty float64            `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	LogitBias        map[string]float64 `json:"logit_bias,omitzero"`
	// See https://cookbook.openai.com/examples/using_logprobs
	Logprobs    bool     `json:"logprobs,omitzero"`
	TopLogprobs int64    `json:"top_logprobs,omitzero"` // [0, 20]
	N           int64    `json:"n,omitzero"`            // Number of choices
	Modalities  []string `json:"modalities,omitzero"`   // text, audio
	Prediction  struct {
		Type    string `json:"type,omitzero"` // "content"
		Content []struct {
			Type string `json:"type,omitzero"` // "text"
			Text string `json:"text,omitzero"`
		} `json:"content,omitzero"`
	} `json:"prediction,omitzero"`
	Audio struct {
		// https://platform.openai.com/docs/guides/text-to-speech#voice-options
		Voice string `json:"voice,omitzero"` // "alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"
		// https://platform.openai.com/docs/guides/text-to-speech#supported-output-formats
		Format string `json:"format,omitzero"` // "mp3", "wav", "flac", "opus", "pcm16", "aac"
	} `json:"audio,omitzero"`
	PresencePenalty float64 `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ResponseFormat  struct {
		Type       string `json:"type,omitzero"` // "text", "json_object", "json_schema"
		JSONSchema struct {
			Description string             `json:"description,omitzero"`
			Name        string             `json:"name,omitzero"`
			Schema      *jsonschema.Schema `json:"schema,omitzero"`
			Strict      bool               `json:"strict,omitzero"`
		} `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	ServiceTier   ServiceTier `json:"service_tier,omitzero"`
	Stop          []string    `json:"stop,omitzero"` // keywords to stop completion
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	TopP  float64 `json:"top_p,omitzero"` // [0, 1]
	Tools []Tool  `json:"tools,omitzero"`
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice        string `json:"tool_choice,omitzero"` // "none", "auto", "required"
	ParallelToolCalls bool   `json:"parallel_tool_calls,omitzero"`
	User              string `json:"user,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	if err := msgs.Validate(); err != nil {
		return err
	}
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return err
		}
		switch v := opts.(type) {
		case *OptionsText:
			c.ReasoningEffort = v.ReasoningEffort
			c.ServiceTier = v.ServiceTier
			unsupported = c.initOptions(&v.OptionsText, model)
			sp = v.SystemPrompt
		case *genai.OptionsText:
			c.ServiceTier = ServiceTierAuto
			unsupported = c.initOptions(v, model)
			sp = v.SystemPrompt
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}

	if sp != "" {
		// Starting with o1.
		c.Messages = append(c.Messages, Message{
			Role:    "developer",
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
	// If we have unsupported features but no other errors, return a continuable error
	if len(unsupported) > 0 && len(errs) == 0 {
		return &genai.UnsupportedContinuableError{Unsupported: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
	c.StreamOptions.IncludeUsage = stream
}

func (c *ChatRequest) initOptions(v *genai.OptionsText, model string) []string {
	var unsupported []string
	c.MaxChatTokens = v.MaxTokens
	// TODO: This is not great.
	if (strings.HasPrefix(model, "gpt-4o-") && strings.Contains(model, "-search")) ||
		model == "o1" ||
		strings.HasPrefix(model, "o1-") ||
		strings.HasPrefix(model, "o3-") ||
		strings.HasPrefix(model, "o4-") {
		if v.Temperature != 0 {
			unsupported = append(unsupported, "Temperature")
		}
	} else {
		c.Temperature = v.Temperature
	}
	c.TopP = v.TopP
	if strings.HasPrefix(model, "gpt-4o-") && strings.Contains(model, "-search") {
		if v.Seed != 0 {
			unsupported = append(unsupported, "Seed")
		}
	} else {
		c.Seed = v.Seed
	}
	if v.TopLogprobs > 0 {
		c.TopLogprobs = v.TopLogprobs
		c.Logprobs = true
	}
	if v.TopK != 0 {
		// Track this as an unsupported feature that can be ignored
		unsupported = append(unsupported, "TopK")
	}
	c.Stop = v.Stop
	if v.DecodeAs != nil {
		c.ResponseFormat.Type = "json_schema"
		// OpenAI requires a name.
		c.ResponseFormat.JSONSchema.Name = "response"
		c.ResponseFormat.JSONSchema.Strict = true
		c.ResponseFormat.JSONSchema.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
	} else if v.ReplyAsJSON {
		c.ResponseFormat.Type = "json_object"
	}
	if len(v.Tools) != 0 {
		// TODO: Determine exactly which models do not support this.
		if model != "o4-mini" {
			c.ParallelToolCalls = true
		}
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
		}
	}
	return unsupported
}

// ReasoningEffort is the effort the model should put into reasoning. Default is Medium.
//
// https://platform.openai.com/docs/api-reference/assistants/createAssistant#assistants-createassistant-reasoning_effort
// https://platform.openai.com/docs/guides/reasoning
type ReasoningEffort string

const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
)

// Message is documented at https://platform.openai.com/docs/api-reference/chat/create
type Message struct {
	Role    string   `json:"role,omitzero"` // "developer", "assistant", "user"
	Name    string   `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content Contents `json:"content,omitzero"`
	Refusal string   `json:"refusal,omitzero"` // The refusal message by the assistant.
	Audio   struct {
		ID string `json:"id,omitzero"`
	} `json:"audio,omitzero"`
	ToolCalls   []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID  string     `json:"tool_call_id,omitzero"` // TODO
	Annotations []struct{} `json:"annotations,omitzero"`
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
	m.Name = in.User
	if len(in.Requests) != 0 {
		m.Content = make([]Content, len(in.Requests))
		for i := range in.Requests {
			if err := m.Content[i].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request %d: %w", i, err)
			}
		}
	}
	if len(in.Replies) != 0 {
		for i := range in.Replies {
			if in.Replies[i].Thinking != "" {
				// Ignore thinking messages.
				continue
			}
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
					return fmt.Errorf("reply #%d: %w", i, err)
				}
				continue
			}
			c := Content{}
			if err := c.FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
			m.Content = append(m.Content, c)
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
// OpenAI replies with content as a string.
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
	*c = Contents{{Type: ContentText, Text: s}}
	return nil
}

type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL    string `json:"url,omitzero"`
		Detail string `json:"detail,omitzero"` // "auto", "low", "high"
	} `json:"image_url,omitzero"`

	// Type == "input_audio"
	InputAudio struct {
		Data []byte `json:"data,omitzero"`
		// https://platform.openai.com/docs/guides/speech-to-text
		Format string `json:"format,omitzero"` // "mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"
	} `json:"input_audio,omitzero"`

	// Type == "file"
	File struct {
		// Either FileID or both Filename and FileData.
		FileID   string `json:"file_id,omitzero"` // Use https://platform.openai.com/docs/api-reference/files
		Filename string `json:"filename,omitzero"`
		FileData string `json:"file_data,omitzero"`
	} `json:"file,omitzero"`
}

func (c *Content) FromRequest(in *genai.Request) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		// OpenAI require a mime-type to determine if image, sound or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case mimeType == "audio/mpeg":
			if in.Doc.URL != "" {
				return errors.New("URL to audio file not supported")
			}
			c.Type = ContentInputAudio
			c.InputAudio.Data = data
			c.InputAudio.Format = "mp3"
		case mimeType == "audio/wav":
			if in.Doc.URL != "" {
				return errors.New("URL to audio file not supported")
			}
			c.Type = ContentInputAudio
			c.InputAudio.Data = data
			c.InputAudio.Format = "wav"
		case strings.HasPrefix(mimeType, "text/plain"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			c.Text = string(data)
		default:
			if in.Doc.URL != "" {
				return fmt.Errorf("URL to %s file not supported", mimeType)
			}
			filename := in.Doc.GetFilename()
			if filename == "" {
				exts, err := mime.ExtensionsByType(mimeType)
				if err != nil {
					return err
				}
				if len(exts) == 0 {
					return fmt.Errorf("unknown extension for mime type %s", mimeType)
				}
				filename = "content" + exts[0]
			}
			c.Type = ContentFile
			c.File.Filename = filename
			c.File.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
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
		// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		// OpenAI require a mime-type to determine if image, sound or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case mimeType == "audio/mpeg":
			if in.Doc.URL != "" {
				return errors.New("URL to audio file not supported")
			}
			c.Type = ContentInputAudio
			c.InputAudio.Data = data
			c.InputAudio.Format = "mp3"
		case mimeType == "audio/wav":
			if in.Doc.URL != "" {
				return errors.New("URL to audio file not supported")
			}
			c.Type = ContentInputAudio
			c.InputAudio.Data = data
			c.InputAudio.Format = "wav"
		case strings.HasPrefix(mimeType, "text/plain"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			c.Text = string(data)
		default:
			if in.Doc.URL != "" {
				return fmt.Errorf("URL to %s file not supported", mimeType)
			}
			filename := in.Doc.GetFilename()
			if filename == "" {
				exts, err := mime.ExtensionsByType(mimeType)
				if err != nil {
					return err
				}
				if len(exts) == 0 {
					return fmt.Errorf("unknown extension for mime type %s", mimeType)
				}
				filename = "content" + exts[0]
			}
			c.Type = ContentFile
			c.File.Filename = filename
			c.File.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		}
		return nil
	}
	return errors.New("unknown Reply type")
}

func (c *Content) To(out *genai.Reply) error {
	switch c.Type {
	case ContentText:
		out.Text = c.Text
		if len(c.Text) == 0 {
			return errors.New("received empty text")
		}
	case ContentImageURL, ContentInputAudio, ContentRefusal, ContentAudio, ContentFile:
		fallthrough
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

type ContentType string

const (
	ContentText       ContentType = "text"
	ContentImageURL   ContentType = "image_url"
	ContentInputAudio ContentType = "input_audio"
	ContentRefusal    ContentType = "refusal"
	ContentAudio      ContentType = "audio"
	ContentFile       ContentType = "file"
)

type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
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

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Description string             `json:"description,omitzero"`
		Name        string             `json:"name,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
		Strict      bool               `json:"strict,omitzero"`
	} `json:"function,omitzero"`
}

// ChatResponse is documented at
// https://platform.openai.com/docs/api-reference/chat/object
type ChatResponse struct {
	Choices []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Created           base.Time `json:"created"`
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	Object            string    `json:"object"`
	Usage             Usage     `json:"usage"`
	ServiceTier       string    `json:"service_tier"`
	SystemFingerprint string    `json:"system_fingerprint"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			ReasoningTokens:   c.Usage.CompletionTokensDetails.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	out.Logprobs = c.Choices[0].Logprobs.To()
	return out, err
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
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
		AudioTokens  int64 `json:"audio_tokens"`
		TextTokens   int64 `json:"text_tokens"`
		ImageTokens  int64 `json:"image_tokens"`
	} `json:"prompt_tokens_details"`
	CompletionTokensDetails struct {
		ReasoningTokens          int64 `json:"reasoning_tokens"`
		AudioTokens              int64 `json:"audio_tokens"`
		AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
		RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
		TextTokens               int64 `json:"text_tokens"`
	} `json:"completion_tokens_details"`
}

type Logprobs struct {
	Content []struct {
		Token       string  `json:"token"`
		Bytes       []byte  `json:"bytes"`
		Logprob     float64 `json:"logprob"`
		TopLogprobs []struct {
			Token   string  `json:"token"`
			Bytes   []byte  `json:"bytes"`
			Logprob float64 `json:"logprob"`
		} `json:"top_logprobs"`
	} `json:"content"`
	Refusal string `json:"refusal"`
}

func (l *Logprobs) To() []genai.Logprobs {
	if len(l.Content) == 0 {
		return nil
	}
	out := make([]genai.Logprobs, 0, len(l.Content))
	for i, lp := range l.Content {
		out = append(out, genai.Logprobs{Text: lp.Token, Bytes: lp.Bytes, Logprob: lp.Logprob, TopLogprobs: make([]genai.TopLogprob, 0, len(lp.TopLogprobs))})
		for _, tlp := range lp.TopLogprobs {
			out[i].TopLogprobs = append(out[i].TopLogprobs, genai.TopLogprob{Text: tlp.Token, Bytes: tlp.Bytes, Logprob: tlp.Logprob})
		}
	}
	// TODO: What happens if l.Refusal is not empty?
	return out
}

// ChatStreamChunkResponse is not documented?
type ChatStreamChunkResponse struct {
	Choices []struct {
		Delta struct {
			Content   string     `json:"content"`
			Role      string     `json:"role"`
			Refusal   string     `json:"refusal"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason FinishReason `json:"finish_reason"`
		Index        int64        `json:"index"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Created           base.Time `json:"created"`
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	Object            string    `json:"object"` // "chat.completion.chunk"
	ServiceTier       string    `json:"service_tier"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Usage             Usage     `json:"usage"`
	Obfuscation       string    `json:"obfuscation"`
}

//

// ImageRequest is documented at https://platform.openai.com/docs/api-reference/images
type ImageRequest struct {
	Prompt            string     `json:"prompt"`
	Model             string     `json:"model,omitzero"`              // Default to dall-e-2, unless a gpt-image-1 specific parameter is used.
	Background        Background `json:"background,omitzero"`         // Default "auto"
	Moderation        string     `json:"moderation,omitzero"`         // gpt-image-1: "low" or "auto"
	N                 int64      `json:"n,omitzero"`                  // Number of images to return
	OutputCompression float64    `json:"output_compression,omitzero"` // Defaults to 100. Only supported on gpt-image-1 with webp or jpeg
	OutputFormat      string     `json:"output_format,omitzero"`      // "png", "jpeg" or "webp". Defaults to png. Only supported on gpt-image-1.
	Quality           string     `json:"quality,omitzero"`            // "auto", gpt-image-1: "high", "medium", "low". dall-e-3: "hd", "standard". dall-e-2: "standard".
	ResponseFormat    string     `json:"response_format,omitzero"`    // "url" or "b64_json"; url is valid for 60 minutes; gpt-image-1 only returns b64_json
	Size              string     `json:"size,omitzero"`               // "auto", gpt-image-1: "1024x1024", "1536x1024", "1024x1536". dall-e-3: "1024x1024", "1792x1024", "1024x1792". dall-e-2: "256x256", "512x512", "1024x1024".
	Style             string     `json:"style,omitzero"`              // dall-e-3: "vivid", "natural"
	User              string     `json:"user,omitzero"`               // End-user to help monitor and detect abuse
}

func (i *ImageRequest) Init(msg genai.Message, opts genai.Options, model string) error {
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

	// This is unfortunate.
	switch model {
	case "gpt-image-1":
		i.Moderation = "low"
		// req.Background = "transparent"
		// req.OutputFormat = "webp"
		// req.OutputCompression = 90
		// req.Quality = "high"
		// req.Size = "1536x1024"
	case "dall-e-3":
		// req.Size = "1792x1024"
		i.ResponseFormat = "b64_json"
	case "dall-e-2":
		// We assume dall-e-2 is only used for smoke testing, so use the smallest image.
		i.Size = "256x256"
		// Maximum prompt length is 1000 characters.
		// Since we assume this is only for testing, silently cut it off.
		if len(i.Prompt) > 1000 {
			i.Prompt = i.Prompt[:1000]
		}
		i.ResponseFormat = "b64_json"
	default:
		// Silently pass.
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return err
		}
		switch v := opts.(type) {
		case *OptionsImage:
			if v.Height != 0 && v.Width != 0 {
				i.Size = fmt.Sprintf("%dx%d", v.Width, v.Height)
			}
			i.Background = v.Background
		case *genai.OptionsImage:
			if v.Height != 0 && v.Width != 0 {
				i.Size = fmt.Sprintf("%dx%d", v.Width, v.Height)
			}
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
	}
	return nil
}

// Background is only supported on gpt-image-1.
type Background string

const (
	BackgroundAuto        Background = "auto"
	BackgroundTransparent Background = "transparent"
	BackgroundOpaque      Background = "opaque"
)

type ImageResponse struct {
	Created base.Time         `json:"created"`
	Data    []ImageChoiceData `json:"data"`
	Usage   struct {
		InputTokens        int64 `json:"input_tokens"`
		OutputTokens       int64 `json:"output_tokens"`
		TotalTokens        int64 `json:"total_tokens"`
		InputTokensDetails struct {
			TextTokens  int64 `json:"text_tokens"`
			ImageTokens int64 `json:"image_tokens"`
		} `json:"input_tokens_details"`
	} `json:"usage"`
	Background   string `json:"background"`    // "opaque"
	Size         string `json:"size"`          // e.g. "1024x1024"
	Quality      string `json:"quality"`       // e.g. "medium"
	OutputFormat string `json:"output_format"` // e.g. "png"
}

type ImageChoiceData struct {
	B64JSON       []byte `json:"b64_json"`
	RevisedPrompt string `json:"revised_prompt"` // dall-e-3 only
	URL           string `json:"url"`            // Unsupported for gpt-image-1
}

//

// File is documented at https://platform.openai.com/docs/api-reference/files/object
type File struct {
	Bytes         int64     `json:"bytes"` // File size
	CreatedAt     base.Time `json:"created_at"`
	ExpiresAt     base.Time `json:"expires_at"`
	Filename      string    `json:"filename"`
	ID            string    `json:"id"`
	Object        string    `json:"object"`         // "file"
	Purpose       string    `json:"purpose"`        // One of: assistants, assistants_output, batch, batch_output, fine-tune, fine-tune-results and vision
	Status        string    `json:"status"`         // Deprecated
	StatusDetails string    `json:"status_details"` // Deprecated
}

func (f *File) GetID() string {
	return f.ID
}

func (f *File) GetDisplayName() string {
	return f.Filename
}

func (f *File) GetExpiry() time.Time {
	return f.ExpiresAt.AsTime()
}

// FileDeleteResponse is documented at https://platform.openai.com/docs/api-reference/files/delete
type FileDeleteResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "file"
	Deleted bool   `json:"deleted"`
}

// FileListResponse is documented at https://platform.openai.com/docs/api-reference/files/list
type FileListResponse struct {
	Data   []File `json:"data"`
	Object string `json:"object"` // "list"
}

//

// BatchRequestInput is documented at https://platform.openai.com/docs/api-reference/batch/request-input
type BatchRequestInput struct {
	CustomID string      `json:"custom_id"`
	Method   string      `json:"method"` // "POST"
	URL      string      `json:"url"`    // "/v1/chat/completions", "/v1/embeddings", "/v1/completions", "/v1/responses"
	Body     ChatRequest `json:"body"`
}

// BatchRequestOutput is documented at https://platform.openai.com/docs/api-reference/batch/request-output
type BatchRequestOutput struct {
	CustomID string `json:"custom_id"`
	ID       string `json:"id"`
	Error    struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"error"`
	Response struct {
		StatusCode int          `json:"status_code"`
		RequestID  string       `json:"request_id"` // To use when contacting support
		Body       ChatResponse `json:"body"`
	} `json:"response"`
}

// BatchRequest is documented at https://platform.openai.com/docs/api-reference/batch/create
type BatchRequest struct {
	CompletionWindow string            `json:"completion_window"` // Must be "24h"
	Endpoint         string            `json:"endpoint"`          // One of /v1/responses, /v1/chat/completions, /v1/embeddings, /v1/completions
	InputFileID      string            `json:"input_file_id"`     // File must be JSONL
	Metadata         map[string]string `json:"metadata,omitzero"` // Maximum 16 keys of 64 chars, values max 512 chars
}

// Batch is documented at https://platform.openai.com/docs/api-reference/batch/object
type Batch struct {
	CancelledAt      base.Time `json:"cancelled_at"`
	CancellingAt     base.Time `json:"cancelling_at"`
	CompletedAt      base.Time `json:"completed_at"`
	CompletionWindow string    `json:"completion_window"` // "24h"
	CreatedAt        base.Time `json:"created_at"`
	Endpoint         string    `json:"endpoint"`      // Same as BatchRequest.Endpoint
	ErrorFileID      string    `json:"error_file_id"` // File ID containing the outputs of requests with errors.
	Errors           struct {
		Data []struct {
			Code    string `json:"code"`
			Line    int64  `json:"line"`
			Message string `json:"message"`
			Param   string `json:"param"`
		} `json:"data"`
	} `json:"errors"`
	ExpiredAt     base.Time         `json:"expired_at"`
	ExpiresAt     base.Time         `json:"expires_at"`
	FailedAt      base.Time         `json:"failed_at"`
	FinalizingAt  base.Time         `json:"finalizing_at"`
	ID            string            `json:"id"`
	InProgressAt  base.Time         `json:"in_progress_at"`
	InputFileID   string            `json:"input_file_id"` // Input data
	Metadata      map[string]string `json:"metadata"`
	Object        string            `json:"object"`         // "batch"
	OutputFileID  string            `json:"output_file_id"` // Output data
	RequestCounts struct {
		Completed int64 `json:"completed"`
		Failed    int64 `json:"failed"`
		Total     int64 `json:"total"`
	} `json:"request_counts"`
	Status string `json:"status"` // "completed", "in_progress", "validating", "finalizing"
}

//

// Model is documented at https://platform.openai.com/docs/api-reference/models/object
//
// Sadly the modalities aren't reported. The only way I can think of to find it at run time is to fetch
// https://platform.openai.com/docs/models/gpt-4o-mini-realtime-preview, find the div containing
// "Modalities:", then extract the modalities from the text
type Model struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created base.Time `json:"created"`
	OwnedBy string    `json:"owned_by"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s)", m.ID, m.Created.AsTime().Format("2006-01-02"))
}

func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for OpenAI models listing
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts OpenAI models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

//

type ErrorResponse struct {
	ErrorVal ErrorResponseError `json:"error"`
}

func (er *ErrorResponse) Error() string {
	out := ""
	if er.ErrorVal.Type != "" {
		out += er.ErrorVal.Type
	}
	if er.ErrorVal.Code != "" {
		if out != "" {
			out += "/"
		}
		out += er.ErrorVal.Code
	}
	if er.ErrorVal.Status != "" {
		out += fmt.Sprintf("(%s)", er.ErrorVal.Status)
	}
	if er.ErrorVal.Param != "" {
		if out != "" {
			out += " "
		}
		out += fmt.Sprintf("for %q", er.ErrorVal.Param)
	}
	if er.ErrorVal.Message != "" {
		if out != "" {
			out += ": "
		}
		out += er.ErrorVal.Message
	}
	return out
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

type ErrorResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
	Type    string `json:"type"`
	Param   string `json:"param"`
}

//

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	impl base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// In May 2025, OpenAI started pushing for Response API. They say it's the only way to keep reasoning items.
// It's interesting because Anthropic did that with the old API but OpenAI can't. Shrug.
// https://cookbook.openai.com/examples/responses_api/reasoning_items
// https://platform.openai.com/docs/api-reference/responses/create
// TODO: Switch over.

// New creates a new client to talk to the OpenAI platform API.
//
// If opts.APIKey is not provided, it tries to load it from the OPENAI_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://platform.openai.com/settings/organization/api-keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://platform.openai.com/docs/models
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
//
// # Documents
//
// OpenAI supports many types of documents, listed at
// https://platform.openai.com/docs/assistants/tools/file-search#supported-files
func New(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "OPENAI_API_KEY", URL: apiKeyURL}
		}
	}
	model := opts.Model
	if model == "" {
		model = genai.ModelGood
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.openai.com/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			ProcessHeaders:       processHeaders,
			Provider: base.Provider[*ErrorResponse]{
				// OpenAI error message prints the api key URL already.
				APIKeyURL: "",
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
	switch model {
	case genai.ModelNone:
		c.impl.Model = ""
	case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA:
		if err == nil {
			if c.impl.Model, err = c.selectBestModel(context.Background(), model); err != nil {
				return nil, err
			}
		}
	}
	return c, err
}

// selectBestModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
func (c *Client) selectBestModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", err
	}

	cheap := preference == genai.ModelCheap
	good := preference == genai.ModelGood
	selectedModel := ""
	var created base.Time
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if cheap {
			if strings.HasSuffix(m.ID, "-nano") && (created == 0 || m.Created < created) {
				// For the cheapest, we want the oldest model as it is generally cheaper.
				created = m.Created
				selectedModel = m.ID
			}
		} else if good {
			if strings.HasSuffix(m.ID, "-mini") && (created == 0 || m.Created > created) {
				// For the greatest, we want the newest model as it is generally better.
				created = m.Created
				selectedModel = m.ID
			}
		} else {
			if strings.HasSuffix(m.ID, "-pro") && (created == 0 || m.Created > created) {
				// For the greatest, we want the newest model as it is generally better.
				created = m.Created
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
	return "openai"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// Scoreboard implements scoreboard.ProviderScore.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard
}

// GenSync implements genai.ProviderGen.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	if c.isImage(opts) {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.GenDoc(ctx, msgs[0], opts)
	}
	return c.impl.GenSync(ctx, msgs, opts)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.ProviderGen.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	if c.isImage(opts) {
		return base.SimulateStream(ctx, c, msgs, chunks, opts)
	}
	return c.impl.GenStream(ctx, msgs, chunks, opts)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	return c.impl.GenStreamRaw(ctx, in, out)
}

// GenDoc implements genai.ProviderGenDoc.
//
// It synchronously generates a document.
func (c *Client) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	// https://platform.openai.com/docs/api-reference/images/create
	res := genai.Result{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	req := ImageRequest{}
	if err := req.Init(msg, opts, c.impl.Model); err != nil {
		return res, err
	}
	url := "https://api.openai.com/v1/images/generations"

	// It is very different because it requires a multi-part upload.
	// https://platform.openai.com/docs/api-reference/images/createEdit
	// url = "https://api.openai.com/v1/images/edits"

	resp := ImageResponse{}
	if err := c.impl.DoRequest(ctx, "POST", url, &req, &resp); err != nil {
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

// ListModels implements genai.ProviderModel.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://platform.openai.com/docs/api-reference/models/list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.openai.com/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

func (c *Client) isImage(opts genai.Options) bool {
	switch c.impl.Model {
	// TODO: Use Scoreboard list.
	case "dall-e-2", "dall-e-3", "gpt-image-1":
		return true
	default:
		return opts != nil && slices.Contains(opts.Modalities(), genai.ModalityImage)
	}
}

// GenAsync implements genai.ProviderGenAsync.
//
// It requests the providers' batch API and returns the job ID. It can take up to 24 hours to complete.
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Job, error) {
	fileID, err := c.CacheAddRequest(ctx, msgs, opts, "TODO", "batch.json", 24*time.Hour)
	if err != nil {
		return "", err
	}
	b := BatchRequest{CompletionWindow: "24h", Endpoint: "/v1/chat/completions", InputFileID: fileID}
	resp, err := c.GenAsyncRaw(ctx, b)
	if len(resp.Errors.Data) != 0 {
		errs := []error{err}
		for _, d := range resp.Errors.Data {
			errs = append(errs, fmt.Errorf("batch error on line %d: %s (%s)", d.Line, d.Message, d.Code))
		}
		err = errors.Join(errs...)
	}
	return genai.Job(resp.ID), err
}

func (c *Client) GenAsyncRaw(ctx context.Context, b BatchRequest) (Batch, error) {
	resp := Batch{}
	err := c.impl.DoRequest(ctx, "POST", "https://api.openai.com/v1/batches", &b, &resp)
	return resp, err
}

// PokeResult implements genai.ProviderGenAsync.
//
// It retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	resp, err := c.PokeResultRaw(ctx, id)
	if len(resp.Errors.Data) != 0 {
		errs := []error{err}
		for _, d := range resp.Errors.Data {
			errs = append(errs, fmt.Errorf("batch error on line %d: %s (%s)", d.Line, d.Message, d.Code))
		}
		err = errors.Join(errs...)
	}
	if resp.Status == "validating" || resp.Status == "in_progress" || resp.Status == "finalizing" {
		res.Usage.FinishReason = genai.Pending
	}
	if resp.OutputFileID != "" {
		f, err2 := c.FileGet(ctx, resp.OutputFileID)
		if err == nil {
			err = err2
		}
		if f != nil {
			defer f.Close()
			out := BatchRequestOutput{}
			d := json.NewDecoder(f)
			d.UseNumber()
			if !c.impl.ClientJSON.Lenient {
				d.DisallowUnknownFields()
			}
			if err = d.Decode(&out); err != nil {
				return res, err
			}
			res, err2 = out.Response.Body.ToResult()
			if err2 == nil && out.Error.Message != "" {
				err2 = fmt.Errorf("error %s: %s", out.Error.Code, out.Error.Message)
			}
		}
		if err == nil {
			err = err2
		}
	}
	// TODO: Delete the input and output files.
	return res, err
}

func (c *Client) PokeResultRaw(ctx context.Context, id genai.Job) (Batch, error) {
	out := Batch{}
	u := "https://api.openai.com/v1/batches/" + url.PathEscape(string(id))
	err := c.impl.DoRequest(ctx, "GET", u, nil, &out)
	return out, err
}

// Cancel cancels an in-progress batch. The batch will be in status cancelling for up to 10 minutes, before
// changing to cancelled, where it will have partial results (if any) available in the output file.
func (c *Client) Cancel(ctx context.Context, id genai.Job) error {
	_, err := c.CancelRaw(ctx, id)
	return err
}

func (c *Client) CancelRaw(ctx context.Context, id genai.Job) (Batch, error) {
	u := "https://api.openai.com/v1/batches/" + url.PathEscape(string(id)) + "/cancel"
	resp := Batch{}
	err := c.impl.DoRequest(ctx, "POST", u, nil, &resp)
	// TODO: Delete the file too.
	return resp, err
}

func (c *Client) CacheAddRequest(ctx context.Context, msgs genai.Messages, opts genai.Options, name, displayName string, ttl time.Duration) (string, error) {
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	// Upload the messages and options as a file.
	b := BatchRequestInput{CustomID: name, Method: "POST", URL: "/v1/chat/completions"}
	if err := b.Body.Init(msgs, opts, c.impl.Model); err != nil {
		return "", err
	}
	raw, err := json.Marshal(b)
	if err != nil {
		return "", err
	}
	return c.FileAdd(ctx, displayName, bytes.NewReader(raw))
}

func (c *Client) CacheList(ctx context.Context) ([]genai.CacheEntry, error) {
	l, err := c.FilesListRaw(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]genai.CacheEntry, len(l))
	for i := range l {
		out[i] = &l[i]
	}
	return out, nil
}

func (c *Client) CacheDelete(ctx context.Context, name string) error {
	return c.FileDel(ctx, name)
}

// FileAdd uploads a file. The TTL is one month.
func (c *Client) FileAdd(ctx context.Context, filename string, r io.ReadSeeker) (string, error) {
	// https://platform.openai.com/docs/api-reference/files/create
	buf := bytes.Buffer{}
	w := multipart.NewWriter(&buf)
	// We don't need this to be random, and setting it to be deterministic makes HTTP playback possible.
	_ = w.SetBoundary("80309819a837f26826233a299e185d0ccf3f559362092bd3278b8a045ee1")
	if err := w.WriteField("purpose", "batch"); err != nil {
		return "", err
	}
	part, err := w.CreateFormFile("file", filename)
	if err != nil {
		return "", err
	}
	if _, err = io.Copy(part, r); err != nil {
		return "", err
	}
	if err = w.Close(); err != nil {
		return "", err
	}
	u := "https://api.openai.com/v1/files"
	req, err := http.NewRequestWithContext(ctx, "POST", u, &buf)
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", w.FormDataContentType())
	resp, err := c.impl.ClientJSON.Client.Do(req)
	if err != nil {
		return "", err
	}
	f := File{}
	err = c.impl.DecodeResponse(resp, u, &f)
	return f.ID, err
}

func (c *Client) FileGet(ctx context.Context, id string) (io.ReadCloser, error) {
	// https://platform.openai.com/docs/api-reference/files/retrieve-contents
	u := "https://api.openai.com/v1/files/" + url.PathEscape(string(id)) + "/content"
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.impl.ClientJSON.Client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, c.impl.DecodeError(u, resp)
	}
	return resp.Body, nil
}

func (c *Client) FileDel(ctx context.Context, id string) error {
	// https://platform.openai.com/docs/api-reference/files/delete
	url := "https://api.openai.com/v1/files/" + url.PathEscape(string(id))
	out := FileDeleteResponse{}
	return c.impl.DoRequest(ctx, "DELETE", url, nil, &out)
}

func (c *Client) FilesListRaw(ctx context.Context) ([]File, error) {
	// TODO: Pagination. It defaults at 10000 items per page.
	resp := FileListResponse{}
	err := c.impl.DoRequest(ctx, "GET", "https://api.openai.com/v1/files", nil, &resp)
	return resp.Data, err
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		if pkt.Usage.PromptTokens != 0 {
			result.Usage.InputTokens = pkt.Usage.PromptTokens
			result.Usage.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
			result.Usage.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
			result.Usage.OutputTokens = pkt.Usage.CompletionTokens
		}
		if len(pkt.Choices) != 1 {
			continue
		}
		if fr := pkt.Choices[0].FinishReason; fr != "" {
			result.Usage.FinishReason = fr.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt)
		}
		if r := pkt.Choices[0].Delta.Refusal; r != "" {
			return fmt.Errorf("refused: %q", r)
		}
		f := genai.ReplyFragment{TextFragment: pkt.Choices[0].Delta.Content}
		// OpenAI streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
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
		result.Logprobs = append(result.Logprobs, pkt.Choices[0].Logprobs.To()...)
	}
	return nil
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests"), 10, 64)
	requestsReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Requests"))

	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)
	tokensReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Tokens"))

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(requestsReset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(tokensReset).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderCache      = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderGenDoc     = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ scoreboard.ProviderScore = &Client{}
)
