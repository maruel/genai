// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openai implements a client for the OpenAI API.
//
// It is described at https://platform.openai.com/docs/api-reference/
package openai

// See official client at https://github.com/openai/openai-go

// TODO: Investigate https://platform.openai.com/docs/api-reference/responses/create

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

// Scoreboard for OpenAI.
//
// # Warnings
//
//   - Thinking is not returned.
//   - OpenAI supports more than what the client supports.
//   - Tool calling works very well but is biased; the model is lazy and when it's unsure, it will use the
//     tool's first argument.
//   - Rate limit is based on how much you spend per month: https://platform.openai.com/docs/guides/rate-limits
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"gpt-4.1-nano",
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
				"gpt-4.1",
				"gpt-4.1-2025-04-14",
				"gpt-4.1-mini",
				"gpt-4.1-mini-2025-04-14",
				"gpt-4.1-nano-2025-04-14",
				"gpt-4.5-preview",
				"gpt-4.5-preview-2025-02-27",
				"gpt-4o",
				"gpt-4o-2024-05-13",
				"gpt-4o-2024-08-06",
				"gpt-4o-2024-11-20",
				"gpt-4o-mini",
				"gpt-4o-mini-2024-07-18",
				"gpt-4o-mini-realtime-preview",
				"gpt-4o-mini-realtime-preview-2024-12-17",
				"gpt-4o-mini-search-preview",
				"gpt-4o-mini-search-preview-2025-03-11",
				"gpt-4o-realtime-preview",
				"gpt-4o-realtime-preview-2024-10-01",
				"gpt-4o-realtime-preview-2024-12-17",
				"gpt-4o-search-preview",
				"gpt-4o-search-preview-2025-03-11",
			},
			GenSync: &genai.FunctionalityText{
				Tools:          genai.True,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
			GenStream: &genai.FunctionalityText{
				Tools:          genai.True,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityAudio, genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"gpt-4o-audio-preview",
				"gpt-4o-audio-preview-2024-10-01",
				"gpt-4o-audio-preview-2024-12-17",
				"gpt-4o-mini-transcribe",
				"gpt-4o-transcribe",

				// Supports audio output
				"gpt-4o-mini-audio-preview",
				"gpt-4o-mini-audio-preview-2024-12-17",
			},
			GenSync: &genai.FunctionalityText{
				InputInline: true,
				Tools:       genai.True,
				BiasedTool:  genai.True,
				JSON:        true,
				JSONSchema:  true,
			},
			GenStream: &genai.FunctionalityText{
				InputInline: true,
				Tools:       genai.True,
				BiasedTool:  genai.True,
				JSON:        true,
				JSONSchema:  true,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityImage, genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"o4-mini",
				"o1",
				"o1-2024-12-17",
				"o1-mini",
				"o1-mini-2024-09-12",
				"o1-preview",
				"o1-preview-2024-09-12",
				"o1-pro",
				"o1-pro-2025-03-19",
				"o3-mini",
				"o3-mini-2025-01-31",
				"o4-mini-2025-04-16",
			},
			GenSync: &genai.FunctionalityText{
				InputInline:    true,
				InputURL:       true,
				NoStopSequence: true,
				Tools:          genai.True,
				BiasedTool:     genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
			GenStream: &genai.FunctionalityText{
				InputInline:    true,
				InputURL:       true,
				NoStopSequence: true,
				Tools:          genai.True,
				BiasedTool:     genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityImage},
			Models: []string{
				"dall-e-2",
				"dall-e-3",
				"gpt-image-1",
			},
			GenDoc: &genai.FunctionalityDoc{
				OutputInline:       true,
				BrokenTokenUsage:   true,
				BrokenFinishReason: true,
			},
		},
		// Audio only output:
		// "gpt-4o-mini-tts",
		// "tts-1",
		// "tts-1-1106",
		// "tts-1-hd",
		// "tts-1-hd-1106",
		// Audio only input:
		// "whisper-1",
		// Image only output
		// "gpt-image-1",
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

// https://platform.openai.com/docs/api-reference/chat/create
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
	sp := ""
	if opts != nil {
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

	offset := 0
	if sp != "" {
		offset = 1
	}
	c.Messages = make([]Message, len(msgs)+offset)
	if sp != "" {
		// Starting with o1.
		c.Messages[0].Role = "developer"
		c.Messages[0].Content = Contents{{Type: "text", Text: sp}}
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
		c.ResponseFormat.JSONSchema.Schema = jsonschema.Reflect(v.DecodeAs)
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

// https://platform.openai.com/docs/api-reference/chat/create
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

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	m.Name = in.User
	if len(in.Contents) != 0 {
		m.Content = make([]Content, len(in.Contents))
		for i := range in.Contents {
			if err := m.Content[i].From(&in.Contents[i]); err != nil {
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
	switch m.Role {
	case "assistant", "user":
		out.Role = genai.Role(m.Role)
	default:
		// case "developer":
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	if len(m.Content) != 0 {
		out.Contents = make([]genai.Content, len(m.Content))
		for i := range m.Content {
			if err := m.Content[i].To(&out.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
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

// OpenAI replies with content as a string.
func (c *Contents) UnmarshalJSON(data []byte) error {
	var v []Content
	if err := json.Unmarshal(data, &v); err != nil {
		s := ""
		if err = json.Unmarshal(data, &s); err != nil {
			return err
		}
		*c = []Content{{Type: ContentText, Text: s}}
		return nil
	}
	*c = Contents(v)
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

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	// OpenAI require a mime-type to determine if image, sound or PDF.
	if mimeType == "" {
		return fmt.Errorf("unspecified mime type for URL %q", in.URL)
	}
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		c.Type = ContentImageURL
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	case mimeType == "audio/mpeg":
		if in.URL != "" {
			return errors.New("URL to audio file not supported")
		}
		c.Type = ContentInputAudio
		c.InputAudio.Data = data
		c.InputAudio.Format = "mp3"
	case mimeType == "audio/wav":
		if in.URL != "" {
			return errors.New("URL to audio file not supported")
		}
		c.Type = ContentInputAudio
		c.InputAudio.Data = data
		c.InputAudio.Format = "wav"
	default:
		if in.URL != "" {
			return fmt.Errorf("URL to %s file not supported", mimeType)
		}
		filename := in.GetFilename()
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

func (t *ToolCall) From(in *genai.ToolCall) {
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
	Created           Time   `json:"created"`
	ID                string `json:"id"`
	Model             string `json:"model"`
	Object            string `json:"object"`
	Usage             Usage  `json:"usage"`
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
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
		Logprob     float64 `json:"logprob"`
		Bytes       []int   `json:"bytes"`
		TopLogprobs []struct {
			TODO string `json:"todo"`
		} `json:"top_logprobs"`
	} `json:"content"`
	Refusal string `json:"refusal"`
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
	Created           Time   `json:"created"`
	ID                string `json:"id"`
	Model             string `json:"model"`
	Object            string `json:"object"` // "chat.completion.chunk"
	ServiceTier       string `json:"service_tier"`
	SystemFingerprint string `json:"system_fingerprint"`
	Usage             Usage  `json:"usage"`
}

// Time is a JSON encoded unix timestamp.
type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

//

// https://platform.openai.com/docs/api-reference/images
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

// Background is only supported on gpt-image-1.
type Background string

const (
	BackgroundAuto        Background = "auto"
	BackgroundTransparent Background = "transparent"
	BackgroundOpaque      Background = "opaque"
)

type ImageResponse struct {
	Created Time              `json:"created"`
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
}

type ImageChoiceData struct {
	B64JSON       []byte `json:"b64_json"`
	RevisedPrompt string `json:"revised_prompt"` // dall-e-3 only
	URL           string `json:"url"`            // Unsupported for gpt-image-1
}

//

// https://platform.openai.com/docs/api-reference/files/object
type File struct {
	Bytes         int64  `json:"bytes"` // File size
	CreatedAt     Time   `json:"created_at"`
	ExpiresAt     Time   `json:"expires_at"`
	Filename      string `json:"filename"`
	ID            string `json:"id"`
	Object        string `json:"object"`         // "file"
	Purpose       string `json:"purpose"`        // One of: assistants, assistants_output, batch, batch_output, fine-tune, fine-tune-results and vision
	Status        string `json:"status"`         // Deprecated
	StatusDetails string `json:"status_details"` // Deprecated
}

// https://platform.openai.com/docs/api-reference/files/delete
type FileDeleteResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "file"
	Deleted bool   `json:"deleted"`
}

// https://platform.openai.com/docs/api-reference/files/list
type FileListResponse struct {
	Data   []File `json:"data"`
	Object string `json:"object"` // "list"
}

//

// https://platform.openai.com/docs/api-reference/batch/request-input
type BatchRequestInput struct {
	CustomID string      `json:"custom_id"`
	Method   string      `json:"method"` // "POST"
	URL      string      `json:"url"`    // "/v1/chat/completions", "/v1/embeddings", "/v1/completions", "/v1/responses"
	Body     ChatRequest `json:"body"`
}

// https://platform.openai.com/docs/api-reference/batch/request-output
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

// https://platform.openai.com/docs/api-reference/batch/create
type BatchRequest struct {
	CompletionWindow string            `json:"completion_window"` // Must be "24h"
	Endpoint         string            `json:"endpoint"`          // One of /v1/responses, /v1/chat/completions, /v1/embeddings, /v1/completions
	InputFileID      string            `json:"input_file_id"`     // File must be JSONL
	Metadata         map[string]string `json:"metadata,omitzero"` // Maximum 16 keys of 64 chars, values max 512 chars
}

// https://platform.openai.com/docs/api-reference/batch/object
type Batch struct {
	CancelledAt      Time   `json:"cancelled_at"`
	CancellingAt     Time   `json:"cancelling_at"`
	CompletedAt      Time   `json:"completed_at"`
	CompletionWindow string `json:"completion_window"` // "24h"
	CreatedAt        Time   `json:"created_at"`
	Endpoint         string `json:"endpoint"`      // Same as BatchRequest.Endpoint
	ErrorFileID      string `json:"error_file_id"` // File ID containing the outputs of requests with errors.
	Errors           struct {
		Data []struct {
			Code    string `json:"code"`
			Line    int64  `json:"line"`
			Message string `json:"message"`
			Param   string `json:"param"`
		} `json:"data"`
	} `json:"errors"`
	ExpiredAt     Time              `json:"expired_at"`
	ExpiresAt     Time              `json:"expires_at"`
	FailedAt      Time              `json:"failed_at"`
	FinalizingAt  Time              `json:"finalizing_at"`
	ID            string            `json:"id"`
	InProgressAt  Time              `json:"in_progress_at"`
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

// https://platform.openai.com/docs/api-reference/models/object
//
// Sadly the modalities aren't reported. The only way I can think of to find it at run time is to fetch
// https://platform.openai.com/docs/models/gpt-4o-mini-realtime-preview, find the div containing
// "Modalities:", then extract the modalities from the text
type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created Time   `json:"created"`
	OwnedBy string `json:"owned_by"`
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
	Error ErrorResponseError `json:"error"`
}

func (er *ErrorResponse) String() string {
	if er.Error.Code == "" {
		return fmt.Sprintf("error %s: %s", er.Error.Type, er.Error.Message)
	}
	return fmt.Sprintf("error %s (%s): %s", er.Error.Code, er.Error.Status, er.Error.Message)
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
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// In May 2025, OpenAI started pushing for Response API. They say it's the only way to keep reasoning items.
// It's interesting because Anthropic did that with the old API but OpenAI can't. Shrug.
// https://cookbook.openai.com/examples/responses_api/reasoning_items
// https://platform.openai.com/docs/api-reference/responses/create
// TODO: Switch over.

// New creates a new client to talk to the OpenAI platform API.
//
// If apiKey is not provided, it tries to load it from the OPENAI_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://platform.openai.com/settings/organization/api-keys
//
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://platform.openai.com/docs/models
//
// Pass model base.PreferredCheap to use a good cheap model, base.PreferredGood for a good model or
// base.PreferredSOTA to use its SOTA model. Keep in mind that as providers cycle through new models, it's
// possible the model is not available anymore.
//
// wrapper can be used to throttle outgoing requests, record calls, etc. It defaults to base.DefaultTransport.
//
// # Documents
//
// OpenAI supports many types of documents, listed at
// https://platform.openai.com/docs/assistants/tools/file-search#supported-files
func New(apiKey, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "OPENAI_API_KEY", URL: apiKeyURL}
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.openai.com/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "openai",
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
	if err == nil && (model == base.PreferredCheap || model == base.PreferredGood || model == base.PreferredSOTA) {
		mdls, err2 := c.ListModels(context.Background())
		if err2 != nil {
			return nil, err2
		}
		cheap := model == base.PreferredCheap
		good := model == base.PreferredGood
		c.Model = ""
		var date Time
		for _, mdl := range mdls {
			m := mdl.(*Model)
			if cheap {
				if strings.HasSuffix(m.ID, "-nano") && (date == 0 || m.Created < date) {
					// For the cheapest, we want the oldest model as it is generally cheaper.
					date = m.Created
					c.Model = m.ID
				}
			} else if good {
				if strings.HasSuffix(m.ID, "-mini") && (date == 0 || m.Created > date) {
					// For the greatest, we want the newest model as it is generally better.
					date = m.Created
					c.Model = m.ID
				}
			} else {
				if strings.HasSuffix(m.ID, "-pro") && (date == 0 || m.Created > date) {
					// For the greatest, we want the newest model as it is generally better.
					date = m.Created
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
	if c.isImage(opts) {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.GenDoc(ctx, msgs[0], opts)
	}
	return c.ProviderGen.GenSync(ctx, msgs, opts)
}

func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if c.isImage(opts) {
		return base.SimulateStream(ctx, c, msgs, chunks, opts)
	}
	return c.ProviderGen.GenStream(ctx, msgs, chunks, opts)
}

func (c *Client) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	// https://platform.openai.com/docs/api-reference/images/create
	res := genai.Result{}
	if err := c.Validate(); err != nil {
		return res, err
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
		Prompt:         msg.AsText(),
		Model:          c.Model,
		ResponseFormat: "b64_json",
	}
	// This is unfortunate.
	switch c.Model {
	case "gpt-image-1":
		req.Moderation = "low"
		// req.Background = "transparent"
		// req.OutputFormat = "webp"
		// req.OutputCompression = 90
		// req.Quality = "high"
		// req.Size = "1536x1024"
	case "dall-e-3":
		// req.Size = "1792x1024"
	case "dall-e-2":
		// We assume dall-e-2 is only used for smoke testing, so use the smallest image.
		req.Size = "256x256"
		// Maximum prompt length is 1000 characters.
		// Since we assume this is only for testing, silently cut it off.
		if len(req.Prompt) > 1000 {
			req.Prompt = req.Prompt[:1000]
		}
	default:
		// Silently pass.
	}
	if opts != nil {
		switch v := opts.(type) {
		case *OptionsImage:
			if v.Height != 0 && v.Width != 0 {
				req.Size = fmt.Sprintf("%dx%d", v.Width, v.Height)
			}
			req.Background = v.Background
		case *genai.OptionsImage:
			if v.Height != 0 && v.Width != 0 {
				req.Size = fmt.Sprintf("%dx%d", v.Width, v.Height)
			}
		default:
			return res, fmt.Errorf("unsupported options type %T", opts)
		}
	}
	url := "https://api.openai.com/v1/images/generations"

	// It is very different because it requires a multi-part upload.
	// https://platform.openai.com/docs/api-reference/images/createEdit
	// url = "https://api.openai.com/v1/images/edits"

	resp := ImageResponse{}
	if err := c.DoRequest(ctx, "POST", url, &req, &resp); err != nil {
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

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://platform.openai.com/docs/api-reference/models/list
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, "https://api.openai.com/v1/models")
}

func (c *Client) isImage(opts genai.Options) bool {
	switch c.Model {
	// TODO: Use Scoreboard list.
	case "dall-e-2", "dall-e-3", "gpt-image-1":
		return true
	default:
		return opts != nil && opts.Modality() == genai.ModalityImage
	}
}

func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Job, error) {
	if err := c.Validate(); err != nil {
		return "", err
	}
	if err := msgs.Validate(); err != nil {
		return "", err
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return "", err
		}
	}
	// Upload the messages and options as a file.
	b2 := BatchRequestInput{CustomID: "TODO", Method: "POST", URL: "/v1/chat/completions"}
	if err := b2.Body.Init(msgs, opts, c.Model); err != nil {
		return "", err
	}
	raw, err := json.Marshal(b2)
	if err != nil {
		return "", err
	}
	fileID, err := c.FileAdd(ctx, "batch.json", bytes.NewReader(raw), 24*time.Hour)
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
	err := c.DoRequest(ctx, "POST", "https://api.openai.com/v1/batches", &b, &resp)
	return resp, err
}

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
		res.FinishReason = genai.Pending
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
			if !c.ClientJSON.Lenient {
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
	err := c.DoRequest(ctx, "GET", u, nil, &out)
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
	err := c.DoRequest(ctx, "POST", u, nil, &resp)
	// TODO: Delete the file too.
	return resp, err
}

func (c *Client) FileAdd(ctx context.Context, filename string, r io.ReadSeeker, ttl time.Duration) (string, error) {
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
	resp, err := c.ClientJSON.Client.Do(req)
	if err != nil {
		return "", err
	}
	f := File{}
	err = c.DecodeResponse(resp, u, &f)
	return f.ID, err
}

func (c *Client) FileGet(ctx context.Context, id string) (io.ReadCloser, error) {
	// https://platform.openai.com/docs/api-reference/files/retrieve-contents
	u := "https://api.openai.com/v1/files/" + url.PathEscape(string(id)) + "/content"
	req, err := http.NewRequestWithContext(ctx, "GET", u, nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.ClientJSON.Client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, c.DecodeError(u, resp)
	}
	return resp.Body, nil
}

func (c *Client) FileDel(ctx context.Context, id string) error {
	// https://platform.openai.com/docs/api-reference/files/delete
	url := "https://api.openai.com/v1/files/" + url.PathEscape(string(id))
	out := FileDeleteResponse{}
	return c.DoRequest(ctx, "DELETE", url, nil, &out)
}

func (c *Client) FileList(ctx context.Context) ([]File, error) {
	// TODO: Pagination. It defaults at 10000 items per page.
	resp := FileListResponse{}
	err := c.DoRequest(ctx, "GET", "https://api.openai.com/v1/files", nil, &resp)
	return resp.Data, err
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		if pkt.Usage.PromptTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
		}
		if len(pkt.Choices) != 1 {
			continue
		}
		if fr := pkt.Choices[0].FinishReason; fr != "" {
			result.FinishReason = fr.ToFinishReason()
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
		f := genai.ContentFragment{TextFragment: pkt.Choices[0].Delta.Content}
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
