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
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for OpenAI.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
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
	ToolChoice        string            `json:"tool_choice,omitzero"` // "none", "auto", "required"
	ParallelToolCalls bool              `json:"parallel_tool_calls,omitzero"`
	User              string            `json:"user,omitzero"`
	WebSearchOptions  *WebSearchOptions `json:"web_search_options,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	c.Model = model
	var errs []error
	var unsupported []string
	if err := msgs.Validate(); err != nil {
		return err
	}
	sp := ""
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *OptionsText:
			c.ReasoningEffort = v.ReasoningEffort
			c.ServiceTier = v.ServiceTier
		case *genai.OptionsText:
			unsupported = append(unsupported, c.initOptionsText(v, model)...)
			sp = v.SystemPrompt
		case *genai.OptionsTools:
			unsupported = append(unsupported, c.initOptionsTools(v, model)...)
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opt))
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

func (c *ChatRequest) initOptionsText(v *genai.OptionsText, model string) []string {
	var unsupported []string
	c.MaxChatTokens = v.MaxTokens
	// TODO: This is not great.
	if (strings.HasPrefix(model, "gpt-4o-") && strings.Contains(model, "-search")) ||
		model == "o1" ||
		strings.HasPrefix(model, "o1-") ||
		strings.HasPrefix(model, "o3-") ||
		strings.HasPrefix(model, "o4-") {
		if v.Temperature != 0 {
			unsupported = append(unsupported, "OptionsText.Temperature")
		}
	} else {
		c.Temperature = v.Temperature
	}
	c.TopP = v.TopP
	if strings.HasPrefix(model, "gpt-4o-") && strings.Contains(model, "-search") {
		if v.Seed != 0 {
			unsupported = append(unsupported, "OptionsText.Seed")
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
		unsupported = append(unsupported, "OptionsText.TopK")
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
	return unsupported
}

func (c *ChatRequest) initOptionsTools(v *genai.OptionsTools, model string) []string {
	var unsupported []string
	if len(v.Tools) != 0 {
		// TODO: Determine exactly which models do not support this.
		if model != "o4-mini" {
			c.ParallelToolCalls = true
		}
		switch v.Force {
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
	if v.WebSearch {
		c.WebSearchOptions = &WebSearchOptions{
			SearchContextSize: "high",
		}
	}
	return unsupported
}

// WebSearchOptions is "documented" at https://platform.openai.com/docs/guides/tools-web-search
type WebSearchOptions struct {
	SearchContextSize string `json:"search_context_size,omitzero"` // "low", "medium", "high"
	UserLocation      struct {
		Type        string `json:"type,omitzero"` // "approximate"
		Approximate struct {
			Country string `json:"country,omitzero"` // "GB"
			City    string `json:"city,omitzero"`    // "London"
			Region  string `json:"region,omitzero"`  // "London"
		} `json:"approximate,omitzero"`
	} `json:"user_location,omitzero"`
}

// Message is documented at https://platform.openai.com/docs/api-reference/chat/create
type Message struct {
	Role    string   `json:"role,omitzero"` // "developer", "assistant", "user"
	Name    string   `json:"name,omitzero"` // An optional name for the participant. Provides the model information to differentiate between participants of the same role.
	Content Contents `json:"content,omitzero"`
	Refusal string   `json:"refusal,omitzero"` // The refusal message by the assistant.
	Audio   struct {
		ID string `json:"id,omitzero"`
	} `json:"audio,omitzero"`
	ToolCalls   []ToolCall   `json:"tool_calls,omitzero"`
	ToolCallID  string       `json:"tool_call_id,omitzero"` // TODO
	Annotations []Annotation `json:"annotations,omitzero"`
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
			if in.Replies[i].Reasoning != "" {
				// Ignore reasoning messages.
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
	for _, a := range m.Annotations {
		if a.Type != "url_citation" {
			return &internal.BadError{Err: fmt.Errorf("unsupported annotation type %q", a.Type)}
		}
		c := genai.Citation{
			StartIndex: a.URLCitation.StartIndex,
			EndIndex:   a.URLCitation.EndIndex,
			Sources: []genai.CitationSource{{
				Type:  genai.CitationWeb,
				Title: a.URLCitation.Title,
				URL:   a.URLCitation.URL,
			}},
		}
		out.Replies = append(out.Replies, genai.Reply{Citations: []genai.Citation{c}})
	}
	return nil
}

type Annotation struct {
	Type        string `json:"type,omitzero"` // "url_citation"
	URLCitation struct {
		StartIndex int64  `json:"start_index,omitzero"`
		EndIndex   int64  `json:"end_index,omitzero"`
		Title      string `json:"title,omitzero"`
		URL        string `json:"url,omitzero"` // Has a ?utm_source=openai suffix.
	} `json:"url_citation,omitzero"`
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
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			// OpenAI chat API doesn't support text documents as attachment.
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
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
		return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
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
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			// OpenAI chat API doesn't support text documents as attachment.
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
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
					return &internal.BadError{Err: fmt.Errorf("unknown extension for mime type %s", mimeType)}
				}
				filename = "content" + exts[0]
			}
			c.Type = ContentFile
			c.File.Filename = filename
			c.File.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		}
		return nil
	}
	return &internal.BadError{Err: errors.New("unknown Reply type")}
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
			Content     string       `json:"content"`
			Role        string       `json:"role"`
			Refusal     string       `json:"refusal"`
			ToolCalls   []ToolCall   `json:"tool_calls"`
			Annotations []Annotation `json:"annotations"`
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

// In May 2025, OpenAI started pushing for Response API. They say it's the only way to keep reasoning items.
// It's interesting because Anthropic did that with the old API but OpenAI can't. Shrug.
// https://cookbook.openai.com/examples/responses_api/reasoning_items
// https://platform.openai.com/docs/api-reference/responses/create
// TODO: Switch over.

// Client implements genai.Provider.
type Client struct {
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

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
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
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
	switch len(opts.OutputModalities) {
	case 0:
		// Auto-detect below.
	case 1:
		switch opts.OutputModalities[0] {
		case genai.ModalityAudio, genai.ModalityImage, genai.ModalityText, genai.ModalityVideo:
		case genai.ModalityDocument:
			fallthrough
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image or text are supported", opts.OutputModalities)
		}
	default:
		return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image or text are supported", opts.OutputModalities)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:           "https://api.openai.com/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			PreloadedModels:      opts.PreloadedModels,
			ProcessHeaders:       processHeaders,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				// OpenAI error message prints the api key URL already.
				APIKeyURL: "",
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
			var mod genai.Modality
			switch len(opts.OutputModalities) {
			case 0:
				mod = genai.ModalityText
			case 1:
				mod = opts.OutputModalities[0]
			default:
				// TODO: Maybe it's possible, need to double check.
				return nil, fmt.Errorf("can't use model %s with option Modalities %s", opts.Model, opts.OutputModalities)
			}
			switch mod {
			case genai.ModalityText:
				if c.impl.Model, err = c.selectBestTextModel(ctx, opts.Model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityImage:
				if c.impl.Model, err = c.selectBestImageModel(ctx, opts.Model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityAudio, genai.ModalityDocument, genai.ModalityVideo:
				fallthrough
			default:
				// TODO: Soon, because it's cool.
				return nil, fmt.Errorf("automatic model selection is not implemented yet for modality %s (send PR to add support)", opts.OutputModalities)
			}
		default:
			c.impl.Model = opts.Model
			switch len(opts.OutputModalities) {
			case 0:
				c.impl.OutputModalities, err = c.detectModelModalities(ctx, opts.Model)
			case 1:
				c.impl.OutputModalities = opts.OutputModalities
			default:
				// TODO: Maybe it's possible, need to double check.
				return nil, fmt.Errorf("can't use model %s with option Modalities %s", opts.Model, opts.OutputModalities)
			}
		}
	}
	return c, err
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "openaichat"
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// GenAsync implements genai.ProviderGenAsync.
//
// It requests the providers' batch API and returns the job ID. It can take up to 24 hours to complete.
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Job, error) {
	fileID, err := c.CacheAddRequest(ctx, msgs, "TODO", "batch.json", 24*time.Hour, opts...)
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
			if !c.impl.Lenient {
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

func (c *Client) CacheAddRequest(ctx context.Context, msgs genai.Messages, name, displayName string, ttl time.Duration, opts ...genai.Options) (string, error) {
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	// Upload the messages and options as a file.
	b := BatchRequestInput{CustomID: name, Method: "POST", URL: "/v1/chat/completions"}
	if err := b.Body.Init(msgs, c.impl.Model, opts...); err != nil {
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
	resp, err := c.impl.Client.Do(req)
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
	resp, err := c.impl.Client.Do(req)
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

func processStreamPackets(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.ReplyFragment], func() (genai.Usage, []genai.Logprobs, error)) {
	var finalErr error
	u := genai.Usage{}
	var l []genai.Logprobs

	return func(yield func(genai.ReplyFragment) bool) {
			pendingToolCall := ToolCall{}
			for pkt := range chunks {
				if pkt.Usage.PromptTokens != 0 {
					u.InputTokens = pkt.Usage.PromptTokens
					u.InputCachedTokens = pkt.Usage.PromptTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Usage.CompletionTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Usage.CompletionTokens
				}
				if len(pkt.Choices) != 1 {
					continue
				}
				l = append(l, pkt.Choices[0].Logprobs.To()...)
				if fr := pkt.Choices[0].FinishReason; fr != "" {
					u.FinishReason = fr.ToFinishReason()
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "", "assistant":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
					finalErr = &internal.BadError{Err: fmt.Errorf("implement multiple tool calls: %#v", pkt)}
					return
				}
				if r := pkt.Choices[0].Delta.Refusal; r != "" {
					finalErr = &internal.BadError{Err: fmt.Errorf("refused: %q", r)}
					return
				}

				f := genai.ReplyFragment{}
				for _, a := range pkt.Choices[0].Delta.Annotations {
					f.Citation.StartIndex = a.URLCitation.StartIndex
					f.Citation.EndIndex = a.URLCitation.EndIndex
					f.Citation.Sources = []genai.CitationSource{{Type: genai.CitationWeb, URL: a.URLCitation.URL}}
					if !yield(f) {
						return
					}
					f = genai.ReplyFragment{}
				}

				f.TextFragment = pkt.Choices[0].Delta.Content
				// OpenAI streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
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
				if !f.IsZero() {
					if !yield(f) {
						return
					}
				}
			}
		}, func() (genai.Usage, []genai.Logprobs, error) {
			return u, l, finalErr
		}
}

var (
	_ genai.Provider      = &Client{}
	_ genai.ProviderCache = &Client{}
)
