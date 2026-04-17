// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the OpenAI Chat Completion and related APIs.
//
// These structs map directly to the JSON objects exchanged with the OpenAI
// platform API. Documentation links are inline.
//
// Source: https://platform.openai.com/docs/api-reference/

package openaichat

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"mime"
	"reflect"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// ============================================================
// Shared types: enums, options, models, files.
// ============================================================

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
	// Flex processing is in beta, and currently only available for GPT-5, o3 and o4-mini models.
	//
	// https://platform.openai.com/docs/guides/flex-processing
	ServiceTierFlex ServiceTier = "flex"
)

// Validate implements genai.Validatable.
func (s ServiceTier) Validate() error {
	switch s {
	case "", ServiceTierAuto, ServiceTierDefault, ServiceTierFlex:
		return nil
	default:
		return fmt.Errorf("invalid service tier %q", s)
	}
}

// ReasoningEffort is the effort the model should put into reasoning. Default is Medium.
//
// https://platform.openai.com/docs/api-reference/assistants/createAssistant#assistants-createassistant-reasoning_effort
// https://platform.openai.com/docs/guides/reasoning
type ReasoningEffort string

// Reasoning effort values.
const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// Validate implements genai.Validatable.
func (r ReasoningEffort) Validate() error {
	switch r {
	case "", ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh:
		return nil
	default:
		return fmt.Errorf("invalid reasoning effort %q", r)
	}
}

// Background is only supported on gpt-image-1.
type Background string

// Background mode values.
const (
	BackgroundAuto        Background = "auto"
	BackgroundTransparent Background = "transparent"
	BackgroundOpaque      Background = "opaque"
)

// ============================================================
// Image generation types.
// ============================================================

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

// Init initializes the request from the given parameters.
func (i *ImageRequest) Init(msg *genai.Message, model string, opts ...genai.GenOption) error {
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
		// Other supported options: Background, OutputFormat, OutputCompression, Quality, Size.
	case "dall-e-3":
		// Other supported options: Size (e.g. 1792x1024).
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
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *GenOptionImage:
			i.Background = v.Background
		case *genai.GenOptionImage:
			if v.Height != 0 && v.Width != 0 {
				i.Size = fmt.Sprintf("%dx%d", v.Width, v.Height)
			}
		default:
			return &base.ErrNotSupported{Options: []string{internal.TypeName(opt)}}
		}
	}
	return nil
}

// ImageResponse is the provider-specific image generation response.
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
		OutputTokensDetails struct {
			TextTokens  int64 `json:"text_tokens"`
			ImageTokens int64 `json:"image_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
	Background   string `json:"background"`    // "opaque"
	Size         string `json:"size"`          // e.g. "1024x1024"
	Quality      string `json:"quality"`       // e.g. "medium"
	OutputFormat string `json:"output_format"` // e.g. "png"
}

// ImageChoiceData is the data for one image generation choice.
type ImageChoiceData struct {
	B64JSON       []byte `json:"b64_json"`
	RevisedPrompt string `json:"revised_prompt"` // dall-e-3 only
	URL           string `json:"url"`            // Unsupported for gpt-image-1
}

// ============================================================
// Model types.
// ============================================================

// Model is documented at https://platform.openai.com/docs/api-reference/models/object
//
// Sadly the modalities aren't reported. The only way I can think of to find it at run time is to fetch
// https://platform.openai.com/docs/models/gpt-4o-mini-realtime-preview, find the div containing
// "Modalities:", then extract the modalities from the text.
type Model struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created base.Time `json:"created"`
	OwnedBy string    `json:"owned_by"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s)", m.ID, m.Created.AsTime().Format("2006-01-02"))
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for OpenAI models listing.
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts OpenAI models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

// ============================================================
// File types.
// ============================================================

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

// GetID implements genai.Model.
func (f *File) GetID() string {
	return f.ID
}

// GetDisplayName implements genai.CacheItem.
func (f *File) GetDisplayName() string {
	return f.Filename
}

// GetExpiry implements genai.CacheItem.
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

// ============================================================
// Chat completion request types.
// ============================================================

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
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
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
		case *GenOptionText:
			c.ReasoningEffort = v.ReasoningEffort
			c.ServiceTier = v.ServiceTier
		case *genai.GenOptionText:
			unsupported = append(unsupported, c.initOptionsText(v, model)...)
			sp = v.SystemPrompt
		case *genai.GenOptionTools:
			c.initOptionsTools(v, model)
		case *genai.GenOptionWeb:
			if v.Search {
				c.WebSearchOptions = &WebSearchOptions{
					SearchContextSize: "high",
				}
			}
			if v.Fetch {
				errs = append(errs, errors.New("unsupported GenOptionWeb.Fetch"))
			}
		case genai.GenOptionSeed:
			if strings.HasPrefix(model, "gpt-4o-") && strings.Contains(model, "-search") {
				unsupported = append(unsupported, "GenOptionSeed")
			} else {
				c.Seed = int64(v)
			}
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
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
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
	c.StreamOptions.IncludeUsage = stream
}

func (c *ChatRequest) initOptionsText(v *genai.GenOptionText, model string) []string {
	var unsupported []string
	c.MaxChatTokens = v.MaxTokens
	// TODO: This is not great.
	if (strings.HasPrefix(model, "gpt-4o-") && strings.Contains(model, "-search")) ||
		model == "o1" ||
		strings.HasPrefix(model, "o1-") ||
		strings.HasPrefix(model, "o3-") ||
		strings.HasPrefix(model, "o4-") {
		if v.Temperature != 0 {
			unsupported = append(unsupported, "GenOptionText.Temperature")
		}
	} else {
		c.Temperature = v.Temperature
	}
	c.TopP = v.TopP
	if v.TopLogprobs > 0 {
		c.TopLogprobs = v.TopLogprobs
		c.Logprobs = true
	}
	if v.TopK != 0 {
		// Track this as an unsupported feature that can be ignored
		unsupported = append(unsupported, "GenOptionText.TopK")
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

func (c *ChatRequest) initOptionsTools(v *genai.GenOptionTools, model string) {
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

// ============================================================
// Chat message types.
// ============================================================

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
	ToolCallID  string       `json:"tool_call_id,omitzero"` // TODO: Document the role of this field.
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

// To converts to the genai equivalent.
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
		out.Replies = append(out.Replies, genai.Reply{Citation: c})
	}
	return nil
}

// Annotation is a provider-specific annotation.
type Annotation struct {
	Type        string `json:"type,omitzero"` // "url_citation"
	URLCitation struct {
		StartIndex int64  `json:"start_index,omitzero"`
		EndIndex   int64  `json:"end_index,omitzero"`
		Title      string `json:"title,omitzero"`
		URL        string `json:"url,omitzero"` // Has a ?utm_source=openai suffix.
	} `json:"url_citation,omitzero"`
}

// Contents is a collection of content blocks.
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

// Content is a provider-specific content block.
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

// FromRequest converts from a genai request.
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

// FromReply converts from a genai reply.
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

// To converts to the genai equivalent.
func (c *Content) To(out *genai.Reply) error {
	switch c.Type {
	case ContentText:
		out.Text = c.Text
		if c.Text == "" {
			return errors.New("received empty text")
		}
	case ContentImageURL, ContentInputAudio, ContentRefusal, ContentAudio, ContentFile:
		return fmt.Errorf("unsupported content type %q", c.Type)
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

// ContentType is a provider-specific content type.
type ContentType string

// Content type values.
const (
	ContentText       ContentType = "text"
	ContentImageURL   ContentType = "image_url"
	ContentInputAudio ContentType = "input_audio"
	ContentRefusal    ContentType = "refusal"
	ContentAudio      ContentType = "audio"
	ContentFile       ContentType = "file"
)

// ============================================================
// Tool types.
// ============================================================

// ToolCall is a provider-specific tool call.
type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

// From converts from the genai equivalent.
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

// To converts to the genai equivalent.
func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

// Tool is a provider-specific tool definition.
type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Description string             `json:"description,omitzero"`
		Name        string             `json:"name,omitzero"`
		Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
		Strict      bool               `json:"strict,omitzero"`
	} `json:"function,omitzero"`
}

// ============================================================
// Chat completion response types.
// ============================================================

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

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: c.Usage.PromptTokensDetails.CachedTokens,
			ReasoningTokens:   c.Usage.CompletionTokensDetails.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       c.Usage.TotalTokens,
			ServiceTier:       c.ServiceTier,
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

// FinishReason is a provider-specific finish reason.
type FinishReason string

// Finish reason values.
const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishToolCalls     FinishReason = "tool_calls"
	FinishContentFilter FinishReason = "content_filter"
)

// ToFinishReason converts to a genai.FinishReason.
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

// Usage is the provider-specific token usage.
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

// Logprobs is the provider-specific log probabilities.
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

// To converts to the genai equivalent.
func (l *Logprobs) To() [][]genai.Logprob {
	if len(l.Content) == 0 {
		return nil
	}
	out := make([][]genai.Logprob, 0, len(l.Content))
	for _, p := range l.Content {
		lp := make([]genai.Logprob, 1, len(p.TopLogprobs)+1)
		// Intentionally discard Bytes.
		lp[0] = genai.Logprob{Text: p.Token, Logprob: p.Logprob}
		for _, tlp := range p.TopLogprobs {
			lp = append(lp, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
		}
		out = append(out, lp)
	}
	// TODO: What happens if l.Refusal is not empty?
	return out
}

// ============================================================
// Streaming types.
// ============================================================

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

// ============================================================
// Batch types.
// ============================================================

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
	Model         string            `json:"model,omitzero"`
	Object        string            `json:"object"`         // "batch"
	OutputFileID  string            `json:"output_file_id"` // Output data
	RequestCounts struct {
		Completed int64 `json:"completed"`
		Failed    int64 `json:"failed"`
		Total     int64 `json:"total"`
	} `json:"request_counts"`
	Status string     `json:"status"`         // "completed", "in_progress", "validating", "finalizing"
	Usage  BatchUsage `json:"usage,omitzero"` // Token usage for the batch
}

// BatchUsage represents token usage information for a batch.
type BatchUsage struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

// ============================================================
// Error types.
// ============================================================

// ErrorResponse is the provider-specific error response.
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

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// ErrorResponseError is the nested error in an error response.
type ErrorResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
	Type    string `json:"type"`
	Param   string `json:"param"`
}
