// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openairesponses implements a client for the OpenAI Responses API.
//
// It is described at https://platform.openai.com/docs/api-reference/responses/create
package openairesponses

// See official client at http://pkg.go.dev/github.com/openai/openai-go/responses

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"mime"
	"net/http"
	"os"
	"reflect"
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
//   - OpenAI supports more than what the client supports.
//   - Tool calling works very well but is biased; the model is lazy and when it's unsure, it will use the
//     tool's first argument.
//   - Rate limit is based on how much you spend per month: https://platform.openai.com/docs/guides/rate-limits
var Scoreboard = genai.Scoreboard{
	Country:      "US",
	DashboardURL: "https://platform.openai.com/usage",
	Scenarios: []genai.Scenario{
		{
			Models: []string{"gpt-4.1"},
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityPDF:  {Inline: true, SupportedFormats: []string{"application/pdf"}},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				NoStopSequence:   true,
				BrokenTokenUsage: genai.Flaky, // When using MaxTokens.
				Tools:            genai.True,
				BiasedTool:       genai.True,
				JSON:             true,
				JSONSchema:       true,
			},
			GenStream: &genai.FunctionalityText{
				NoStopSequence:   true,
				BrokenTokenUsage: genai.Flaky, // When using MaxTokens.
				Tools:            genai.True,
				BiasedTool:       genai.True,
				JSON:             true,
				JSONSchema:       true,
			},
		},
		// https://platform.openai.com/docs/guides/audio
		// Audio is not yet supported in the Responses API.
		{
			Models: []string{"gpt-4o-audio-preview"},
		},
		// The model is completely misbehaving with PDFs. This is weird.
		{
			Models: []string{"o4-mini"},
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				NoStopSequence:   true,
				BrokenTokenUsage: genai.Flaky, // When using MaxTokens.
				Tools:            genai.True,
				BiasedTool:       genai.True,
				JSON:             true,
				JSONSchema:       true,
			},
			GenStream: &genai.FunctionalityText{
				NoStopSequence:   true,
				BrokenTokenUsage: genai.Flaky, // When using MaxTokens.
				Tools:            genai.True,
				BiasedTool:       genai.True,
				JSON:             true,
				JSONSchema:       true,
			},
		},
		{
			Models: []string{"gpt-image-1"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out: map[genai.Modality]genai.ModalCapability{
				// TODO: Expose other supported image formats.
				genai.ModalityImage: {
					Inline:           true,
					SupportedFormats: []string{"image/jpeg"},
				},
			},
			GenDoc: &genai.FunctionalityDoc{
				Seed:               true,
				BrokenTokenUsage:   genai.True,
				BrokenFinishReason: true,
			},
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
				"o1",
				"o1-2024-12-17",
				"o1-mini",
				"o1-mini-2024-09-12",
				"o1-preview",
				"o1-preview-2024-09-12",
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

// Response represents a request to the OpenAI Responses API.
//
// https://platform.openai.com/docs/api-reference/responses/object
type Response struct {
	Model              string            `json:"model"`
	Background         bool              `json:"background"`
	Instructions       string            `json:"instructions,omitzero"`
	MaxOutputTokens    int64             `json:"max_output_tokens,omitzero"`
	MaxToolCalls       int64             `json:"max_tool_calls,omitzero"`
	Metadata           map[string]string `json:"metadata,omitzero"`
	ParallelToolCalls  bool              `json:"parallel_tool_calls,omitzero"`
	PreviousResponseID string            `json:"previous_response_id,omitzero"`
	PromptCacheKey     struct{}          `json:"prompt_cache_key,omitzero"`
	Reasoning          ReasoningConfig   `json:"reasoning,omitzero"`
	SafetyIdentifier   struct{}          `json:"safety_identifier,omitzero"`
	ServiceTier        ServiceTier       `json:"service_tier,omitzero"`
	Store              bool              `json:"store"`
	Temperature        float64           `json:"temperature,omitzero"`
	Text               struct {
		Format struct {
			Type        string             `json:"type"` // "text", "json_schema", "json_object"
			Name        string             `json:"name,omitzero"`
			Description string             `json:"description,omitzero"`
			Schema      *jsonschema.Schema `json:"schema,omitzero"`
			Strict      bool               `json:"strict,omitzero"`
		} `json:"format"`
	} `json:"text,omitzero"`
	TopLogprobs float64 `json:"top_logprobs,omitzero"`
	TopP        float64 `json:"top_p,omitzero"`
	ToolChoice  string  `json:"tool_choice,omitzero"` // "none", "auto", "required"
	Truncation  string  `json:"truncation,omitzero"`  // "disabled"
	Tools       []Tool  `json:"tools,omitzero"`
	User        string  `json:"user,omitzero"`

	// Request only
	Input  []Message `json:"input,omitzero"`
	Stream bool      `json:"stream,omitzero"`

	// Response only
	ID                string            `json:"id,omitzero"`
	Object            string            `json:"object,omitzero"` // "response"
	CreatedAt         base.Time         `json:"created_at,omitzero"`
	Status            string            `json:"status,omitzero"` // "completed"
	IncompleteDetails IncompleteDetails `json:"incomplete_details,omitzero"`
	Error             APIError          `json:"error,omitzero"`
	Output            []Message         `json:"output,omitzero"`
	Usage             Usage             `json:"usage,omitzero"`
}

// Init implements base.InitializableRequest.
func (r *Response) Init(msgs genai.Messages, opts genai.Options, model string) error {
	var unsupported []string
	var errs []error
	r.Model = model
	if opts != nil {
		switch v := opts.(type) {
		case *OptionsText:
			r.Reasoning.Effort = v.ReasoningEffort
			r.ServiceTier = v.ServiceTier
			unsupported, errs = r.initOptions(&v.OptionsText, model)
		case *genai.OptionsText:
			unsupported, errs = r.initOptions(v, model)
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
	}
	if len(msgs) == 0 {
		return fmt.Errorf("no messages provided")
	}
	r.Input = make([]Message, len(msgs))
	for i := range msgs {
		if err := r.Input[i].From(&msgs[i]); err != nil {
			errs = append(errs, err)
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

// SetStream implements base.InitializableRequest.
func (r *Response) SetStream(stream bool) {
	r.Stream = stream
}

// ToResult implements base.ResultConverter.
func (r *Response) ToResult() (genai.Result, error) {
	res := genai.Result{
		Usage: genai.Usage{
			InputTokens:       r.Usage.InputTokens,
			InputCachedTokens: r.Usage.InputTokensDetails.CachedTokens,
			OutputTokens:      r.Usage.OutputTokens,
		},
	}
	if len(r.Output) > 1 {
		// OpenAI devs smoked crack. We need to merge the messages into one.
		// return res, fmt.Errorf("multiple outputs not supported: %#v", r.Output)
	}
	for _, output := range r.Output {
		if err := output.To(&res.Message); err != nil {
			return res, err
		}
	}
	if r.IncompleteDetails.Reason != "" {
		if r.IncompleteDetails.Reason == "max_output_tokens" {
			res.FinishReason = genai.FinishedLength
		} else {
			res.FinishReason = genai.FinishReason(r.IncompleteDetails.Reason)
		}
	} else if len(res.ToolCalls) != 0 {
		res.FinishReason = genai.FinishedToolCalls
	} else {
		res.FinishReason = genai.FinishedStop
	}
	return res, nil
}

func (r *Response) initOptions(v *genai.OptionsText, model string) ([]string, []error) {
	var unsupported []string
	var errs []error
	r.MaxOutputTokens = v.MaxTokens
	r.Temperature = v.Temperature
	r.TopP = v.TopP
	if v.SystemPrompt != "" {
		r.Instructions = v.SystemPrompt
	}
	if v.Seed != 0 {
		unsupported = append(unsupported, "Seed")
	}
	if v.TopK != 0 {
		unsupported = append(unsupported, "TopK")
	}
	if len(v.Stop) != 0 {
		errs = append(errs, errors.New("unsupported option Stop"))
	}
	if v.DecodeAs != nil {
		r.Text.Format.Type = "json_schema"
		// OpenAI requires a name.
		r.Text.Format.Name = "response"
		r.Text.Format.Strict = true
		r.Text.Format.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
	} else if v.ReplyAsJSON {
		r.Text.Format.Type = "json_object"
	}
	if len(v.Tools) != 0 {
		r.ParallelToolCalls = true
		switch v.ToolCallRequest {
		case genai.ToolCallAny:
			r.ToolChoice = "auto"
		case genai.ToolCallRequired:
			r.ToolChoice = "required"
		case genai.ToolCallNone:
			r.ToolChoice = "none"
		}
		r.Tools = make([]Tool, len(v.Tools))
		for i, t := range v.Tools {
			if t.Name == "" {
				errs = append(errs, errors.New("tool name is required"))
			}
			r.Tools[i].Type = "function"
			r.Tools[i].Name = t.Name
			r.Tools[i].Description = t.Description
			if r.Tools[i].Parameters = t.InputSchemaOverride; r.Tools[i].Parameters == nil {
				r.Tools[i].Parameters = t.GetInputSchema()
			}
		}
	}
	return unsupported, errs
}

// ReasoningConfig represents reasoning configuration for o-series models.
type ReasoningConfig struct {
	Effort  ReasoningEffort `json:"effort,omitzero"`
	Summary string          `json:"summary,omitzero"` // "auto", "concise", "detailed"
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

// Tool represents a tool that can be called by the model.
type Tool struct {
	Type string `json:"type,omitzero"` // "function", "file_search", "computer_use_preview", "mcp", "code_interpreter", "image_generation", "local_shell"

	// Type == "function"
	Name        string             `json:"name,omitzero"`
	Description string             `json:"description,omitzero"`
	Parameters  *jsonschema.Schema `json:"parameters,omitzero"`
	Strict      bool               `json:"strict,omitzero"`

	FileSearchVectorStoreIDs []string `json:"vector_store_ids,omitzero"` // for file_search tools
}

type MessageType string

const (
	// Inputs and Outputs
	MessageMessage MessageType = "message"
	// Outputs
	MessageFileSearchCall      MessageType = "file_search_call"
	MessageComputerCall        MessageType = "computer_call"
	MessageWebSearchCall       MessageType = "web_search_call"
	MessageFunctionCall        MessageType = "function_call"
	MessageReasoning           MessageType = "reasoning"
	MessageImageGenerationCall MessageType = "image_generation_call"
	MessageCodeInterpreterCall MessageType = "code_interpreter_call"
	MessageLocalShellCall      MessageType = "local_shell_call"
	MessageMcpListTools        MessageType = "mcp_list_tools"
	MessageMcpApprovalRequest  MessageType = "mcp_approval_request"
	MessageMcpCall             MessageType = "mcp_call"
	// Inputs
	MessageComputerCallOutput   MessageType = "computer_call_output"
	MessageFunctionCallOutput   MessageType = "function_call_output"
	MessageLocalShellCallOutput MessageType = "local_shell_call_output"
	MessageMcpApprovalResponse  MessageType = "mcp_approval_response"
	MessageItemReference        MessageType = "item_reference"
)

// Message represents a message input to the model.
type Message struct {
	Type MessageType `json:"type,omitzero"`

	// Type == MessageMessage
	Role    string    `json:"role,omitzero"` // "user", "assistant", "system", "developer"
	Content []Content `json:"content,omitzero"`

	// Type == MessageMessage, MessageFileSearchCall, MessageFunctionCall, MessageReasoning
	Status string `json:"status,omitzero"` // "in_progress", "completed", "incomplete", "searching", "failed"

	// Type == MessageMessage (with Role == "assistant"), MessageFileSearchCall, MessageItemReference,
	// MessageFunctionCall, MessageFunctionCallOutput, MessageReasoning
	ID string `json:"id,omitzero"` // MessageItemReference: an internal identifier for an item to reference; Others: tool call ID

	// Type == MessageFileSearchCall
	Queries []string `json:"queries,omitzero"`
	Results []struct {
		Attributes map[string]string `json:"attributes,omitzero"`
		FileID     string            `json:"file_id,omitzero"`
		Filename   string            `json:"filename,omitzero"`
		Score      float64           `json:"score,omitzero"` // [0, 1]
		Text       string            `json:"text,omitzero"`
	} `json:"results,omitzero"`

	// Type == MessageFunctionCall
	Arguments string `json:"arguments,omitzero"` // JSON
	Name      string `json:"name,omitzero"`

	// Type == MessageFunctionCall, MessageFunctionCallOutput
	CallID string `json:"call_id,omitzero"`

	// Type == MessageFunctionCallOutput
	Output string `json:"output,omitzero"` // JSON

	// Type == MessageReasoning
	EncryptedContent string             `json:"encrypted_content,omitzero"`
	Summary          []ReasoningSummary `json:"summary,omitzero"`
}

func (m *Message) From(msg *genai.Message) error {
	switch msg.Role {
	case genai.Assistant, genai.User:
		m.Role = string(msg.Role)
	default:
		return fmt.Errorf("implement role %q", msg.Role)
	}
	if len(msg.ToolCallResults) != 0 {
		if len(msg.ToolCallResults) > 1 {
			return fmt.Errorf("multiple tool outputs not supported in a single message for OpenAI Responses API")
		}
		m.Type = MessageFunctionCallOutput
		m.Role = ""
		m.CallID = msg.ToolCallResults[0].ID
		m.Output = msg.ToolCallResults[0].Result
		return nil
	}
	if len(msg.ToolCalls) != 0 {
		if len(msg.ToolCalls) > 1 {
			return fmt.Errorf("multiple tool calls not supported in a single message for OpenAI Responses API")
		}
		m.Type = MessageFunctionCall
		m.Role = ""
		m.CallID = msg.ToolCalls[0].ID
		m.Name = msg.ToolCalls[0].Name
		m.Arguments = msg.ToolCalls[0].Arguments
		return nil
	}
	if len(msg.Contents) != 0 {
		m.Type = MessageMessage
		m.Content = make([]Content, len(msg.Contents))
		for j := range msg.Contents {
			if err := m.Content[j].From(&msg.Contents[j]); err != nil {
				return fmt.Errorf("block %d: %w", j, err)
			}
		}
		return nil
	}
	return fmt.Errorf("implement message: %#v", msg)
}

func (m *Message) To(out *genai.Message) error {
	switch m.Role {
	case "assistant", "":
		// genai requires a role.
		out.Role = genai.Assistant
	default:
		return fmt.Errorf("unsupported role %q", m.Role)
	}
	// We only need to implement the types that can be returned from the LLM.
	switch m.Type {
	case MessageMessage:
		for i := range m.Content {
			c := genai.Content{}
			if err := m.Content[i].To(&c); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
			out.Contents = append(out.Contents, c)
		}
	case MessageReasoning:
		for i := range m.Summary {
			if m.Summary[i].Type != "summary_text" {
				return fmt.Errorf("unsupported summary type %q", m.Summary[i].Type)
			}
			out.Contents = append(out.Contents, genai.Content{Thinking: m.Summary[i].Text})
		}
	case MessageFunctionCall:
		out.ToolCalls = append(out.ToolCalls, genai.ToolCall{ID: m.CallID, Name: m.Name, Arguments: m.Arguments})
	default:
		return fmt.Errorf("unsupported output type %q", m.Type)
	}
	return nil
}

type ContentType string

const (
	// Inputs
	ContentInputText  ContentType = "input_text"
	ContentInputImage ContentType = "input_image"
	ContentInputFile  ContentType = "input_file"

	// Outputs
	ContentOutputText ContentType = "output_text"
	ContentRefusal    ContentType = "refusal"
)

// Content represents different types of input content.
type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == ContentInputText, ContentOutputText
	Text string `json:"text,omitzero"`

	// Type == ContentInputImage, ContentInputFile
	FileID string `json:"file_id,omitzero"`

	// Type == ContentInputImage
	ImageURL string `json:"image_url,omitzero"` // URL or base64
	Detail   string `json:"detail,omitzero"`    // "high", "low", "auto" (default)

	// Type == ContentInputFile
	FileData string `json:"file_data,omitzero"` // TODO: confirm if base64
	Filename string `json:"filename,omitzero"`

	// Type == ContentOutputText
	Annotations []Annotation `json:"annotations,omitzero"`
	Logprobs    []Logprobs   `json:"logprobs,omitzero"` // TODO: I believe this is incorrect.

	// Type == ContentRefusal
	Refusal string `json:"refusal,omitzero"`
}

func (c *Content) To(out *genai.Content) error {
	if len(c.Annotations) != 0 {
		// Citations!!
		return fmt.Errorf("implement citations: %#v", c.Annotations)
	}
	switch c.Type {
	case ContentOutputText:
		out.Text = c.Text
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = ContentInputText
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
		c.Type = ContentInputImage
		c.Detail = "auto" // TODO: Make it configurable.
		if in.URL == "" {
			c.ImageURL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL = in.URL
		}
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
		c.Type = ContentInputFile
		c.Filename = filename
		c.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
	}
	return nil
}

// APIError represents an API error in the response.
type APIError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// IncompleteDetails represents details about why a response is incomplete.
type IncompleteDetails struct {
	Reason string `json:"reason"`
}

// Annotation represents annotations in output text.
type Annotation struct {
	Type string `json:"type,omitzero"` // "file_citation", "url_citation", "container_file_citation", "file_path"

	// Type == "file_citation", "container_file_citation", "file_path"
	FileID string `json:"file_id,omitzero"`

	// Type == "file_citation", "file_path"
	Index int64 `json:"index,omitzero"`

	// Type == "url_citation"
	URL   string `json:"url,omitzero"`
	Title string `json:"title,omitzero"`

	// Type == "url_citation", "container_file_citation"
	StartIndex int64 `json:"start_index,omitzero"`
	EndIndex   int64 `json:"end_index,omitzero"`
}

type Logprobs struct {
	Bytes       []int64 `json:"bytes,omitzero"`
	Token       string  `json:"token,omitzero"`
	Logprob     float64 `json:"logprob,omitzero"`
	TopLogprobs []struct {
		Bytes   []int64 `json:"bytes,omitzero"`
		Token   string  `json:"token,omitzero"`
		Logprob float64 `json:"logprob,omitzero"`
	} `json:"top_logprobs,omitzero"`
}

// ReasoningSummary represents reasoning summary content.
type ReasoningSummary struct {
	Type string `json:"type,omitzero"` // "summary_text"
	Text string `json:"text,omitzero"`
}

// Usage represents token usage statistics.
type Usage struct {
	InputTokens        int64 `json:"input_tokens"`
	InputTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokens        int64 `json:"output_tokens"`
	OutputTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
	TotalTokens int64 `json:"total_tokens"`
}

// TokenDetails provides detailed token usage breakdown.
type TokenDetails struct {
	CachedTokens    int64 `json:"cached_tokens,omitzero"`
	AudioTokens     int64 `json:"audio_tokens,omitzero"`
	ReasoningTokens int64 `json:"reasoning_tokens,omitzero"`
}

// ResponseType is one of the event at https://platform.openai.com/docs/api-reference/responses-streaming
type ResponseType string

const (
	ResponseCompleted                       ResponseType = "response.completed"
	ResponseContentPartAdded                ResponseType = "response.content_part.added"
	ResponseContentPartDone                 ResponseType = "response.content_part.done"
	ResponseCreated                         ResponseType = "response.created"
	ResponseError                           ResponseType = "error"
	ResponseFailed                          ResponseType = "response.failed"
	ResponseFileSearchCallCompleted         ResponseType = "response.file_search_call.completed"
	ResponseFileSearchCallInProgress        ResponseType = "response.file_search_call.in_progress"
	ResponseFileSearchCallSearching         ResponseType = "response.file_search_call.searching"
	ResponseFunctionCallArgumentsDelta      ResponseType = "response.function_call_arguments.delta"
	ResponseFunctionCallArgumentsDone       ResponseType = "response.function_call_arguments.done"
	ResponseImageGenerationCallCompleted    ResponseType = "response.image_generation_call.completed"
	ResponseImageGenerationCallGenerating   ResponseType = "response.image_generation_call.generating"
	ResponseImageGenerationCallInProgress   ResponseType = "response.image_generation_call.in_progress"
	ResponseImageGenerationCallPartialImage ResponseType = "response.image_generation_call.partial_image"
	ResponseInProgress                      ResponseType = "response.in_progress"
	ResponseIncomplete                      ResponseType = "response.incomplete"
	ResponseMCPCallArgumentsDelta           ResponseType = "response.mcp_call.arguments.delta"
	ResponseMCPCallArgumentsDone            ResponseType = "response.mcp_call.arguments.done"
	ResponseMCPCallCompleted                ResponseType = "response.mcp_call.completed"
	ResponseMCPCallFailed                   ResponseType = "response.mcp_call.failed"
	ResponseMCPCallInProgress               ResponseType = "response.mcp_call.in_progress"
	ResponseMCPListToolsCompleted           ResponseType = "response.mcp_list_tools.completed"
	ResponseMCPListToolsFailed              ResponseType = "response.mcp_list_tools.failed"
	ResponseMCPListToolsInProgress          ResponseType = "response.mcp_list_tools.in_progress"
	ResponseOutputItemAdded                 ResponseType = "response.output_item.added"
	ResponseOutputItemDone                  ResponseType = "response.output_item.done"
	ResponseOutputTextDelta                 ResponseType = "response.output_text.delta"
	ResponseOutputTextDone                  ResponseType = "response.output_text.done"
	ResponseOutputTextAnnotationAdded       ResponseType = "response.output_text_annotation.added"
	ResponseQueued                          ResponseType = "response.queued"
	ResponseReasoningDelta                  ResponseType = "response.reasoning.delta"
	ResponseReasoningDone                   ResponseType = "response.reasoning.done"
	ResponseReasoningSummaryDelta           ResponseType = "response.reasoning_summary.delta"
	ResponseReasoningSummaryDone            ResponseType = "response.reasoning_summary.done"
	ResponseReasoningSummaryPartAdded       ResponseType = "response.reasoning_summary_part.added"
	ResponseReasoningSummaryPartDone        ResponseType = "response.reasoning_summary_part.done"
	ResponseReasoningSummaryTextDelta       ResponseType = "response.reasoning_summary_text.delta"
	ResponseReasoningSummaryTextDone        ResponseType = "response.reasoning_summary_text.done"
	ResponseRefusalDelta                    ResponseType = "response.refusal.delta"
	ResponseRefusalDone                     ResponseType = "response.refusal.done"
	ResponseWebSearchCallCompleted          ResponseType = "response.web_search_call.completed"
	ResponseWebSearchCallInProgress         ResponseType = "response.web_search_call.in_progress"
	ResponseWebSearchCallSearching          ResponseType = "response.web_search_call.searching"
)

// ResponseStreamChunkResponse represents a streaming response chunk.
//
// https://platform.openai.com/docs/api-reference/responses-streaming
type ResponseStreamChunkResponse struct {
	Type           ResponseType `json:"type,omitzero"`
	SequenceNumber int64        `json:"sequence_number,omitzero"`

	// Type == ResponseCreated, ResponseInProgress, ResponseCompleted, ResponseFailed, ResponseIncomplete,
	// ResponseQueued
	Response Response `json:"response,omitzero"`

	// Type == ResponseOutputItemAdded, ResponseOutputItemDone, ResponseContentPartAdded,
	// ResponseContentPartDone, ResponseOutputTextDelta, ResponseOutputTextDone, ResponseRefusalDelta,
	// ResponseRefusalDone, ResponseFunctionCallArgumentsDelta, ResponseFunctionCallArgumentsDone,
	// ResponseReasoningSummaryPartAdded, ResponseReasoningSummaryPartDone, ResponseReasoningSummaryTextDelta,
	// ResponseReasoningSummaryTextDone, ResponseReasoningDone, ResponseReasoningSummaryDelta,
	// ResponseReasoningSummaryDone, ResponseOutputTextAnnotationAdded
	OutputIndex int64 `json:"output_index,omitzero"`

	// Type == ResponseOutputItemAdded, ResponseOutputItemDone
	Item Message `json:"item,omitzero"`

	// Type == ResponseContentPartAdded, ResponseContentPartDone, ResponseOutputTextDelta,
	// ResponseOutputTextDone, ResponseRefusalDelta, ResponseRefusalDone, ResponseReasoningDelta,
	// ResponseReasoningDone, ResponseOutputTextAnnotationAdded
	ContentIndex int64 `json:"content_index,omitzero"`

	// Type == ResponseContentPartAdded, ResponseContentPartDone, ResponseOutputTextDelta,
	// ResponseOutputTextDone, ResponseRefusalDelta, ResponseRefusalDone, ResponseFunctionCallArgumentsDelta,
	// ResponseFunctionCallArgumentsDone, ResponseReasoningSummaryPartAdded, ResponseReasoningSummaryPartDone,
	// ResponseReasoningSummaryTextDelta, ResponseReasoningSummaryTextDone, ResponseReasoningDelta,
	// ResponseReasoningDone, ResponseReasoningSummaryDelta, ResponseReasoningSummaryDone, ResponseOutputTextAnnotationAdded
	ItemID string `json:"item_id,omitzero"`

	// Type == ResponseContentPartAdded, ResponseContentPartDone, ResponseReasoningSummaryPartAdded,
	// ResponseReasoningSummaryPartDone
	Part Content `json:"part,omitzero"`

	// Type == ResponseOutputTextDelta, ResponseRefusalDelta, ResponseFunctionCallArgumentsDelta,
	// ResponseReasoningSummaryTextDelta, ResponseReasoningDelta, ResponseReasoningSummaryDelta
	Delta string `json:"delta,omitzero"`

	// Type == ResponseOutputTextDone, ResponseReasoningSummaryTextDone, ResponseReasoningDone,
	// ResponseReasoningSummaryDone
	Text string `json:"text,omitzero"`

	// Type == ResponseRefusalDone
	Refusal string `json:"refusal,omitzero"`

	// Type == ResponseFunctionCallArgumentsDone
	Arguments string `json:"arguments,omitzero"`

	// Type == ResponseReasoningSummaryPartAdded, ResponseReasoningSummaryPartDone,
	// ResponseReasoningSummaryTextDelta, ResponseReasoningSummaryTextDone, ResponseReasoningSummaryDelta,
	// ResponseReasoningSummaryDone
	SummaryIndex int64 `json:"summary_index,omitzero"`

	// Type == ResponseOutputTextAnnotationAdded
	Annotation      Annotation `json:"annotation,omitzero"`
	AnnotationIndex int64      `json:"annotation_index,omitzero"`

	// Type == ResponseError
	Code    string `json:"code,omitzero"`
	Message string `json:"message,omitzero"`
	Param   string `json:"param,omitzero"`

	Logprobs []Logprobs `json:"logprobs,omitzero"` // TODO: I believe this is incorrect.

	/* TODO
	ResponseFileSearchCallCompleted
	ResponseFileSearchCallInProgress
	ResponseFileSearchCallSearching
	ResponseFunctionCallArgumentsDelta
	ResponseFunctionCallArgumentsDone
	ResponseImageGenerationCallCompleted
	ResponseImageGenerationCallGenerating
	ResponseImageGenerationCallInProgress
	ResponseImageGenerationCallPartialImage
	ResponseMCPCallArgumentsDelta
	ResponseMCPCallArgumentsDone
	ResponseMCPCallCompleted
	ResponseMCPCallFailed
	ResponseMCPCallInProgress
	ResponseMCPListToolsCompleted
	ResponseMCPListToolsFailed
	ResponseMCPListToolsInProgress
	ResponseWebSearchCallCompleted
	ResponseWebSearchCallInProgress
	ResponseWebSearchCallSearching
	*/
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

// https://platform.openai.com/docs/api-reference/files/object
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
	CustomID string   `json:"custom_id"`
	Method   string   `json:"method"` // "POST"
	URL      string   `json:"url"`    // "/v1/chat/completions", "/v1/embeddings", "/v1/completions", "/v1/responses"
	Body     Response `json:"body"`
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
		StatusCode int      `json:"status_code"`
		RequestID  string   `json:"request_id"` // To use when contacting support
		Body       Response `json:"body"`
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

// https://platform.openai.com/docs/api-reference/models/object
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

// ErrorResponse represents an error response from the OpenAI API.
type ErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
		Param   string `json:"param"`
	} `json:"error"`
}

func (e *ErrorResponse) String() string {
	return fmt.Sprintf("openai responses error: %s (type: %s, code: %s)", e.Error.Message, e.Error.Type, e.Error.Code)
}

//

// Client is a client for the OpenAI Responses API.
type Client struct {
	base.ProviderGen[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]
}

// New creates a new client to talk to the OpenAI Responses API.
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
		ProviderGen: base.ProviderGen[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.openai.com/v1/responses",
			GenStreamURL:         "https://api.openai.com/v1/responses",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "openairesponses",
				APIKeyURL:    "", // OpenAI error message prints the api key URL already.
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
		var created base.Time
		for _, mdl := range mdls {
			m := mdl.(*Model)
			if cheap {
				if strings.HasSuffix(m.ID, "-nano") && (created == 0 || m.Created < created) {
					// For the cheapest, we want the oldest model as it is generally cheaper.
					created = m.Created
					c.Model = m.ID
				}
			} else if good {
				if strings.HasSuffix(m.ID, "-mini") && (created == 0 || m.Created > created) {
					// For the greatest, we want the newest model as it is generally better.
					created = m.Created
					c.Model = m.ID
				}
			} else {
				if strings.HasSuffix(m.ID, "-pro") && (created == 0 || m.Created > created) {
					// For the greatest, we want the newest model as it is generally better.
					created = m.Created
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

// Scoreboard implements genai.ProviderScoreboard.
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
		Prompt: msg.AsText(),
		Model:  c.Model,
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
		req.ResponseFormat = "b64_json"
	case "dall-e-2":
		// We assume dall-e-2 is only used for smoke testing, so use the smallest image.
		req.Size = "256x256"
		// Maximum prompt length is 1000 characters.
		// Since we assume this is only for testing, silently cut it off.
		if len(req.Prompt) > 1000 {
			req.Prompt = req.Prompt[:1000]
		}
		req.ResponseFormat = "b64_json"
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

// processStreamPackets processes stream packets for the OpenAI Responses API.
// This is a placeholder - will be implemented when GenStream is added.
func processStreamPackets(ch <-chan ResponseStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()

	pendingToolCall := genai.ToolCall{}
	for pkt := range ch {
		f := genai.ContentFragment{}
		switch pkt.Type {
		case ResponseCreated, ResponseInProgress:
		case ResponseCompleted:
			result.InputTokens = pkt.Response.Usage.InputTokens
			result.InputCachedTokens = pkt.Response.Usage.InputTokensDetails.CachedTokens
			result.OutputTokens = pkt.Response.Usage.OutputTokens
			if len(pkt.Response.Output) == 0 {
				// TODO: Likely failed.
				return fmt.Errorf("no output: %#v", pkt)
			}
			// TODO: OpenAI supports "multiple messages" as output.
			result.FinishReason = genai.FinishedStop
			for i := range pkt.Response.Output {
				msg := pkt.Response.Output[i]
				switch msg.Status {
				case "":
				case "completed":
					if msg.Type == MessageFunctionCall {
						result.FinishReason = genai.FinishedToolCalls
					}
				case "in_progress":
				case "incomplete":
					result.FinishReason = genai.FinishedLength
				case "failed":
					return fmt.Errorf("failed: %#v", pkt)
				default:
					return fmt.Errorf("unknown status %q: %#v", msg.Status, pkt)
				}
			}
		case ResponseFailed:
			return fmt.Errorf("response failed: %s", pkt.Response.Error.Message)
		case ResponseIncomplete:
			result.InputTokens = pkt.Response.Usage.InputTokens
			result.OutputTokens = pkt.Response.Usage.OutputTokens
			result.FinishReason = genai.FinishedLength // Likely reason for incomplete
		case ResponseOutputTextDelta:
			f.TextFragment = pkt.Delta
		case ResponseOutputTextDone:
			// Unnecessary, we captured the text via ResponseOutputTextDelta.
		case ResponseOutputItemAdded:
			switch pkt.Item.Type {
			case MessageMessage:
				// Unnecessary.
			case MessageFunctionCall:
				pendingToolCall.Name = pkt.Item.Name
				pendingToolCall.ID = pkt.Item.CallID
			case MessageReasoning:
				var bits []string
				for i := range pkt.Item.Summary {
					bits = append(bits, pkt.Item.Summary[i].Text)
				}
				f.ThinkingFragment = strings.Join(bits, "")
			default:
				return fmt.Errorf("implement item: %q", pkt.Item.Type)
			}
		case ResponseOutputItemDone:
			// Unnecessary.
		case ResponseContentPartAdded:
			switch pkt.Part.Type {
			case ContentOutputText:
				if len(pkt.Part.Annotations) > 0 {
					return fmt.Errorf("implement citations: %#v", pkt.Part.Annotations)
				}
				if len(pkt.Part.Text) > 0 {
					return fmt.Errorf("unexpected text: %q", pkt.Part.Text)
				}
			case ContentRefusal:
				return fmt.Errorf("implement part: %q", pkt.Part.Type)
			default:
				return fmt.Errorf("implement part: %q", pkt.Part.Type)
			}
		case ResponseContentPartDone:
			// Unnecessary, as we already streamed the content in ResponseContentPartAdded.
		case ResponseRefusalDelta:
			// TODO: It's not an error.
			return fmt.Errorf("refused: %s", pkt.Delta)
		case ResponseRefusalDone:
			// TODO: It's not an error.
			return fmt.Errorf("refused: %s", pkt.Refusal)
		case ResponseFunctionCallArgumentsDelta:
			// Unnecessary. The content is sent in ResponseFunctionCallArgumentsDone.
		case ResponseFunctionCallArgumentsDone:
			pendingToolCall.Arguments = pkt.Arguments
			f.ToolCall = pendingToolCall
			pendingToolCall = genai.ToolCall{}
		case ResponseError:
			return fmt.Errorf("error: %s", pkt.Message)
		default:
			return fmt.Errorf("implement packet: %q", pkt.Type)
		}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	if !pendingToolCall.IsZero() {
		return errors.New("unexpected pending tool call")
	}
	return nil
}

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
