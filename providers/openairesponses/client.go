// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openairesponses implements a client for the OpenAI Responses API.
//
// It is described at https://platform.openai.com/docs/api-reference/responses/create
package openairesponses

// See official client at http://pkg.go.dev/github.com/openai/openai-go/responses

import (
	"errors"
	"fmt"
	"net/http"
	"os"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for OpenAI Responses API.
var Scoreboard = genai.Scoreboard{
	Country:      "US",
	DashboardURL: "https://platform.openai.com/usage",
	Scenarios: []genai.Scenario{
		{
			Models: []string{
				"gpt-4-turbo",
				"gpt-4-turbo-2024-04-09",
				"gpt-4-turbo-preview",
				"gpt-4",
				"gpt-4-0125-preview",
				"gpt-4-0613",
				"gpt-4-1106-preview",
				"gpt-3.5-turbo",
				"gpt-3.5-turbo-0125",
				"gpt-3.5-turbo-1106",
				"gpt-3.5-turbo-16k",
				"gpt-3.5-turbo-instruct",
				"gpt-3.5-turbo-instruct-0914",
			},
			In:  map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:       genai.True,
				BiasedTool:  genai.True,
				NoMaxTokens: true, // Technically not true but it requires at least 16 tokens and the smoke test is 3.
				// IndecisiveTool: genai.True,
				NoStopSequence: true,
				JSON:           true,
			},
			/*
				GenStream: &genai.FunctionalityText{
					Tools:          genai.True,
					BiasedTool:     genai.True,
					IndecisiveTool: genai.True,
					NoStopSequence: true,
					JSON:           true,
				},
			*/
		},
	},
}

// ResponseStreamChunkResponse represents a streaming response chunk.
// This is a placeholder - will be implemented when GenStream is added.
type ResponseStreamChunkResponse struct {
	// TODO: Implement when GenStream support is added
}

// Response represents a request to the OpenAI Responses API.
type Response struct {
	Model              string            `json:"model"`
	Background         bool              `json:"background"`
	Instructions       string            `json:"instructions,omitzero"`
	MaxOutputTokens    int64             `json:"max_output_tokens,omitzero"`
	Metadata           map[string]string `json:"metadata,omitzero"`
	ParallelToolCalls  bool              `json:"parallel_tool_calls,omitzero"`
	PreviousResponseID string            `json:"previous_response_id,omitzero"`
	Reasoning          ReasoningConfig   `json:"reasoning,omitzero"`
	ServiceTier        string            `json:"service_tier,omitzero"`
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
	TopP       float64 `json:"top_p,omitzero"`
	ToolChoice string  `json:"tool_choice,omitzero"` // "none", "auto", "required"
	Truncation string  `json:"truncation,omitzero"`  // "disabled"
	Tools      []Tool  `json:"tools,omitzero"`
	User       string  `json:"user,omitzero"`

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
		return res, fmt.Errorf("multiple outputs not supported: %#v", r.Output)
	}
	for _, output := range r.Output {
		if err := output.To(&res.Message); err != nil {
			return res, err
		}
	}
	if r.IncompleteDetails.Reason != "" {
		res.FinishReason = genai.FinishReason(r.IncompleteDetails.Reason)
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
		r.Text.Format.Schema = jsonschema.Reflect(v.DecodeAs)
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
	Effort  string `json:"effort,omitzero"`  // "low", "medium", "high"
	Summary string `json:"summary,omitzero"` // "auto", "concise", "detailed"
}

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
		for j, content := range msg.Contents {
			if content.Text != "" {
				m.Content[j] = Content{Type: ContentInputText, Text: content.Text}
			} else if content.Document != nil {
				// Handle document content
				// TODO: Implement proper document/image content conversion
				m.Content[j] = Content{Type: ContentInputImage, Detail: "auto"}
				return fmt.Errorf("implement me")
			} else {
				return fmt.Errorf("unsupported content type")
			}
		}
		return nil
	}
	return fmt.Errorf("implement me: %#v", msg)
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
		out.Contents = make([]genai.Content, len(m.Content))
		for i := range m.Content {
			if err := m.Content[i].To(&out.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		}
	case MessageReasoning:
		return errors.New("implement reasoning")
	case MessageFunctionCall:
		out.ToolCalls = []genai.ToolCall{{ID: m.CallID, Name: m.Name, Arguments: m.Arguments}}
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
	FileData []byte `json:"file_data,omitzero"` // TODO: confirm if base64
	Filename string `json:"filename,omitzero"`

	// Type == ContentOutputText
	Annotations []Annotation `json:"annotations,omitzero"`
	Logprobs    struct {
		Bytes       []int   `json:"bytes,omitzero"`
		Token       string  `json:"token,omitzero"`
		Logprob     float64 `json:"logprob,omitzero"`
		TopLogprobs []struct {
			Bytes   []int   `json:"bytes,omitzero"`
			Token   string  `json:"token,omitzero"`
			Logprob float64 `json:"logprob,omitzero"`
		} `json:"top_logprobs,omitzero"`
	} `json:"logprobs,omitzero"`

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
	CachedTokens    int `json:"cached_tokens,omitzero"`
	AudioTokens     int `json:"audio_tokens,omitzero"`
	ReasoningTokens int `json:"reasoning_tokens,omitzero"`
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
// base.PreferredSOTA for a state-of-the-art model.
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
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "openairesponses",
				APIKeyURL:    "", // OpenAI error message prints the api key URL already.
				ClientJSON: httpjson.Client{
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
	return c, err
}

// Scoreboard implements genai.ProviderScoreboard.
func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

// processStreamPackets processes stream packets for the OpenAI Responses API.
// This is a placeholder - will be implemented when GenStream is added.
func processStreamPackets(ch <-chan ResponseStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	// TODO: Implement when GenStream support is added
	return fmt.Errorf("streaming not yet implemented for OpenAI Responses API")
}

// Interface compliance checks
var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
