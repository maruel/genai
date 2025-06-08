// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package mistral implements a client for the Mistral API.
//
// It is described at https://docs.mistral.ai/api/
package mistral

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Mistral.
//
// # Warnings
//
//   - Mistral supports more than what is exposed by the provider.
//   - Tool calling is excellent and unbiased for non "mini" models.
//   - PDF doesn't support inline document while images do.
//   - Rate limit depends on your tier: https://docs.mistral.ai/deployment/laplateforme/tier/
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"ministral-3b-latest",
				"codestral-2501",
				"codestral-latest",
				"devstral-small-2505",
				"devstral-small-latest",
				"ministral-3b-2410",
				"ministral-8b-2410",
				"ministral-8b-latest",
				"mistral-large-pixtral-2411",
				"mistral-saba-2502",
				"mistral-saba-latest",
				"mistral-tiny",
				"mistral-tiny-2407",
				"mistral-tiny-latest",
				"open-mistral-7b",
				"open-mistral-nemo",
				"open-mistral-nemo-2407",
				"open-mixtral-8x22b",
				"open-mixtral-8x22b-2404",
				"open-mixtral-8x7b",
			},
			GenSync: &genai.FunctionalityText{
				Tools:      genai.True,
				BiasedTool: genai.True,
				JSON:       true,
				JSONSchema: true,
			},
			GenStream: &genai.FunctionalityText{
				Tools:      genai.True,
				BiasedTool: genai.True,
				JSON:       true,
				JSONSchema: true,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityImage, genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"mistral-small-latest",
				"mistral-small",
				"mistral-small-2501",
				//"mistral-small-2503",
				"mistral-large-2411",
				"mistral-large-latest",
				"mistral-medium",
				"mistral-medium-2505",
				"mistral-medium-latest",
				"pixtral-12b",
				"pixtral-12b-2409",
				"pixtral-12b-latest",
				"pixtral-large-2411",
				"pixtral-large-latest",
			},
			GenSync: &genai.FunctionalityText{
				InputInline:    true,
				InputURL:       true,
				Tools:          genai.True,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
			GenStream: &genai.FunctionalityText{
				InputInline:    true,
				InputURL:       true,
				Tools:          genai.True,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityPDF, genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"mistral-small-2503",
				/*
					"mistral-small",
					"mistral-small-2501",
					"mistral-small-latest",
					"mistral-large-2411",
					"mistral-large-latest",
					"mistral-medium",
					"mistral-medium-2505",
					"mistral-medium-latest",
				*/
				"mistral-ocr-2503",
				"mistral-ocr-2505",
				"mistral-ocr-latest",
			},
			GenSync: &genai.FunctionalityText{
				InputURL:       true,
				Tools:          genai.True,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
			GenStream: &genai.FunctionalityText{
				InputURL:       true,
				Tools:          genai.True,
				IndecisiveTool: genai.True,
				JSON:           true,
				JSONSchema:     true,
			},
		},
	},
}

// https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
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
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.OptionsText:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.RandomSeed = v.Seed
				if v.TopK != 0 {
					unsupported = append(unsupported, "TopK")
				}
				c.Stop = v.Stop
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json_schema"
					// Mistral requires a name.
					c.ResponseFormat.JSONSchema.Name = "response"
					c.ResponseFormat.JSONSchema.Strict = true
					c.ResponseFormat.JSONSchema.Schema = jsonschema.Reflect(v.DecodeAs)
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
	}

	if err := msgs.Validate(); err != nil {
		errs = append(errs, err)
	} else {
		offset := 0
		if sp != "" {
			offset = 1
		}
		c.Messages = make([]Message, len(msgs)+offset)
		if sp != "" {
			c.Messages[0].Role = "system"
			c.Messages[0].Content = []Content{{Type: ContentText, Text: sp}}
		}
		for i := range msgs {
			if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			}
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

// https://docs.mistral.ai/api/#tag/chat/operation/chat_completion_v1_chat_completions_post
type Message struct {
	Role       string     `json:"role"` // "system", "assistant", "user", "tool"
	Content    []Content  `json:"content,omitzero"`
	Prefix     bool       `json:"prefix,omitzero"`
	ToolCalls  []ToolCall `json:"tool_calls,omitzero"`
	ToolCallID string     `json:"tool_call_id,omitzero"`
	Name       string     `json:"name,omitzero"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
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
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Name = in.ToolCallResults[0].Name
		// Mistral supports images urls!!
		m.Content = []Content{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
	}
	return nil
}

type Content struct {
	Type ContentType `json:"type"`

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "reference"
	ReferenceIDs []int64 `json:"reference_ids,omitzero"`

	// Type == "document_url"
	DocumentURL  string `json:"document_url,omitzero"`
	DocumentName string `json:"document_name,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL    string `json:"url,omitzero"`
		Detail string `json:"detail,omitzero"` // undocumented, likely "auto" like OpenAI
	} `json:"image_url,omitzero"`
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
	case (in.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
		c.Type = ContentImageURL
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	case mimeType == "application/pdf":
		c.Type = ContentDocumentURL
		if in.URL == "" {
			// Inexplicably, Mistral supports inline images but not PDF.
			return errors.New("unsupported inline document")
		}
		c.DocumentName = in.GetFilename()
		c.DocumentURL = in.URL
	default:
		return fmt.Errorf("unsupported mime type %s", mimeType)
	}
	return nil
}

type ContentType string

const (
	ContentText        ContentType = "text"
	ContentReference   ContentType = "reference"
	ContentDocumentURL ContentType = "document_url"
	ContentImageURL    ContentType = "image_url"
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
	ID      string `json:"id"`
	Object  string `json:"object"` // "chat.completion"
	Model   string `json:"model"`
	Created Time   `json:"created"`
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
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type MessageResponse struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	Prefix    bool       `json:"prefix"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "system", "assistant", "user":
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
	if m.Content != "" {
		out.Contents = []genai.Content{{Text: m.Content}}
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

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Mistral doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

type ChatStreamChunkResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "chat.completion.chunk"
	Created Time   `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index int64 `json:"index"`
		Delta struct {
			Role      genai.Role `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason FinishReason `json:"finish_reason"`
		Logprobs     struct{}     `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

// Time is a JSON encoded unix timestamp.
type Time int64

func (t *Time) AsTime() time.Time {
	return time.Unix(int64(*t), 0)
}

// https://docs.mistral.ai/api/#tag/models/operation/retrieve_model_v1_models__model_id__get
type Model struct {
	ID           string `json:"id"`
	Object       string `json:"object"`
	Created      Time   `json:"created"`
	OwnedBy      string `json:"owned_by"`
	Capabilities struct {
		CompletionChat  bool `json:"completion_chat"`
		CompletionFim   bool `json:"completion_fim"`
		FunctionCalling bool `json:"function_calling"`
		FineTuning      bool `json:"fine_tuning"`
		Vision          bool `json:"vision"`
		Classification  bool `json:"classification"`
	} `json:"capabilities"`
	Name                    string   `json:"name"`
	Description             string   `json:"description"`
	MaxContextLength        int64    `json:"max_context_length"`
	Aliases                 []string `json:"aliases"`
	Deprecation             string   `json:"deprecation"`
	DefaultModelTemperature float64  `json:"default_model_temperature"`
	Type                    string   `json:"type"`
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
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
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

func (er *ErrorResponse) String() string {
	out := er.Type
	if out != "" {
		out += ": "
	}
	if s := er.Detail.String(); s != "" {
		return "error " + out + s
	}
	return "error " + out + er.Message.Detail.String()
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

func (er *ErrorDetail) String() string {
	if er.Type == "" && len(er.Loc) == 0 {
		// This was actually a string
		return er.Msg
	}
	return fmt.Sprintf("%s: %s at %s", er.Type, er.Msg, er.Loc)
}

type ErrorDetails []ErrorDetail

func (er *ErrorDetails) String() string {
	out := ""
	for _, e := range *er {
		out += e.String()
	}
	return out
}

type ErrorMessage struct {
	Detail ErrorDetails `json:"detail"`
}

func (er *ErrorMessage) UnmarshalJSON(d []byte) error {
	s := ""
	if err := json.Unmarshal(d, &s); err == nil {
		er.Detail = ErrorDetails{{Msg: s}}
		return nil
	}
	var x struct {
		Detail ErrorDetails `json:"detail"`
	}
	if err := json.Unmarshal(d, &x); err != nil {
		return err
	}
	er.Detail = x.Detail
	return nil
}

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// TODO:
// https://codestral.mistral.ai/v1/fim/completions
// https://codestral.mistral.ai/v1/chat/completions

// New creates a new client to talk to the Mistral platform API.
//
// If apiKey is not provided, it tries to load it from the MISTRAL_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://console.mistral.ai/api-keys or https://console.mistral.ai/codestral
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.mistral.ai/getting-started/models/models_overview/
//
// wrapper can be used to throttle outgoing requests, record calls, etc. It defaults to base.DefaultTransport.
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
func New(apiKey, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://console.mistral.ai/api-keys"
	if apiKey == "" {
		if apiKey = os.Getenv("MISTRAL_API_KEY"); apiKey == "" {
			return nil, errors.New("mistral API key is required; get one at " + apiKeyURL)
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	return &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.mistral.ai/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "mistral",
				APIKeyURL:    apiKeyURL,
				ClientJSON: httpjson.Client{
					Lenient: internal.BeLenient,
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
							Transport: t,
						},
					},
				},
			},
		},
	}, nil
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.mistral.ai/api/#tag/models
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, "https://api.mistral.ai/v1/models")
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
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
			result.InputTokens = pkt.Usage.PromptTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		f := genai.ContentFragment{TextFragment: pkt.Choices[0].Delta.Content}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		// Mistral is one of the rare provider that can stream multiple tool calls all at once. It's probably
		// because it's buffering server-side.
		for i := range pkt.Choices[0].Delta.ToolCalls {
			f := genai.ContentFragment{}
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

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
