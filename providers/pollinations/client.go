// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package pollinations implements a client for the Pollinations API.
//
// It is described at https://github.com/pollinations/pollinations/blob/master/APIDOCS.md
package pollinations

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Pollinations.
//
// # Warnings
//
//   - This is a completely free provider so you get what you pay for. I would recommend against using this
//     provider with any private data.
//   - Pollinations is a router to other backends, so it inherits the drawback of each sub-provider.
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"llama-scout",
				"deepseek",
				"evil",
				"grok",
				"llama",
				"mistral",
				"openai-fast",
				"qwen-coder",
			},
			GenSync: &genai.FunctionalityText{
				NoMaxTokens:    true,
				NoStopSequence: true,
				Tools:          genai.Flaky,
				IndecisiveTool: genai.True,
				JSON:           true,
			},
			GenStream: &genai.FunctionalityText{
				BrokenTokenUsage: true,
				NoMaxTokens:      true,
				NoStopSequence:   true,
				Tools:            genai.Flaky,
				IndecisiveTool:   genai.True,
				JSON:             true,
			},
		},
		/* GenStream is particularly broken.
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"deepseek-reasoning",
			},
			GenSync: genai.Functionality{
				Thinking:           true,
				NoStopSequence: true,
			},
			GenStream: genai.Functionality{
				Thinking:           false, // Upstream parsing is broken
				NoStopSequence: true,
			},
		},
		*/
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityImage},
			Models: []string{
				"flux",
				"gptimage",
				"turbo",
			},
			GenDoc: &genai.FunctionalityDoc{
				OutputInline:       true,
				BrokenTokenUsage:   true,
				BrokenFinishReason: true,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityImage, genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"openai",
				"openai-large",
			},
			GenSync: &genai.FunctionalityText{
				InputInline:    true,
				InputURL:       true,
				NoMaxTokens:    true,
				NoStopSequence: true,
				Tools:          genai.True,
				BiasedTool:     genai.True,
				JSON:           true,
			},
			GenStream: &genai.FunctionalityText{
				InputInline:      true,
				InputURL:         true,
				BrokenTokenUsage: true,
				NoMaxTokens:      true,
				NoStopSequence:   true,
				Tools:            genai.True,
				BiasedTool:       genai.True,
				JSON:             true,
			},
		},
		/*
			// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#speech-to-text-capabilities-audio-input-%EF%B8%8F
			// Getting error: invalid_value: azure-openai error: This model does not support image_url content.; provider:azure-openai
			{
				In:  []genai.Modality{genai.ModalityAudio, genai.ModalityText},
				Out: []genai.Modality{genai.ModalityText},
				Models: []string{
					"openai-audio",
				},
				GenSync: genai.Functionality{
					InputInline:             true,
					NoMaxTokens:        true,
					JSON:               true,
				},
				GenStream: genai.Functionality{
					InputInline:             true,
					BrokenTokenUsage:   true,
					NoMaxTokens:        true,
					JSON:               true,
				},
			},
		*/
		/*
			// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#generate-audio-api-
			// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#text-to-speech-post---openai-compatible-%EF%B8%8F%EF%B8%8F
			{
				In:  []genai.Modality{genai.ModalityText},
				Out: []genai.Modality{genai.ModalityAudio},
				Models: []string{
					"openai-audio",
				},
				GenSync: genai.Functionality{
					InputInline:             true,
					NoMaxTokens:        true,
					JSON:               true,
				},
				GenStream: genai.Functionality{
					InputInline:             true,
					BrokenTokenUsage:   true,
					NoMaxTokens:        true,
					JSON:               true,
				},
			},
			//*/
	},
}

// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#text--multimodal-openai-compatible-post-%EF%B8%8F%EF%B8%8F
//
// The structure is severely underdocumented.
type ChatRequest struct {
	Messages       []Message `json:"messages"`
	MaxTokens      int64     `json:"max_tokens,omitzero"`
	Model          string    `json:"model"`
	Seed           int64     `json:"seed,omitzero"`
	Stream         bool      `json:"stream"`
	ResponseFormat struct {
		Type string `json:"type,omitzero"` // "json_object"
	} `json:"response_format,omitzero"`
	Tools           []Tool          `json:"tools,omitzero"`
	ToolChoice      string          `json:"tool_choice,omitzero"` // "none", "auto", "required", or struct {"type": "function", "function": {"name": "my_function"}}
	Private         bool            `json:"private,omitzero"`     // Set to true to prevent the response from appearing in the public feed.
	ReasoningEffort ReasoningEffort `json:"reasoning_format,omitzero"`

	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#text-to-speech-post---openai-compatible-%EF%B8%8F%EF%B8%8F
	Modalities []string `json:"modalities,omitzero"`
	Audio      struct {
		Voice  string `json:"voice,omitzero"`
		Format string `json:"format,omitzero"` // "pcm16"
	} `json:"audio,omitzero"`

	// These are not documented at all.
	Temperature   float64  `json:"temperature,omitzero"`
	TopP          float64  `json:"top_p,omitzero"` // [0, 1.0]
	Stop          []string `json:"stop,omitzero"`  // Up to 4 sequences
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	c.Private = true // Not sure why we'd want to broadcast?
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsText:
			unsupported, errs = c.initOptions(v, model)
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
		c.Messages[0].Role = "system"
		c.Messages[0].Content = []Content{{Type: ContentText, Text: sp}}
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
	c.StreamOptions.IncludeUsage = true
}

func (c *ChatRequest) initOptions(v *genai.OptionsText, model string) ([]string, []error) {
	var errs []error
	var unsupported []string
	c.MaxTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	c.Seed = v.Seed
	if v.TopK != 0 {
		unsupported = append(unsupported, "TopK")
	}
	c.Stop = v.Stop
	if v.DecodeAs != nil {
		errs = append(errs, errors.New("unsupported option DecodeAs"))
	}
	if v.ReplyAsJSON {
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
		}
	}
	return unsupported, errs
}

// ReasoningEffort is the effort the model should put into reasoning. Default is Medium.
type ReasoningEffort string

const (
	ReasoningEffortLow    ReasoningEffort = "low"
	ReasoningEffortMedium ReasoningEffort = "medium"
	ReasoningEffortHigh   ReasoningEffort = "high"
)

type Message struct {
	Role       string     `json:"role"` // "system", "assistant", "user"
	Content    Contents   `json:"content,omitzero"`
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
		m.Content = make(Contents, 0, len(in.Contents))
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

// Contents exists to marshal single content text block as a string.
type Contents []Content

func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal(([]Content)(*c))
}

type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "image_url"
	ImageURL struct {
		URL string `json:"url,omitzero"` // URL or base64 encoded image
	} `json:"image_url,omitzero"`

	// Type == "input_audio"
	InputAudio struct {
		Data   []byte `json:"data,omitzero"`   // base64 encoded audio
		Format string `json:"format,omitzero"` // "wav"
	} `json:"input_audio,omitzero"`
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
	case strings.HasPrefix(mimeType, "audio/"):
		// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#speech-to-text-capabilities-audio-input-%EF%B8%8F
		c.Type = ContentImageURL
		c.InputAudio.Data = data
		switch mimeType {
		case "audio/mpeg":
			c.InputAudio.Format = "mp3"
		default:
			return fmt.Errorf("implement mime type %s conversion", mimeType)
		}
	case strings.HasPrefix(mimeType, "image/") || in.URL != "":
		c.Type = ContentImageURL
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
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

type ContentType string

const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
	ContentAudio    ContentType = "input_audio"
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
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Index    int64  `json:"index,omitzero"`
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

type ChatResponse struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"` // "chat.completion"
	Created base.Time `json:"created"`
	Model   string    `json:"model"` // The actual model name, which is likely different fro the alias.
	Choices []struct {
		Index                int64               `json:"index"`
		Message              MessageResponse     `json:"message"`
		FinishReason         FinishReason        `json:"finish_reason"`
		StopReason           struct{}            `json:"stop_reason"`
		Logprobs             struct{}            `json:"logprobs"`
		ContentFilterResults ContentFilterResult `json:"content_filter_results"`
	} `json:"choices"`
	Usage               Usage                `json:"usage"`
	PromptFilterResults []PromptFilterResult `json:"prompt_filter_results"`
	SystemFingerprint   string               `json:"system_fingerprint"`
	PromptLogprobs      struct{}             `json:"prompt_logprobs"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	// There's a "X-Cache" HTTP response header that says when the whole request was cached.
	out := genai.Result{
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

type FinishReason string

const (
	FinishStop      FinishReason = "stop"
	FinishToolCalls FinishReason = "tool_calls"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishToolCalls:
		return genai.FinishedToolCalls
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

type PromptFilterResult struct {
	PromptIndex int64 `json:"prompt_index"`
	// One of the following is set.
	ContentFilterResults ContentFilterResult `json:"content_filter_results"`
	ContentFilterResult  ContentFilterResult `json:"content_filter_result"`
}

type Usage struct {
	PromptTokens            int64 `json:"prompt_tokens"`
	AudioPromptTokens       int64 `json:"audio_prompt_tokens"`
	CompletionTokens        int64 `json:"completion_tokens"`
	TotalTokens             int64 `json:"total_tokens"`
	CompletionTokensDetails struct {
		AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
		AudioTokens              int64 `json:"audio_tokens"`
		ReasoningTokens          int64 `json:"reasoning_tokens"`
		RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
	} `json:"completion_tokens_details"`
	PromptTokensDetails struct {
		AudioTokens  int64 `json:"audio_tokens"`
		CachedTokens int64 `json:"cached_tokens"`
		ImageTokens  int64 `json:"image_tokens"`
		TextTokens   int64 `json:"text_tokens"`
	} `json:"prompt_tokens_details"`
}

type MessageResponse struct {
	Role             genai.Role `json:"role"`
	ReasoningContent string     `json:"reasoning_content"`
	Content          string     `json:"content"`
	ToolCalls        []ToolCall `json:"tool_calls"`
	Annotations      []struct{} `json:"annotations"`
	Refusal          struct{}   `json:"refusal"`
	Audio            struct {
		Data []byte `json:"data"`
	} `json:"audio"`
}

func (m *MessageResponse) To(out *genai.Message) error {
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
	if m.Content != "" {
		out.Contents = append(out.Contents, genai.Content{Text: m.Content})
	}
	if m.ReasoningContent != "" {
		// Paper over broken "deepseek".
		if len(out.Contents) == 1 && out.Contents[0].Text != "" {
			out.Contents = append(out.Contents, genai.Content{Thinking: m.ReasoningContent})
		} else {
			out.Contents = append(out.Contents, genai.Content{Text: m.ReasoningContent})
		}
	}
	if len(m.Audio.Data) != 0 {
		out.Contents = append(out.Contents, genai.Content{Filename: "sound.wav", Document: bytes.NewReader(m.Audio.Data)})
	}
	return nil
}

type ChatStreamChunkResponse struct {
	ID      string    `json:"id"`      //
	Object  string    `json:"object"`  // "chat.completion.chunk"
	Created base.Time `json:"created"` //
	Model   string    `json:"model"`   // Original model full name
	Choices []struct {
		ContentFilterResults ContentFilterResult `json:"content_filter_results"`
		Index                int64               `json:"index"`
		Logprobs             struct{}            `json:"logprobs"`
		FinishReason         FinishReason        `json:"finish_reason"`
		StopReason           struct{}            `json:"stop_reason"`
		MatchedStop          int64               `json:"matched_stop"`
		Delta                struct {
			Role             string     `json:"role"`
			Content          string     `json:"content"`
			ReasoningContent string     `json:"reasoning_content"`
			ToolCalls        []ToolCall `json:"tool_calls"`
			Refusal          struct{}   `json:"refusal"`
		} `json:"delta"`
	} `json:"choices"`
	PromptFilterResults []PromptFilterResult `json:"prompt_filter_results"`
	SystemFingerprint   string               `json:"system_fingerprint"`
	Usage               Usage                `json:"usage"`
}

type ContentFilterResult struct {
	Custom_blocklists struct {
		Filtered bool `json:"filtered"`
		Details  []struct {
			BlocklistName string `json:"blocklist_name"`
		} `json:"details"`
	} `json:"custom_blocklists"`
	Hate struct {
		Filtered bool   `json:"filtered"`
		Severity string `json:"severity"`
	} `json:"hate"`
	Jailbreak struct {
		Filtered bool `json:"filtered"`
		Detected bool `json:"detected"`
	} `json:"jailbreak"`
	Protected_material_code struct {
		Filtered bool `json:"filtered"`
		Detected bool `json:"detected"`
	} `json:"protected_material_code"`
	Protected_material_text struct {
		Filtered bool `json:"filtered"`
		Detected bool `json:"detected"`
	} `json:"protected_material_text"`
	SelfHarm struct {
		Filtered bool   `json:"filtered"`
		Severity string `json:"severity"`
	} `json:"self_harm"`
	Sexual struct {
		Filtered bool   `json:"filtered"`
		Severity string `json:"severity"`
	} `json:"sexual"`
	Violence struct {
		Filtered bool   `json:"filtered"`
		Severity string `json:"severity"`
	} `json:"violence"`
}

type ImageModel string

func (i ImageModel) GetID() string {
	return string(i)
}

func (i ImageModel) String() string {
	return fmt.Sprintf("%s in:text out:image", string(i))
}

func (i ImageModel) Context() int64 {
	return 0
}

func (i ImageModel) Inputs() []string {
	return []string{"text"}
}

func (i ImageModel) Outputs() []string {
	return []string{"image"}
}

type ImageModelsResponse []ImageModel

func (r *ImageModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(*r))
	for i := range *r {
		models[i] = (*r)[i]
	}
	return models
}

type TextModel struct {
	Audio            bool     `json:"audio"`
	Aliases          string   `json:"aliases"`
	Description      string   `json:"description"`
	InputModalities  []string `json:"input_modalities"`
	Name             string   `json:"name"`
	OutputModalities []string `json:"output_modalities"`
	Provider         string   `json:"provider"`
	Reasoning        bool     `json:"reasoning"`
	Tools            bool     `json:"tools"`
	Uncensored       bool     `json:"uncensored"`
	Voices           []string `json:"voices"`
	Vision           bool     `json:"vision"`
}

func (t *TextModel) GetID() string {
	return t.Name
}

func (t *TextModel) String() string {
	var in []string
	if len(t.InputModalities) != 0 {
		in = make([]string, len(t.InputModalities))
		copy(in, t.InputModalities)
		sort.Strings(in)
	}
	if t.Tools {
		in = append(in, "tools")
	}
	var out []string
	if len(t.OutputModalities) != 0 {
		out = make([]string, len(t.OutputModalities))
		copy(out, t.OutputModalities)
		sort.Strings(out)
	}
	return fmt.Sprintf("%s in:%s; out:%s; provider:%s; %s",
		t.Name, strings.Join(in, ","), strings.Join(out, ","), t.Provider, t.Description)
}

func (t *TextModel) Context() int64 {
	return 0
}

func (t *TextModel) Inputs() []string {
	return t.InputModalities
}

func (t *TextModel) Outputs() []string {
	return t.OutputModalities
}

type TextModelsResponse []TextModel

func (r *TextModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(*r))
	for i := range *r {
		models[i] = &(*r)[i]
	}
	return models
}

//

type ErrorResponse struct {
	Error   string `json:"error"`
	Status  int64  `json:"status"`
	Details struct {
		Detail string `json:"detail"`
		Error  struct {
			Message string `json:"message"`
			Type    string `json:"type"`
			Param   string `json:"param"`
			Code    string `json:"code"`
		} `json:"error"`
		Message   string            `json:"message"`
		QueueInfo map[string]string `json:"queueInfo"`
		Timestamp string            `json:"timestamp"`
		Provider  string            `json:"provider"`
	} `json:"details"`

	Message    string   `json:"message"`
	Debug      struct{} `json:"debug"`
	TimingInfo []struct {
		Step      string `json:"step"`
		Timestamp int64  `json:"timestamp"`
	} `json:"timingInfo"`
	RequestId         string         `json:"requestId"`
	RequestParameters map[string]any `json:"requestParameters"`
}

func (er *ErrorResponse) String() string {
	suffix := ""
	if er.Details.Provider != "" {
		suffix = "; provider:" + er.Details.Provider
	}
	if er.Details.Error.Message != "" {
		// It already contains the Provider name.
		return fmt.Sprintf("error %s/%s%s: %s%s", er.Details.Error.Type, er.Details.Error.Param, er.Details.Error.Code, er.Details.Error.Message, suffix)
	}
	if er.Details.Message != "" {
		return fmt.Sprintf("error %s%s", er.Details.Message, suffix)
	}
	if er.Details.Detail != "" {
		return fmt.Sprintf("error %s%s", er.Details.Detail, suffix)
	}
	if er.Message != "" {
		return fmt.Sprintf("error %s %s%s", er.Error, er.Message, suffix)
	}
	return fmt.Sprintf("error %s%s", er.Error, suffix)
}

// Client implements genai.ProviderModel.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Pollinations platform API.
// The value for auth can be either an API key retrieved from https://auth.pollinations.ai/ or a referrer.
// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#referrer-
//
// auth is optional. Providing one, either via environment variable POLLINATIONS_API_KEY, will increase quota.
//
// Pass model base.PreferredCheap to use a good cheap model, base.PreferredGood for a good model or
// base.PreferredSOTA to use its SOTA model. Keep in mind that as providers cycle through new models, it's
// possible the model is not available anymore.
//
// wrapper can be used to throttle outgoing requests, record calls, etc. It defaults to base.DefaultTransport.
func New(auth, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	var h http.Header
	if auth == "" {
		auth = os.Getenv("POLLINATIONS_API_KEY")
	}
	if auth != "" {
		if strings.HasPrefix(auth, "http://") || strings.HasPrefix(auth, "https://") {
			h = http.Header{"Referrer": {auth}}
		} else {
			h = http.Header{"Authorization": {"Bearer " + auth}}
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://text.pollinations.ai/openai",
			ProcessStreamPackets: processStreamPackets,
			LieToolCalls:         true,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "pollinations",
				ClientJSON: httpjson.Client{
					Lenient: internal.BeLenient,
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    h,
							Transport: &roundtrippers.RequestID{Transport: t},
						},
					},
				},
			},
		},
	}
	if model == base.PreferredCheap || model == base.PreferredGood || model == base.PreferredSOTA {
		mdls, err := c.ListTextModels(context.Background())
		if err != nil {
			return nil, err
		}
		cheap := model == base.PreferredCheap
		good := model == base.PreferredGood
		c.Model = ""
		for _, mdl := range mdls {
			m := mdl.(*TextModel)
			if m.Audio || strings.HasSuffix(m.Name, "roblox") {
				continue
			}
			// This is meh.
			if cheap {
				if strings.HasPrefix(m.Name, "llama") {
					c.Model = m.Name
				}
			} else if good {
				if strings.HasPrefix(m.Name, "openai") {
					c.Model = m.Name
				}
			} else {
				if m.Reasoning {
					c.Model = m.Name
				}
			}
		}
		if c.Model == "" {
			return nil, errors.New("failed to find a model automatically")
		}
	}
	return c, nil
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

// GenDoc uses the text-to-image API to generate an image.
//
// Default rate limit is 0.2 QPS / IP.
func (c *Client) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#text-to-image-get-%EF%B8%8F
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
	qp := url.Values{}
	qp.Add("model", c.Model)
	switch v := opts.(type) {
	case *genai.OptionsImage:
		if v.Seed != 0 {
			// Defaults to 42 otherwise.
			qp.Add("seed", strconv.FormatInt(v.Seed, 10))
		}
		if v.Width != 0 {
			qp.Add("width", strconv.Itoa(v.Width))
		}
		if v.Height != 0 {
			qp.Add("height", strconv.Itoa(v.Height))
		}
	case *genai.OptionsText:
		// TODO: Deny most flags.
		if v.Seed != 0 {
			// Defaults to 42 otherwise.
			qp.Add("seed", strconv.FormatInt(v.Seed, 10))
		}
	default:
	}

	qp.Add("nologo", "true")
	qp.Add("private", "true") // "nofeed"
	qp.Add("enhance", "false")
	qp.Add("safe", "false")
	// qp.Add("negative_prompt", "worst quality, blurry")
	qp.Add("quality", "medium")
	for _, mc := range msg.Contents {
		if mc.Document != nil {
			return res, errors.New("inline document is not supported")
		}
		if mc.URL != "" {
			qp.Add("image", mc.URL)
		}
	}

	prompt := url.QueryEscape(msg.AsText())
	url := "https://image.pollinations.ai/prompt/" + url.PathEscape(prompt) + "?" + qp.Encode()
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return res, err
	}
	resp, err := c.ClientJSON.Client.Do(req)
	if err != nil {
		return res, err
	}
	if resp.StatusCode != 200 {
		_ = resp.Body.Close()
		return res, c.DecodeError(url, resp)
	}
	b, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil {
		return res, err
	}
	res.Role = genai.Assistant
	res.Contents = []genai.Content{{Document: &bb.BytesBuffer{D: b}}}
	if ct := resp.Header.Get("Content-Type"); strings.HasPrefix(ct, "image/jpeg") {
		res.Contents[0].Filename = "content.jpg"
	} else {
		return res, fmt.Errorf("unknown Content-Type: %s", ct)
	}
	return res, nil
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#list-available-image-models-
	var out []genai.Model
	img, err1 := c.ListImageGenModels(ctx)
	out = append(out, img...)
	txt, err2 := c.ListTextModels(ctx)
	out = append(out, txt...)
	if err1 == nil {
		err1 = err2
	}
	return out, err1
}

func (c *Client) ListImageGenModels(ctx context.Context) ([]genai.Model, error) {
	return base.ListModels[*ErrorResponse, *ImageModelsResponse](ctx, &c.Provider, "https://image.pollinations.ai/models")
}

func (c *Client) ListTextModels(ctx context.Context) ([]genai.Model, error) {
	return base.ListModels[*ErrorResponse, *TextModelsResponse](ctx, &c.Provider, "https://text.pollinations.ai/models")
}

func (c *Client) isImage(opts genai.Options) bool {
	// TODO: Use Scoreboard list.
	switch c.Model {
	case "flux", "gptimage", "turbo":
		return true
	default:
		return opts != nil && opts.Modality() == genai.ModalityImage
	}
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
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt)
		}
		f := genai.ContentFragment{
			TextFragment:     pkt.Choices[0].Delta.Content,
			ThinkingFragment: pkt.Choices[0].Delta.ReasoningContent,
		}
		// Pollinations streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
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
