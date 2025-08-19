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
	"slices"
	"sort"
	"strconv"
	"strings"
	"sync"
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

// Scoreboard for Pollinations.
//
// # Warnings
//
//   - This is a completely free provider so you get what you pay for. I would recommend against using this
//     provider with any private data.
//   - Pollinations is a router to other backends, so it inherits the drawback of each sub-provider.
var Scoreboard = scoreboard.Score{
	Country:      "DE",
	DashboardURL: "https://auth.pollinations.ai/",
	Scenarios: []scoreboard.Scenario{
		{
			Models: []string{"llamascout"},
			In:     map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				NoMaxTokens:    true,
				NoStopSequence: true,
				Tools:          scoreboard.Flaky,
				IndecisiveTool: scoreboard.Flaky,
				JSON:           true,
				Seed:           true,
			},
			// JSON generated is often bad.
			GenStream: &scoreboard.FunctionalityText{
				NoMaxTokens:    true,
				NoStopSequence: true,
				Tools:          scoreboard.Flaky,
				IndecisiveTool: scoreboard.Flaky,
				Seed:           true,
			},
		},
		{
			Models:   []string{"deepseek-reasoning"},
			Thinking: true,
			/*
				In:     map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
				Out:    map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
				GenSync: &scoreboard.FunctionalityText{
					NoMaxTokens:    true,
					NoStopSequence: true,
					Seed:           true,
				},
				// Upstream parsing is broken, which means we can't recommend GenStream.
			*/
		},
		{
			Models: []string{"flux"},
			In:     map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			Out: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
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
			Models: []string{"openai"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				NoMaxTokens:    true,
				NoStopSequence: true,
				Tools:          scoreboard.True,
				BiasedTool:     scoreboard.True,
				JSON:           true,
				Seed:           true,
			},
			GenStream: &scoreboard.FunctionalityText{
				BrokenTokenUsage: scoreboard.True,
				NoMaxTokens:      true,
				NoStopSequence:   true,
				Tools:            scoreboard.True,
				BiasedTool:       scoreboard.True,
				JSON:             true,
				Seed:             true,
			},
		},
		// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md
		{
			Models: []string{"openai-audio"},
			/* TODO: Requires audio and Scoreboard fails to test this use case.
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityAudio: {
					Inline:           true,
					SupportedFormats: []string{"audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a", "audio/webm"},
				},
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityAudio: {
					Inline:           true,
					SupportedFormats: []string{"audio/mpeg", "audio/wav", "audio/mp4", "audio/x-m4a", "audio/webm"},
				},
				genai.ModalityText: {Inline: true},
			},
			GenSync: &scoreboard.FunctionalityText{
				NoMaxTokens: true,
				JSON:        true,
				Seed:        true,
			},
			// GenStream doesn't succeed in the smoke test, so consider it broken for now.
			*/
		},
		// Ignored.
		{
			Models: []string{
				"bidara",
				"evil",
				"gemini",
				"geminisearch",
				"glm",
				"hypnosis-tracy",
				"kontext",
				"llama-fast-roblox",
				"llama-roblox",
				"midijourney",
				"mirexa",
				"mistral",
				"mistral-nemo-roblox",
				"mistral-roblox",
				"openai-fast",
				"openai-large",
				"openai-roblox",
				"qwen-coder",
				"rtist",
				"sur",
				"turbo",
				"unity",
			},
		},
	},
}

// ChatRequest is barely documented at
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

	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md
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
	Logprobs    bool  `json:"logprobs,omitzero"`
	TopLogprobs int64 `json:"top_logprobs,omitzero"`
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
	if v.TopLogprobs > 0 {
		// Try to request it but it's known to not work, so add it anyway to the unsupported flag.
		c.TopLogprobs = v.TopLogprobs
		c.Logprobs = true
		unsupported = append(unsupported, "TopLogprobs")
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
		m.Content = make(Contents, len(in.Requests))
		for i := range in.Requests {
			if err := m.Content[i].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request %d: %w", i, err)
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
				m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall)
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply %d: %w", i, err)
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

// Contents exists to marshal single content text block as a string.
type Contents []Content

func (c *Contents) IsZero() bool {
	return len(*c) == 0
}

func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal(([]Content)(*c))
}

// UnmarshalJSON implements custom unmarshalling for Contents type
// to handle cases where content could be a string or a raw JSON object.
func (c *Contents) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*c = nil
		return nil
	}
	s := ""
	if err := json.Unmarshal(b, &s); err == nil {
		*c = Contents{{Type: ContentText, Text: s}}
		return nil
	}

	// Otherwise, it's likely a raw JSON object. Convert it back to a string.
	data := map[string]any{}
	if err := json.Unmarshal(b, &data); err != nil {
		return err
	}
	raw, err := json.Marshal(data)
	if err != nil {
		return err
	}
	*c = Contents{{Type: ContentText, Text: string(raw)}}
	return nil
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
			// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#speech-to-text-capabilities-audio-input-%EF%B8%8F
			c.Type = ContentAudio
			c.InputAudio.Data = data
			switch mimeType {
			case "audio/mpeg":
				c.InputAudio.Format = "mp3"
			default:
				return fmt.Errorf("implement mime type %s conversion", mimeType)
			}
		case strings.HasPrefix(mimeType, "image/") || in.Doc.URL != "":
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
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
			// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#speech-to-text-capabilities-audio-input-%EF%B8%8F
			c.Type = ContentAudio
			c.InputAudio.Data = data
			switch mimeType {
			case "audio/mpeg":
				c.InputAudio.Format = "mp3"
			default:
				return fmt.Errorf("implement mime type %s conversion", mimeType)
			}
		case strings.HasPrefix(mimeType, "image/") || in.Doc.URL != "":
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
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
	Model   string    `json:"model"` // The actual model name, which is likely different from the alias.
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
	KVTransferParams    struct{}             `json:"kv_transfer_params"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	// There's a "X-Cache" HTTP response header that says when the whole request was cached.
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
	return out, err
}

type FinishReason string

const (
	FinishStop      FinishReason = "stop"
	FinishLength    FinishReason = "length"
	FinishToolCalls FinishReason = "tool_calls"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishLength:
		return genai.FinishedLength
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
		TextTokens               int64 `json:"text_tokens"`
	} `json:"completion_tokens_details"`
	PromptTokensDetails struct {
		AudioTokens  int64 `json:"audio_tokens"`
		CachedTokens int64 `json:"cached_tokens"`
		ImageTokens  int64 `json:"image_tokens"`
		TextTokens   int64 `json:"text_tokens"`
	} `json:"prompt_tokens_details"`
}

type MessageResponse struct {
	Role             string     `json:"role"`
	ReasoningContent string     `json:"reasoning_content"`
	Content          Contents   `json:"content"`
	ToolCalls        []ToolCall `json:"tool_calls"`
	Annotations      []struct{} `json:"annotations"`
	Refusal          struct{}   `json:"refusal"`
	Audio            struct {
		Data []byte `json:"data"`
	} `json:"audio"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	for i := range m.Content {
		if m.Content[i].Text != "" {
			out.Replies = append(out.Replies, genai.Reply{Text: m.Content[i].Text})
		} else {
			return fmt.Errorf("unsupported content #%d: %q", i, m.Content[i])
		}
	}
	if m.ReasoningContent != "" {
		// Paper over broken "deepseek".
		if len(out.Replies) == 1 && out.Replies[0].Text != "" {
			out.Replies = append(out.Replies, genai.Reply{Thinking: m.ReasoningContent})
		} else {
			out.Replies = append(out.Replies, genai.Reply{Text: m.ReasoningContent})
		}
	}
	if len(m.Audio.Data) != 0 {
		out.Replies = append(out.Replies, genai.Reply{
			Doc: genai.Doc{Filename: "sound.wav", Src: &bb.BytesBuffer{D: m.Audio.Data}},
		})
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
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
	CustomBlocklists struct {
		Filtered bool `json:"filtered"`
		Details  []struct {
			BlocklistName string `json:"blocklist_name"`
		} `json:"details"`
	} `json:"custom_blocklists"`
	Hate struct {
		Filtered bool   `json:"filtered"`
		Severity string `json:"severity"` // "safe"
	} `json:"hate"`
	Jailbreak struct {
		Filtered bool `json:"filtered"`
		Detected bool `json:"detected"`
	} `json:"jailbreak"`
	ProtectedMaterialCode struct {
		Filtered bool `json:"filtered"`
		Detected bool `json:"detected"`
	} `json:"protected_material_code"`
	ProtectedMaterialText struct {
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
	Community        bool     `json:"community"`
	Description      string   `json:"description"`
	InputModalities  []string `json:"input_modalities"`
	MaxInputChars    int64    `json:"maxInputChars"`
	Name             string   `json:"name"`
	OriginalName     string   `json:"original_name"`
	OutputModalities []string `json:"output_modalities"`
	Pricing          struct {
		PromptTokens     float64 `json:"prompt_tokens"`
		CompletionTokens float64 `json:"completion_tokens"`
	} `json:"pricing,omitzero"`
	Provider   string   `json:"provider"`
	Reasoning  bool     `json:"reasoning"`
	Search     bool     `json:"search"`
	Tier       string   `json:"tier"` // "seed", "flower"
	Tools      bool     `json:"tools"`
	Uncensored bool     `json:"uncensored"`
	Voices     []string `json:"voices"`
	Vision     bool     `json:"vision"`
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
	ErrorVal string `json:"error"`
	Status   int64  `json:"status"`
	Details  struct {
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
		Errors    []struct {
			Path        []string     `json:"path"`
			Message     string       `json:"message"`
			Code        string       `json:"code"`
			UnionErrors []UnionError `json:"unionErrors"`
		} `json:"errors"`
		Success  bool       `json:"success"`
		Result   struct{}   `json:"result"`
		Messages []struct{} `json:"messages"`
	} `json:"details"`

	Message    string   `json:"message"`
	Debug      struct{} `json:"debug"`
	TimingInfo []struct {
		Step      string `json:"step"`
		Timestamp int64  `json:"timestamp"`
	} `json:"timingInfo"`
	RequestID         string         `json:"requestId"`
	RequestParameters map[string]any `json:"requestParameters"`
}

func (er *ErrorResponse) Error() string {
	suffix := ""
	if er.Details.Provider != "" {
		suffix = "; provider:" + er.Details.Provider
	}
	if len(er.Details.Errors) != 0 {
		return fmt.Sprintf("%s%s", er.Details.Errors[0].Message, suffix)
	}
	if er.Details.Error.Message != "" {
		// It already contains the Provider name.
		return fmt.Sprintf("%s/%s%s: %s%s", er.Details.Error.Type, er.Details.Error.Param, er.Details.Error.Code, er.Details.Error.Message, suffix)
	}
	if er.Details.Message != "" {
		return fmt.Sprintf("%s%s", er.Details.Message, suffix)
	}
	if er.Details.Detail != "" {
		return fmt.Sprintf("%s%s", er.Details.Detail, suffix)
	}
	if er.Message != "" {
		return fmt.Sprintf("%s %s%s", er.ErrorVal, er.Message, suffix)
	}
	return fmt.Sprintf("%s%s", er.ErrorVal, suffix)
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

type UnionError struct {
	Issues []struct {
		Code        string       `json:"code"`
		Expected    string       `json:"expected"`
		Received    string       `json:"received"`
		Path        []string     `json:"path"`
		Message     string       `json:"message"`
		UnionErrors []UnionError `json:"unionErrors"`
	} `json:"issues"`
	Name string `json:"name"`
}

// Client implements genai.ProviderModel.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Pollinations platform API.
// The value for options APIKey can be either an API key retrieved from https://auth.pollinations.ai/ or a referrer.
// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#referrer-
//
// options APIKey is optional. Providing one, either via environment variable POLLINATIONS_API_KEY, will increase quota.
//
// To use multiple models, create multiple clients.
// Models are listed at https://docs.perplexity.ai/guides/model-cards
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	var h http.Header
	if apiKey == "" {
		apiKey = os.Getenv("POLLINATIONS_API_KEY")
	}
	if apiKey != "" {
		if strings.HasPrefix(apiKey, "http://") || strings.HasPrefix(apiKey, "https://") {
			h = http.Header{"Referrer": {apiKey}}
		} else {
			h = http.Header{"Authorization": {"Bearer " + apiKey}}
		}
	}
	model := opts.Model
	if model == "" {
		model = genai.ModelGood
	}
	t := base.DefaultTransport
	if r, ok := t.(*roundtrippers.Retry); ok {
		// Make a copy so we can edit it.
		c := *r
		if p, ok := c.Policy.(*roundtrippers.ExponentialBackoff); ok {
			// Tweak the policy.
			c.Policy = &exponentialBackoff{ExponentialBackoff: *p}
		} else {
			return nil, fmt.Errorf("unsupported retry policy %T", c.Policy)
		}
		t = &c
	} else {
		return nil, fmt.Errorf("unsupported transport %T", t)
	}
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
	switch model {
	case genai.ModelNone:
		c.Model = ""
	case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA:
		var err error
		if c.Model, err = c.selectBestModel(context.Background(), model); err != nil {
			return nil, err
		}
	}
	return c, nil
}

// selectBestModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
func (c *Client) selectBestModel(ctx context.Context, preference string) (string, error) {
	// We only list text models here, not images generation ones.
	mdls, err := c.ListTextModels(ctx)
	if err != nil {
		return "", err
	}
	cheap := preference == genai.ModelCheap
	good := preference == genai.ModelGood
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*TextModel)
		if m.Audio || strings.HasSuffix(m.Name, "roblox") {
			continue
		}
		// This is meh.
		if cheap {
			if strings.HasPrefix(m.Name, "llama") {
				selectedModel = m.Name
			}
		} else if good {
			if strings.HasPrefix(m.Name, "openai") && !m.Reasoning {
				selectedModel = m.Name
			}
		} else {
			if !strings.HasPrefix(m.Name, "openai") && m.Reasoning {
				selectedModel = m.Name
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard
}

func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	if c.isAudio(opts) || c.isImage(opts) {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.GenDoc(ctx, msgs[0], opts)
	}
	if err := Cache.ValidateModality(c, genai.ModalityText); err != nil {
		return genai.Result{}, err
	}
	return c.ProviderGen.GenSync(ctx, msgs, opts)
}

func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	if c.isAudio(opts) || c.isImage(opts) {
		return base.SimulateStream(ctx, c, msgs, chunks, opts)
	}
	if err := Cache.ValidateModality(c, genai.ModalityText); err != nil {
		return genai.Result{}, err
	}
	return c.ProviderGen.GenStream(ctx, msgs, chunks, opts)
}

// GenDoc uses the text-to-image API to generate an image.
//
// Default rate limit is 0.2 QPS / IP.
func (c *Client) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#1-text-to-image-get-%EF%B8%8F
	// TODO:
	// https://github.com/pollinations/pollinations/blob/master/APIDOCS.md#4-text-to-speech-get-%EF%B8%8F%EF%B8%8F
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
		if supported := opts.Modalities(); !slices.Contains(supported, genai.ModalityImage) {
			return res, fmt.Errorf("modality image not supported, supported: %s", supported)
		}
	}
	for i := range msg.Requests {
		if msg.Requests[i].Text == "" {
			return res, errors.New("only text can be passed as input")
		}
	}
	if err := Cache.ValidateModality(c, genai.ModalityImage); err != nil {
		return genai.Result{}, err
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
		return genai.Result{}, fmt.Errorf("unsupported options type %T", opts)
	}

	qp.Add("nologo", "true")
	qp.Add("private", "true") // "nofeed"
	qp.Add("enhance", "false")
	qp.Add("safe", "false")
	// qp.Add("negative_prompt", "worst quality, blurry")
	qp.Add("quality", "medium")
	for _, mc := range msg.Requests {
		if mc.Doc.Src != nil {
			return res, errors.New("inline document is not supported")
		}
		if mc.Doc.URL != "" {
			qp.Add("image", mc.Doc.URL)
		}
	}

	prompt := url.QueryEscape(msg.String())
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
	res.Replies = []genai.Reply{{Doc: genai.Doc{Src: &bb.BytesBuffer{D: b}}}}
	if ct := resp.Header.Get("Content-Type"); strings.HasPrefix(ct, "image/jpeg") {
		res.Replies[0].Doc.Filename = "content.jpg"
	} else {
		return res, fmt.Errorf("unknown Content-Type: %s", ct)
	}
	return res, res.Validate()
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

func (c *Client) isAudio(opts genai.Options) bool {
	return opts != nil && slices.Contains(opts.Modalities(), genai.ModalityAudio)
}

func (c *Client) isImage(opts genai.Options) bool {
	// TODO: Use Scoreboard list.
	switch c.Model {
	case "flux", "gptimage", "turbo":
		return true
	default:
		return opts != nil && slices.Contains(opts.Modalities(), genai.ModalityImage)
	}
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
			result.Usage.TotalTokens = pkt.Usage.TotalTokens
		}
		if len(pkt.Choices) != 1 {
			continue
		}
		if fr := pkt.Choices[0].FinishReason; fr != "" {
			result.Usage.FinishReason = fr.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt)
		}
		f := genai.ReplyFragment{
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

type exponentialBackoff struct {
	roundtrippers.ExponentialBackoff
}

func (e *exponentialBackoff) ShouldRetry(ctx context.Context, start time.Time, try int, err error, resp *http.Response) bool {
	if resp != nil && resp.StatusCode == 402 {
		return true
	}
	return e.ExponentialBackoff.ShouldRetry(ctx, start, try, err, resp)
}

// ModelCache is a cache of the list of models.
type ModelCache struct {
	// Keep a cache of the model to ensure we don't send a text requested to a image generation model and
	// vice-versa.
	mu     sync.Mutex
	models []genai.Model
}

// ValidateModality returns nil if the modality is supported by the model.
func (m *ModelCache) ValidateModality(c genai.ProviderModel, mod genai.Modality) error {
	if _, err := m.Warmup(c); err != nil {
		return err
	}
	isText := false
	isImage := false
	found := false
	model := c.ModelID()
	for i := range m.models {
		if m.models[i].GetID() == model {
			found = true
			_, isText = m.models[i].(*TextModel)
			_, isImage = m.models[i].(ImageModel)
			break
		}
	}
	if !found {
		return fmt.Errorf("model %q not supported by pollinations", model)
	}
	switch mod {
	case genai.ModalityText:
		if isText {
			return nil
		}
	case genai.ModalityImage:
		if isImage {
			return nil
		}
	case genai.ModalityVideo, genai.ModalityDocument, genai.ModalityAudio:
	}
	return fmt.Errorf("modality %s not supported", mod)
}

func (m *ModelCache) Warmup(c genai.ProviderModel) ([]genai.Model, error) {
	var err error
	m.mu.Lock()
	if m.models == nil {
		m.models, err = c.ListModels(context.Background())
	}
	m.mu.Unlock()
	return m.models, err
}

func (m *ModelCache) Clear() {
	m.mu.Lock()
	m.models = nil
	m.mu.Unlock()
}

// Cache is the global cache of the list of models.
var Cache ModelCache

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderGenDoc     = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ scoreboard.ProviderScore = &Client{}
)
