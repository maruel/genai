// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Together.ai chat completions, image generation, and models REST API.
//
// Reference: https://docs.together.ai/reference/chat-completions-1

package togetherai

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math/big"
	"strconv"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// ChatRequest is documented at https://docs.together.ai/reference/chat-completions-1
//
// https://docs.together.ai/docs/chat-overview
type ChatRequest struct {
	Model                         string             `json:"model"`
	Stream                        bool               `json:"stream"`
	Messages                      []Message          `json:"messages"`
	MaxTokens                     int64              `json:"max_tokens,omitzero"`
	Stop                          []string           `json:"stop,omitzero"`
	Temperature                   float64            `json:"temperature,omitzero"` // [0, 1]
	TopP                          float64            `json:"top_p,omitzero"`       // [0, 1]
	TopK                          int64              `json:"top_k,omitzero"`
	ContextLengthExceededBehavior string             `json:"context_length_exceeded_behavior,omitzero"` // "error", "truncate"
	RepetitionPenalty             float64            `json:"repetition_penalty,omitzero"`
	Logprobs                      int64              `json:"logprobs,omitzero"` // Actually toplogprobs; [0, 20]
	Echo                          bool               `json:"echo,omitzero"`
	N                             int32              `json:"n,omitzero"`                 // Number of completions to generate
	PresencePenalty               float64            `json:"presence_penalty,omitzero"`  // [-2.0, 2.0]
	FrequencyPenalty              float64            `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	LogitBias                     map[string]float64 `json:"logit_bias,omitzero"`
	Seed                          int64              `json:"seed,omitzero"`
	ResponseFormat                struct {
		Type   string           `json:"type,omitzero"` // "json_object", "json_schema" according to python library.
		Schema genai.JSONSchema `json:"schema,omitzero"`
	} `json:"response_format,omitzero"`
	Tools       []Tool `json:"tools,omitzero"`
	ToolChoice  string `json:"tool_choice,omitzero"`  // "auto" or a []Tool
	SafetyModel string `json:"safety_model,omitzero"` // https://docs.together.ai/docs/inference-models#moderation-models
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
	c.Model = model
	// Validate messages
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
		case *genai.GenOptionText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			c.Logprobs = v.TopLogprobs
			// TODO: Toplogprobs are not returned unless streaming. lol. Sadly we do not know yet here if streaming
			// is enabled.
			// if v.TopLogprobs > 1 && !Stream {
			// 	unsupported = append(unsupported, "GenOptionText.TopLogprobs")
			// }
			c.TopK = v.TopK
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				// Warning: using a model small may fail.
				c.ResponseFormat.Type = "json_schema"
				s, err := v.DecodeSchema()
				if err != nil {
					errs = append(errs, err)
				} else {
					c.ResponseFormat.Schema = s
				}
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json_object"
			}
		case *genai.GenOptionTools:
			if len(v.Tools) != 0 {
				switch v.Force {
				case genai.ToolCallAny:
					c.ToolChoice = "auto"
				case genai.ToolCallRequired:
					// Interestingly, https://docs.together.ai/reference/chat-completions-1 doesn't document anything
					// beside "auto" but https://docs.livekit.io/agents/integrations/llm/together/ says that
					// "required" works. I'll have to confirm.
					c.ToolChoice = "required"
				case genai.ToolCallNone:
					c.ToolChoice = "none"
				}
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					s, err := t.GetInputSchema()
					if err != nil {
						errs = append(errs, err)
					}
					c.Tools[i].Function.Parameters = s
				}
			}
		case genai.GenOptionSeed:
			c.Seed = int64(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: "text", Text: sp}}})
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
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Message is documented at https://docs.together.ai/reference/chat-completions-1
type Message struct {
	Role      string   `json:"role,omitzero"` // "system", "assistant", "user"
	Content   Contents `json:"content,omitzero"`
	Reasoning string   `json:"reasoning,omitzero"`
	// Warning: using a small model may fail.
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
		m.Content = make([]Content, 0, len(in.Requests))
		for i := range in.Requests {
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request #%d: %w", i, err)
			}
		}
	}
	if len(in.Replies) != 0 {
		for i := range in.Replies {
			if in.Replies[i].Reasoning != "" {
				continue
			}
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
					return fmt.Errorf("reply #%d: %w", i, err)
				}
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
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
	if m.Reasoning != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.Reasoning})
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	// Handle models that return empty content with reasoning (e.g., thinking models with very low MaxTokens).
	// Filter out any empty replies while keeping valid ones (like reasoning).
	if len(out.Replies) > 0 {
		filtered := make([]genai.Reply, 0, len(out.Replies))
		for i := range out.Replies {
			if !out.Replies[i].IsZero() {
				filtered = append(filtered, out.Replies[i])
			}
		}
		if len(filtered) == 0 {
			// All replies were empty, create an empty Reply with Opaque set to pass validation.
			out.Replies = []genai.Reply{{Opaque: map[string]any{"empty": true}}}
		} else {
			out.Replies = filtered
		}
	} else {
		// No replies at all, create an empty Reply with Opaque set to pass validation.
		out.Replies = []genai.Reply{{Opaque: map[string]any{"empty": true}}}
	}
	return nil
}

// Contents is a collection of content blocks.
type Contents []Content

// UnmarshalJSON implements json.Unmarshaler.
//
// Together.AI replies with content as a string.
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
	*c = []Content{{Type: "text", Text: s}}
	return nil
}

// MarshalJSON implements json.Marshaler.
//
// Together.AI really prefer simple strings.
func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal([]Content(*c))
}

// Content is a provider-specific content block.
type Content struct {
	Type ContentType `json:"type,omitzero"`

	// Type == ContentText
	Text string `json:"text,omitzero"`

	// Type == ContentImageURL
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`

	// Type == ContentVideoURL
	VideoURL struct {
		URL string `json:"url,omitzero"`
	} `json:"video_url,omitzero"`
}

// FromRequest converts from a genai request.
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
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case strings.HasPrefix(mimeType, "video/"):
			c.Type = ContentVideoURL
			if in.Doc.URL == "" {
				c.VideoURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.VideoURL.URL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Request type")
}

// FromReply converts from a genai reply.
func (c *Content) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field ToolCall.Opaque not supported")}
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
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
		case strings.HasPrefix(mimeType, "video/"):
			c.Type = ContentVideoURL
			if in.Doc.URL == "" {
				c.VideoURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.VideoURL.URL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			return &internal.BadError{Err: fmt.Errorf("unsupported mime type %s", mimeType)}
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
	case ContentImageURL, ContentVideoURL:
		return &internal.BadError{Err: fmt.Errorf("unsupported content type %q", c.Type)}
	default:
		return &internal.BadError{Err: fmt.Errorf("unsupported content type %q", c.Type)}
	}
	return nil
}

// ContentType is a provider-specific content type.
type ContentType string

// Content type values.
const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
	ContentVideoURL ContentType = "video_url"
)

// Tool is a provider-specific tool definition.
type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string           `json:"name,omitzero"`
		Description string           `json:"description,omitzero"`
		Parameters  genai.JSONSchema `json:"parameters,omitzero"`
	} `json:"function,omitzero"`
}

// ToolCall is a provider-specific tool call.
type ToolCall struct {
	Index    int64  `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"` // function
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// From converts from the genai equivalent.
func (t *ToolCall) From(in *genai.ToolCall) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field ToolCall.Opaque not supported")}
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
	// Empty arguments default to empty JSON object.
	if out.Arguments == "" {
		out.Arguments = "{}"
	}
}

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	ID             string   `json:"id"`
	Prompt         []string `json:"prompt"`
	PromptTokenIDs TokenIDs `json:"prompt_token_ids,omitzero"`
	Choices        []struct {
		// Text  string `json:"text"`
		Index    int64    `json:"index"`
		TokenIDs TokenIDs `json:"token_ids,omitzero"`
		// The seed is returned as a int128.
		Seed         big.Int      `json:"seed"`
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
		Logprobs     Logprobs     `json:"logprobs"`
	} `json:"choices"`
	Usage            Usage      `json:"usage"`
	Created          base.TimeS `json:"created"`
	Model            string     `json:"model"`
	KVTransferParams struct{}   `json:"kv_transfer_params"`
	Object           string     `json:"object"` // "chat.completion"
	Warnings         []struct {
		Message string `json:"message"`
	} `json:"warnings"`
	SystemFingerprint struct{} `json:"system_fingerprint"`
	Servicetier       struct{} `json:"service_tier"`
	Metadata          struct {
		WeightVersion string `json:"weight_version"` // "default"
	} `json:"metadata"`
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.Usage.PromptTokens,
			InputCachedTokens: max(c.Usage.CachedTokens, c.Usage.PromptTokensDetails.CachedTokens),
			ReasoningTokens:   c.Usage.ReasoningTokens,
			OutputTokens:      c.Usage.CompletionTokens,
			TotalTokens:       c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	out.Logprobs = c.Choices[0].Logprobs.To()
	err := c.Choices[0].Message.To(&out.Message)
	if err == nil && len(c.Warnings) != 0 {
		ent := &base.ErrNotSupported{}
		for _, w := range c.Warnings {
			if strings.Contains(w.Message, "tool_choice") {
				ent.Options = append(ent.Options, "GenOptionTools.Force")
			} else {
				ent.Options = append(ent.Options, w.Message)
			}
		}
		return out, ent
	}
	return out, err
}

// FinishReason is a provider-specific finish reason.
type FinishReason string

// Finish reason values.
const (
	FinishStop         FinishReason = "stop"
	FinishEOS          FinishReason = "eos"
	FinishLength       FinishReason = "length"
	FinishFunctionCall FinishReason = "function_call"
	FinishToolCalls    FinishReason = "tool_calls"
)

// ToFinishReason converts to a genai.FinishReason.
func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop, "":
		return genai.FinishedStop
	case FinishEOS:
		return genai.FinishedStopSequence
	case FinishLength:
		return genai.FinishedLength
	case FinishToolCalls, FinishFunctionCall:
		return genai.FinishedToolCalls
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

// TokenIDs is a slice of token IDs returned by the API. Values are nullable and
// can be int or float, so we use *json.Number to handle both nulls and varying
// numeric types.
type TokenIDs []*json.Number

// Logprobs is the provider-specific log probabilities.
type Logprobs struct {
	Tokens        []string          `json:"tokens"`
	TokenLogprobs []float64         `json:"token_logprobs"`
	TokenIDs      TokenIDs          `json:"token_ids,omitzero"` // Not set.
	Content       []json.RawMessage `json:"content,omitzero"`   // Complex structure with logprobs data.
}

// To converts to the genai equivalent.
func (l *Logprobs) To() [][]genai.Logprob {
	if len(l.Tokens) == 0 {
		return nil
	}
	// Toplogprobs are not returned when not streaming (!!)
	out := make([][]genai.Logprob, 0, len(l.Tokens))
	for i := range l.Tokens {
		out = append(out, []genai.Logprob{{Text: l.Tokens[i], Logprob: l.TokenLogprobs[i]}})
	}
	return out
}

// LogprobsChunk is a chunk of log probability data.
type LogprobsChunk struct {
	Tokens        []string  `json:"tokens"`
	TokenLogprobs []float64 `json:"token_logprobs"`
	TokenIDs      TokenIDs  `json:"token_ids,omitzero"`
	TopLogprobs   [][]struct {
		Token   string  `json:"token"`
		Logprob float64 `json:"logprob"`
	} `json:"top_logprobs"`
	Content json.RawMessage `json:"content,omitzero"` // Undocumented model-specific field, not in Together.AI's OpenAPI LogprobsPart spec.
}

// UnmarshalJSON implements json.Unmarshaler.
//
// For some models, the logprobs are returned as a float with no context.
func (l *LogprobsChunk) UnmarshalJSON(b []byte) error {
	// Sometimes it's a 0. Yo.
	v := 0.
	if err := json.Unmarshal(b, &v); err == nil {
		*l = LogprobsChunk{TokenLogprobs: []float64{v}}
		return nil
	}

	type Alias LogprobsChunk
	a := struct{ *Alias }{Alias: (*Alias)(l)}
	d := json.NewDecoder(bytes.NewReader(b))
	if internal.BeLenient {
		// Allow unknown fields when lenient mode is on
	} else {
		d.DisallowUnknownFields()
	}
	return d.Decode(&a)
}

// To converts to the genai equivalent.
func (l *LogprobsChunk) To() [][]genai.Logprob {
	if len(l.Tokens) == 0 {
		return nil
	}
	out := make([][]genai.Logprob, 0, len(l.Tokens))
	for i := range l.Tokens {
		lp := make([]genai.Logprob, 1, len(l.TopLogprobs[i])+1)
		lp[0] = genai.Logprob{Text: l.Tokens[i], Logprob: l.TokenLogprobs[i]}
		for _, tlp := range l.TopLogprobs[i] {
			lp = append(lp, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
		}
		out = append(out, lp)
	}
	return out
}

// Usage is the provider-specific token usage.
type Usage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	ReasoningTokens     int64 `json:"reasoning_tokens"`
	TotalTokens         int64 `json:"total_tokens"`
	CachedTokens        int64 `json:"cached_tokens"`
	PromptTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

// ChoiceError is the choice-level error in a streaming chat chunk.
type ChoiceError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
	Param   string `json:"param,omitzero"`
}

func (e *ChoiceError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s (%s): %s", e.Code, e.Type, e.Message)
	}
	if e.Type != "" {
		return fmt.Sprintf("%s: %s", e.Type, e.Message)
	}
	return e.Message
}

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	ID      string     `json:"id"`
	Object  string     `json:"object"` // "chat.completion.chunk"
	Created base.TimeS `json:"created"`
	Model   string     `json:"model"`
	Choices []struct {
		Index       int64              `json:"index"`
		Text        string             `json:"text"` // Duplicated to Delta.Text
		Seed        big.Int            `json:"seed"`
		Error       *ChoiceError       `json:"error,omitzero"`
		Role        string             `json:"role,omitzero"` // Sometimes appears in streaming
		Logprobs    LogprobsChunk      `json:"logprobs"`
		TopLogprobs map[string]float64 `json:"top_logprobs,omitzero"`
		Delta       struct {
			TokenID   int64      `json:"token_id"`
			Role      string     `json:"role"`
			Content   string     `json:"content"`
			ToolCalls []ToolCall `json:"tool_calls"`
			Reasoning string     `json:"reasoning"`
		} `json:"delta"`
		FinishReason FinishReason `json:"finish_reason"`
		MatchedStop  int64        `json:"matched_stop"`
		StopReason   StopReason   `json:"stop_reason"`
		ToolCalls    []ToolCall   `json:"tool_calls"`
	} `json:"choices"`
	// SystemFingerprint string `json:"system_fingerprint"`
	Usage    Usage `json:"usage"`
	Warnings []struct {
		Message string `json:"message"`
	} `json:"warnings"`
}

// StopReason is sometimes a string, sometimes a number, maybe it is a token number?
type StopReason string

// UnmarshalJSON implements json.Unmarshaler.
func (s *StopReason) UnmarshalJSON(b []byte) error {
	v := 0
	if err := json.Unmarshal(b, &v); err == nil {
		*s = StopReason(strconv.Itoa(v))
		return nil
	}
	return json.Unmarshal(b, (*string)(s))
}

// Model is the provider-specific model metadata.
type Model struct {
	ID            string     `json:"id"`
	Object        string     `json:"object"`
	Created       base.TimeS `json:"created"`
	Type          string     `json:"type"` // "chat", "moderation", "image"
	Running       bool       `json:"running"`
	DisplayName   string     `json:"display_name"`
	Organization  string     `json:"organization"`
	Link          string     `json:"link"`
	License       string     `json:"license"`
	UUID          string     `json:"uuid,omitzero"`
	ContextLength int64      `json:"context_length"`
	Config        struct {
		ChatTemplate    string   `json:"chat_template"`
		Stop            []string `json:"stop"`
		BosToken        string   `json:"bos_token"`
		EosToken        string   `json:"eos_token"`
		MaxOutputLength int64    `json:"max_output_length"`
	} `json:"config"`
	Pricing struct {
		Hourly      float64           `json:"hourly"`
		Input       float64           `json:"input"`
		Output      float64           `json:"output"`
		Base        float64           `json:"base"`
		Finetune    float64           `json:"finetune"`
		CachedInput float64           `json:"cached_input,omitzero"`
		ImagePixel  PricingImagePixel `json:"image_pixel,omitzero"`
		Transcribe  PricingTranscribe `json:"transcribe,omitzero"`
		Image       PricingMedia      `json:"image,omitzero"`
		Video       PricingMedia      `json:"video,omitzero"`
	} `json:"pricing"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	c := ""
	if m.Config.MaxOutputLength != 0 {
		c = fmt.Sprintf("%d/%d", m.ContextLength, m.Config.MaxOutputLength)
	} else {
		c = strconv.FormatInt(m.ContextLength, 10)
	}
	created := ""
	if m.Created > 1000 {
		created = " (" + m.Created.AsTime().Format("2006-01-02") + ")"
	}
	pricing := ""
	if m.Pricing.Input != 0 || m.Pricing.Output != 0 {
		pricing = fmt.Sprintf("; in: %.2f$/Mt out: %.2f$/Mt", m.Pricing.Input, m.Pricing.Output)
	}
	return fmt.Sprintf("%s%s: %s Context: %s%s", m.ID, created, m.Type, c, pricing)
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return m.ContextLength
}

// PricingImagePixel is the per-megapixel pricing for image generation.
//
// The API returns either 0 (no pricing) or an object with pricing details.
type PricingImagePixel struct {
	PricePerMegapixel float64 `json:"price_per_megapixel,omitzero"`
	MinSteps          int     `json:"min_steps,omitzero"`
}

// UnmarshalJSON handles the polymorphic API response (number or object).
func (p *PricingImagePixel) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] != '{' {
		*p = PricingImagePixel{}
		return json.Unmarshal(b, &p.PricePerMegapixel)
	}
	type alias PricingImagePixel
	return json.Unmarshal(b, (*alias)(p))
}

// PricingTranscribe is the per-minute pricing for audio transcription.
//
// The API returns either 0 (no pricing) or an object with pricing details.
type PricingTranscribe struct {
	PricePerMinute float64 `json:"price_per_minute,omitzero"`
}

// UnmarshalJSON handles the polymorphic API response (number or object).
func (p *PricingTranscribe) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] != '{' {
		*p = PricingTranscribe{}
		return json.Unmarshal(b, &p.PricePerMinute)
	}
	type alias PricingTranscribe
	return json.Unmarshal(b, (*alias)(p))
}

// PricingMedia is the example-based pricing for image or video generation.
//
// The API returns either 0 (no pricing) or an object with an example price and description.
type PricingMedia struct {
	ExamplePrice       float64 `json:"example_price,omitzero"`
	ExampleDescription string  `json:"example_description,omitzero"`
}

// UnmarshalJSON handles the polymorphic API response (number or object).
func (p *PricingMedia) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] != '{' {
		*p = PricingMedia{}
		return json.Unmarshal(b, &p.ExamplePrice)
	}
	type alias PricingMedia
	return json.Unmarshal(b, (*alias)(p))
}

// ModelsResponse represents the response structure for TogetherAI models listing.
type ModelsResponse []Model

// ToModels converts TogetherAI models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(*r))
	for i := range *r {
		models[i] = &(*r)[i]
	}
	return models
}

// ImageRequest doesn't have a formal documentation.
//
// https://github.com/togethercomputer/together-python/blob/main/src/together/resources/images.py is the
// closest.
type ImageRequest struct {
	Prompt         string `json:"prompt"`
	Model          string `json:"model,omitzero"`
	Steps          int64  `json:"steps,omitzero"`  // Default 20
	Seed           int64  `json:"seed,omitzero"`   //
	N              int64  `json:"n,omitzero"`      // Default 1
	Height         int64  `json:"height,omitzero"` // Default 1024
	Width          int64  `json:"width,omitzero"`  // Default 1024
	NegativePrompt string `json:"negative_prompt,omitzero"`
	ImageURL       string `json:"image_url,omitzero"`
	Image          []byte `json:"image_base64,omitzero"`
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
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *genai.GenOptionImage:
			i.Height = int64(v.Height)
			i.Width = int64(v.Width)
		case genai.GenOptionSeed:
			i.Seed = int64(v)
		default:
			return &base.ErrNotSupported{Options: []string{internal.TypeName(opt)}}
		}
	}
	return nil
}

// ImageResponse doesn't have a formal documentation.
//
// https://github.com/togethercomputer/together-python/blob/main/src/together/types/images.py is the
// closest.
type ImageResponse struct {
	ID     string            `json:"id"`
	Model  string            `json:"model"`
	Object string            `json:"object"` // "list"
	Data   []ImageChoiceData `json:"data"`
}

// ImageChoiceData is the data for one image generation choice.
type ImageChoiceData struct {
	Index   int64  `json:"index"`
	B64JSON []byte `json:"b64_json"`
	URL     string `json:"url"`
	Timings struct {
		Inference float64 `json:"inference"`
	} `json:"timings"`
}

//

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
	ID       string `json:"id"`
	ErrorVal struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code"`
		Param   string `json:"param"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	if er.ErrorVal.Code != "" {
		return fmt.Sprintf("%s (%s): %s", er.ErrorVal.Code, er.ErrorVal.Type, er.ErrorVal.Message)
	}
	if er.ErrorVal.Type != "" {
		return fmt.Sprintf("%s: %s", er.ErrorVal.Type, er.ErrorVal.Message)
	}
	return er.ErrorVal.Message
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
