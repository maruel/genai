// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openaicompatible implements a minimal client for "OpenAI-compatible" providers.
//
// It's a good starting point to implement a client for a new platform.
package openaicompatible

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

type ChatRequest struct {
	Model            string    `json:"model,omitzero"`
	Messages         []Message `json:"messages"`
	MaxTokens        int64     `json:"max_tokens,omitzero"`
	Stop             []string  `json:"stop,omitzero"`
	Stream           bool      `json:"stream,omitzero"`
	Temperature      float64   `json:"temperature,omitzero"`
	TopP             float64   `json:"top_p,omitzero"` // [0, 1.0]
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"`
	PresencePenalty  float64   `json:"presence_penalty,omitzero"` // [-2, 2]
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable, model string) error {
	c.Model = model
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.ChatOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				if v.Seed != 0 {
					unsupported = append(unsupported, "Seed")
				}
				if v.TopK != 0 {
					unsupported = append(unsupported, "TopK")
				}
				c.Stop = v.Stop
				if v.ReplyAsJSON {
					errs = append(errs, errors.New("unsupported option ReplyAsJSON"))
				}
				if v.DecodeAs != nil {
					errs = append(errs, errors.New("unsupported option DecodeAs"))
				}
				if len(v.Tools) != 0 {
					errs = append(errs, errors.New("unsupported option Tools"))
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
			c.Messages[0].Content = []Content{{
				Type: ContentText,
				Text: sp,
			}}
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

// Message is completely undocumented as of May 2025.
type Message struct {
	Role    string   `json:"role,omitzero"` // "system", "assistant", "user"
	Content Contents `json:"content,omitzero"`
}

func (m *Message) IsZero() bool {
	return m.Role == "" && len(m.Content) == 0
}

type Content struct {
	Type ContentType `json:"type,omitzero"`
	Text string      `json:"text,omitzero"`
}

type ContentType string

const (
	ContentText ContentType = "text"
)

// Contents represents a slice of Content with custom unmarshalling to handle
// both string and Content struct types.
type Contents []Content

func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal(([]Content)(*c))
}

// UnmarshalJSON implements custom unmarshalling for Contents type
// to handle cases where content could be a string or Content struct.
func (c *Contents) UnmarshalJSON(data []byte) error {
	// Try unmarshalling as a string first
	var contentStr string
	if err := json.Unmarshal(data, &contentStr); err == nil {
		// If it worked, create a single content with the string
		*c = Contents{{
			Type: ContentText,
			Text: contentStr,
		}}
		return nil
	}
	// If that failed, try as array of Content
	var contents []Content
	if err := json.Unmarshal(data, &contents); err == nil {
		*c = contents
		return nil
	}

	// If that failed, try as one Content
	var content Content
	err := json.Unmarshal(data, &content)
	if err == nil {
		*c = Contents{content}
	}
	return err
}

// From converts from a genai.Message to a Message.
func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) > 0 {
		m.Content = make([]Content, 0, len(in.Contents))
		for i := range in.Contents {
			if in.Contents[i].Text != "" {
				m.Content = append(m.Content, Content{
					Type: ContentText,
					Text: in.Contents[i].Text,
				})
			} else if in.Contents[i].Thinking != "" {
				// Ignore
			} else {
				// Cerebras doesn't support documents yet.
				return fmt.Errorf("unsupported content type %#v", in.Contents[i])
			}
		}
	}
	if len(in.ToolCalls) != 0 {
		return errors.New("tool calls not supported")
	}
	if len(in.ToolCallResults) != 0 {
		return errors.New("tool call results not supported")
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "user", "assistant":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	if len(m.Content) != 0 {
		out.Contents = make([]genai.Content, len(m.Content))
		for i, content := range m.Content {
			if content.Type == ContentText {
				out.Contents[i] = genai.Content{Text: content.Text}
			}
		}
	}
	return nil
}

// ChatResponse captures all the different ways providers can reply.
type ChatResponse struct {
	Message
	Message2     Message      `json:"message"`
	FinishReason FinishReason `json:"finish_reason"`
	Choices      []struct {
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) > 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c)
	}
	if len(c.Choices) == 1 {
		out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
		err := c.Choices[0].Message.To(&out.Message)
		return out, err
	}
	m := c.Message2
	if m.IsZero() {
		m = c.Message
	}
	if m.Role == "" {
		return out, fmt.Errorf("expected 1 choice, got %#v", c)
	}
	if err := m.To(&out.Message); err != nil {
		return out, err
	}
	out.FinishReason = c.FinishReason.ToFinishReason()
	return out, nil
}

type FinishReason string

const (
	FinishStop   FinishReason = "stop"
	FinishLength FinishReason = "length"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishLength:
		return genai.FinishedLength
	default:
		return genai.FinishReason(f)
	}
}

type ChatStreamChunkResponse struct {
	Delta struct {
		Text    string  `json:"text"`
		Message Message `json:"message"`
	} `json:"delta"`
	FinishReason FinishReason `json:"finish_reason"`
	Choices      []struct {
		Delta struct {
			Message
		} `json:"delta"`
		FinishReason FinishReason `json:"finish_reason"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

//

// ErrorResponse is as generic as possible since error responses are highly non-standard.
type ErrorResponse map[string]any

func (er *ErrorResponse) String() string {
	return fmt.Sprintf("error %s", *er)
}

// Client implements genai.ProviderChat.
type Client struct {
	internal.BaseChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

func (c *Client) Name() string {
	return "openaicompatible"
}

// New creates a new client to talk to an "OpenAI-compatible" platform API.
//
// It only support text exchanges (no multi-modal) and no tool calls.
//
// r can be used to throttle outgoing requests, record calls, etc. It defaults to http.DefaultTransport.
func New(chatURL string, h http.Header, model string, r http.RoundTripper) (*Client, error) {
	if r == nil {
		r = http.DefaultTransport
	}
	return &Client{
		BaseChat: internal.BaseChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			ChatURL:              chatURL,
			ModelOptional:        true,
			ProcessStreamPackets: processStreamPackets,
			Base: internal.Base[*ErrorResponse]{
				ClientJSON: httpjson.Client{
					Client: &http.Client{Transport: &roundtrippers.Header{
						Transport: &roundtrippers.Retry{
							Transport: &roundtrippers.RequestID{
								Transport: r,
							},
						},
						Header: h,
					}},
					// It is always lenient by definition.
					Lenient: true,
				},
			},
		},
	}, nil
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if pkt.Usage.TotalTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
		}
		if len(pkt.Choices) == 1 {
			if pkt.Choices[0].FinishReason != "" {
				result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
			}
			switch role := pkt.Choices[0].Delta.Role; role {
			case "", "assistant":
			default:
				return fmt.Errorf("unexpected role %q", role)
			}
			for _, content := range pkt.Choices[0].Delta.Content {
				switch content.Type {
				case ContentText:
					f := genai.MessageFragment{TextFragment: content.Text}
					if !f.IsZero() {
						if err := result.Accumulate(f); err != nil {
							return err
						}
						chunks <- f
					}
				default:
					return fmt.Errorf("unexpected content type %q", content.Type)
				}
			}
			continue
		}
		if pkt.FinishReason != "" {
			result.FinishReason = pkt.FinishReason.ToFinishReason()
		}
		m := pkt.Delta.Message
		c := pkt.Delta.Message.Content
		if m.IsZero() {
			m = pkt.Delta.Message
		}
		switch role := m.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		if m.IsZero() {
			f := genai.MessageFragment{TextFragment: pkt.Delta.Text}
			if !f.IsZero() {
				if err := result.Accumulate(f); err != nil {
					return err
				}
				chunks <- f
			}
			continue
		}
		for _, content := range c {
			f := genai.MessageFragment{TextFragment: content.Text}
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

var _ genai.ProviderChat = &Client{}
