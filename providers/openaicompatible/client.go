// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openaicompatible implements a minimal client for "OpenAI-compatible" providers.
//
// It's a good starting point to implement a client for a new platform.
package openaicompatible

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for generic OpenAI compatible API.
var Scoreboard = scoreboard.Score{
	Scenarios: []scoreboard.Scenario{
		{
			In:        map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:       map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync:   &scoreboard.FunctionalityText{},
			GenStream: &scoreboard.FunctionalityText{},
		},
	},
}

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
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return err
		}
		switch v := opts.(type) {
		case *genai.OptionsText:
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
			if v.TopLogprobs > 0 {
				unsupported = append(unsupported, "TopLogprobs")
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

	offset := 0
	if sp != "" {
		offset = 1
	}
	c.Messages = make([]Message, len(msgs)+offset)
	if sp != "" {
		c.Messages[0] = Message{Role: "system", Content: []Content{{Type: ContentText, Text: sp}}}
	}
	for i := range msgs {
		if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
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
func (c *Contents) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*c = nil
		return nil
	}
	if err := json.Unmarshal(b, (*[]Content)(c)); err == nil {
		return nil
	}

	v := Content{}
	if err := json.Unmarshal(b, &v); err == nil {
		*c = Contents{v}
		return nil
	}

	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*c = Contents{{Type: ContentText, Text: s}}
	return nil
}

// From converts from a genai.Message to a Message.
func (m *Message) From(in *genai.Message) error {
	switch r := in.Role(); r {
	case "user", "assistant":
		m.Role = r
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	if len(in.Requests) > 0 {
		m.Content = make([]Content, 0, len(in.Requests))
		for i := range in.Requests {
			if in.Requests[i].Text != "" {
				m.Content = append(m.Content, Content{Type: ContentText, Text: in.Requests[i].Text})
			} else if !in.Requests[i].Doc.IsZero() {
				// Check if this is a text/plain document
				mimeType, data, err := in.Requests[i].Doc.Read(10 * 1024 * 1024)
				if err != nil {
					return fmt.Errorf("request #%d: failed to read document: %w", i, err)
				}
				if !strings.HasPrefix(mimeType, "text/plain") {
					return fmt.Errorf("request #%d: openaicompatible only supports text/plain documents, got %s", i, mimeType)
				}
				if in.Requests[i].Doc.URL != "" {
					return fmt.Errorf("request #%d: text/plain documents must be provided inline, not as a URL", i)
				}
				m.Content = append(m.Content, Content{Type: ContentText, Text: string(data)})
			} else {
				return fmt.Errorf("request #%d: unknown Request type", 0)
			}
		}
		for i := range in.Replies {
			if len(in.Replies[i].Opaque) != 0 {
				return fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)
			}
			if in.Replies[i].Text != "" {
				m.Content = append(m.Content, Content{Type: ContentText, Text: in.Replies[i].Text})
			} else if in.Replies[i].Thinking != "" {
				// Ignore
			} else if !in.Replies[i].Doc.IsZero() {
				// Check if this is a text/plain document
				mimeType, data, err := in.Replies[i].Doc.Read(10 * 1024 * 1024)
				if err != nil {
					return fmt.Errorf("reply #%d: failed to read document: %w", i, err)
				}
				if !strings.HasPrefix(mimeType, "text/plain") {
					return fmt.Errorf("reply #%d: openaicompatible only supports text/plain documents, got %s", i, mimeType)
				}
				if in.Replies[i].Doc.URL != "" {
					return fmt.Errorf("reply #%d: text/plain documents must be provided inline, not as a URL", i)
				}
				m.Content = append(m.Content, Content{Type: ContentText, Text: string(data)})
			} else {
				return fmt.Errorf("reply #%d: unknown Reply type", i)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		return errors.New("tool call results not supported")
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	if len(m.Content) != 0 {
		out.Replies = make([]genai.Reply, len(m.Content))
		for i, content := range m.Content {
			if content.Type == ContentText {
				out.Replies[i] = genai.Reply{Text: content.Text}
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
			TotalTokens:  c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) > 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c)
	}
	if len(c.Choices) == 1 {
		out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
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
	out.Usage.FinishReason = c.FinishReason.ToFinishReason()
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
		if !internal.BeLenient {
			panic(f)
		}
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

func (er *ErrorResponse) Error() string {
	return fmt.Sprintf("%s", *er)
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to an "OpenAI-compatible" platform API.
//
// It only support text exchanges (no multi-modal) and no tool calls.
//
// Option Remote must be set.
//
// Automatic model selection via ModelCheap, ModelGood, ModelSOTA is not supported and it will specify no
// model in this case.
//
// Exceptionally it will interpret Model set to "" as no model to specify since there is no automatic model
// selection.
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.APIKey != "" {
		return nil, errors.New("unexpected option APIKey")
	}
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote == "" {
		return nil, errors.New("option Remote is required")
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(opts.Modalities) != 0 && !slices.Equal(opts.Modalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	model := opts.Model
	switch model {
	case "", genai.ModelNone, genai.ModelCheap, genai.ModelGood, genai.ModelSOTA:
		model = ""
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	return &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           opts.Remote,
			ModelOptional:        true,
			ProcessStreamPackets: processStreamPackets,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				Modalities: mod,
				ClientJSON: httpjson.Client{
					// It is always lenient by definition.
					Lenient: true,
					Client:  &http.Client{Transport: &roundtrippers.RequestID{Transport: t}},
				},
			},
		},
	}, nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "openaicompatible"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// Modalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) Modalities() genai.Modalities {
	return c.impl.Modalities
}

// Scoreboard implements scoreboard.ProviderScore.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	return c.impl.GenStream(ctx, msgs, chunks, opts)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	return c.impl.GenStreamRaw(ctx, in, out)
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if pkt.Usage.TotalTokens != 0 {
			result.Usage.InputTokens = pkt.Usage.PromptTokens
			result.Usage.OutputTokens = pkt.Usage.CompletionTokens
			result.Usage.TotalTokens = pkt.Usage.TotalTokens
		}
		if len(pkt.Choices) == 1 {
			if pkt.Choices[0].FinishReason != "" {
				result.Usage.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
			}
			switch role := pkt.Choices[0].Delta.Role; role {
			case "", "assistant":
			default:
				return fmt.Errorf("unexpected role %q", role)
			}
			for _, content := range pkt.Choices[0].Delta.Content {
				switch content.Type {
				case ContentText:
					f := genai.ReplyFragment{TextFragment: content.Text}
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
			result.Usage.FinishReason = pkt.FinishReason.ToFinishReason()
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
			f := genai.ReplyFragment{TextFragment: pkt.Delta.Text}
			if !f.IsZero() {
				if err := result.Accumulate(f); err != nil {
					return err
				}
				chunks <- f
			}
			continue
		}
		for _, content := range c {
			f := genai.ReplyFragment{TextFragment: content.Text}
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

var _ genai.Provider = &Client{}
