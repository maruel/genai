// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package anthropic implements a client for the Anthropic API, to use Claude.
//
// It is described at
// https://docs.anthropic.com/en/api/
package anthropic

// See official client at https://github.com/anthropics/anthropic-sdk-go

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
)

// https://docs.anthropic.com/en/api/messages
type CompletionRequest struct {
	Model    string    `json:"model,omitzero"`
	MaxToks  int64     `json:"max_tokens"`
	Messages []Message `json:"messages"`
	Metadata struct {
		UserID string `json:"user_id,omitzero"`
	} `json:"metadata,omitzero"`
	StopSequences []string   `json:"stop_sequences,omitzero"`
	Stream        bool       `json:"stream,omitzero"`
	System        string     `json:"system,omitzero"`
	Temperature   float64    `json:"temperature,omitzero"` // [0, 1]
	Thinking      Thinking   `json:"thinking,omitzero"`
	ToolChoice    ToolChoice `json:"tool_choice,omitzero"`
	Tools         []Tool     `json:"tools,omitzero"`
	TopK          int64      `json:"top_k,omitzero"` // [1, ]
	TopP          float64    `json:"top_p,omitzero"` // [0, 1]
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *CompletionRequest) Init(msgs genai.Messages, opts genai.Validatable) error {
	var errs []error
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.CompletionOptions:
				c.MaxToks = v.MaxTokens
				c.Temperature = v.Temperature
				c.System = v.SystemPrompt
				c.TopP = v.TopP
				if v.Seed != 0 {
					errs = append(errs, errors.New("anthropic doesn't support seed"))
				}
				c.TopK = v.TopK
				c.StopSequences = v.Stop
				if v.ReplyAsJSON || v.DecodeAs != nil {
					errs = append(errs, errors.New("anthropic doesn't support JSON schema"))
				}
				if len(v.Tools) != 0 {
					c.ToolChoice.Type = "any"
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						// Weirdly enough, we must not set it. See example at https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview
						// c.Tools[i].Type = "custom"
						c.Tools[i].Name = t.Name
						c.Tools[i].Description = t.Description
						if t.InputsAs != nil {
							c.Tools[i].InputSchema = jsonschema.Reflect(t.InputsAs)
						}
						// Unclear if this has any impact: c.Tools[i].CacheControl.Type = "ephemeral"
					}
				}
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}
	if c.MaxToks == 0 {
		// TODO: Query the model. Anthropic requires a value! Use the lowest common denominator for 3.5+.
		c.MaxToks = 8192
	}

	if err := msgs.Validate(); err != nil {
		errs = append(errs, err)
	} else {
		c.Messages = make([]Message, 0, len(msgs))
		for i := range msgs {
			switch msgs[i].Role {
			case genai.User, genai.Assistant:
				c.Messages = append(c.Messages, Message{})
				if err := c.Messages[len(c.Messages)-1].From(&msgs[i]); err != nil {
					errs = append(errs, fmt.Errorf("message %d: %w", i, err))
				}
			default:
				errs = append(errs, fmt.Errorf("message %d: unsupported role %q", i, msgs[i].Role))
				continue
			}
		}
	}
	return errors.Join(errs...)
}

// https://docs.anthropic.com/en/api/messages
type Message struct {
	Role string `json:"role"` // "assistant", "user"
	// Anthropic's Content doesn't distinguish between actual content (text,
	// documents) and tool use.
	Content []Content `json:"content"`
}

func (m *Message) From(in *genai.Message) error {
	switch role := in.Role; role {
	case genai.Assistant, genai.User:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	m.Content = make([]Content, len(in.Contents)+len(in.ToolCalls))
	for i := range in.Contents {
		if err := m.Content[i].FromContent(&in.Contents[i]); err != nil {
			return fmt.Errorf("block %d: %w", i, err)
		}
	}
	offset := len(in.Contents)
	for i := range in.ToolCalls {
		if err := m.Content[offset+i].FromToolCall(&in.ToolCalls[i]); err != nil {
			return fmt.Errorf("block %d: %w", offset+i, err)
		}
	}
	return nil
}

func (m *Message) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant", "user":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	// We need to split actual content and tool calls.
	for i := range m.Content {
		switch m.Content[i].Type {
		case "text":
			out.Contents = append(out.Contents, genai.Content{})
			if err := m.Content[i].ToContent(&out.Contents[len(out.Contents)-1]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		case "tool_use":
			out.ToolCalls = append(out.ToolCalls, genai.ToolCall{})
			if err := m.Content[i].ToToolCall(&out.ToolCalls[len(out.ToolCalls)-1]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		default:
			return fmt.Errorf("unsupported content type %q", m.Content[i].Type)
		}
	}
	return nil
}

type Content struct {
	Type string `json:"type"` // "text", "image", "tool_use", "tool_result", "document"
	// Type == "text"
	Text string `json:"text,omitzero"`

	// Type == "text", "image", "tool_use", "tool_result", "document"
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`

	// Type == "text", "document"
	Citations Citation `json:"citations,omitzero"`

	// Type == "image", "document"
	Source struct {
		// Content.Type == "image": "base64", "url"
		// Content.Type == "document": "base64, "url", "text", "content"
		Type string `json:"type,omitzero"`

		// Type == "base64", "url", "text"
		// Content.Type == "image": "image/jpeg", "image/png", "image/gif", "image/webp"
		// Content.Type == "document": "application/pdf"
		MediaType string `json:"media_type,omitzero"`

		// Type == "base64", "text"
		Data []byte `json:"data,omitzero"` // base64 encoded if base64, else as is.

		// Type == "url"
		URL string `json:"url,omitzero"`

		// Type == "content"
		// Only "text" and "image" are allowed.
		Content []Content `json:"content,omitzero"`
	} `json:"source,omitzero"`

	// Type == "tool_use"
	ID    string `json:"id,omitzero"`
	Input any    `json:"input,omitzero"`
	Name  string `json:"name,omitzero"`

	// Type == "tool_result"
	ToolUseID string `json:"tool_use_id,omitzero"`
	IsError   bool   `json:"is_error,omitzero"`
	// Only "text" and "image" are allowed.
	Content []Content `json:"content,omitzero"`

	// "document"
	Context string `json:"context,omitzero"`
	Title   string `json:"title,omitzero"`
}

func (c *Content) FromContent(in *genai.Content) error {
	if in.Text != "" {
		c.Type = "text"
		c.Text = in.Text
		return nil
	}

	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	// Anthropic require a mime-type to determine if image or PDF.
	if mimeType == "" {
		return fmt.Errorf("unspecified mime type for URL %q", in.URL)
	}
	c.CacheControl.Type = "ephemeral"
	switch {
	case strings.HasPrefix(mimeType, "image/"):
		c.Type = "image"
		if in.URL != "" {
			c.Source.Type = "url"
			c.Source.URL = in.URL
		} else {
			c.Source.MediaType = mimeType
			c.Source.Type = "base64"
			c.Source.Data = data
		}
	case mimeType == "application/pdf":
		c.Type = "document"
		if in.URL != "" {
			c.Source.Type = "url"
			c.Source.URL = in.URL
		} else {
			c.Source.MediaType = mimeType
			c.Source.Type = "base64"
			c.Source.Data = data
		}
	default:
		return fmt.Errorf("unsupported content mime-type %s", mimeType)
	}
	return nil
}

func (c *Content) FromToolCall(in *genai.ToolCall) error {
	c.Type = "tool_use"
	c.ID = in.ID
	c.Name = in.Name
	if err := json.Unmarshal([]byte(in.Arguments), &c.Input); err != nil {
		return fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, in)
	}
	return nil
}

func (c *Content) ToContent(out *genai.Content) error {
	switch c.Type {
	case "text":
		out.Text = c.Text
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

func (c *Content) ToToolCall(out *genai.ToolCall) error {
	switch c.Type {
	case "tool_use":
		out.ID = c.ID
		out.Name = c.Name
		raw, err := json.Marshal(c.Input)
		if err != nil {
			return fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, c)
		}
		out.Arguments = string(raw)
	default:
		return fmt.Errorf("unsupported content type %q", c.Type)
	}
	return nil
}

type Citation struct {
	// Content.Type == "text"
	CitedText     string `json:"cited_text,omitzero"`
	DocumentIndex int64  `json:"document_index,omitzero"`
	DocumentTitle string `json:"document_title,omitzero"`
	Type          string `json:"type,omitzero"` // "char_location", "page_location", "content_block_location"
	// Type == "char_location"
	EndCharIndex   int64 `json:"end_char_index,omitzero"`
	StartCharIndex int64 `json:"start_char_index,omitzero"`
	// Type == "page_location"
	EndPageNumber   int64 `json:"end_page_number,omitzero"`
	StartPageNumber int64 `json:"start_page_number,omitzero"`
	// Type == "content_block_location"
	EndBlockIndex   int64 `json:"end_block_index,omitzero"`
	StartBlockIndex int64 `json:"start_block_index,omitzero"`

	// Content.Type == "document"
	Enabled bool `json:"enabled,omitzero"`
}

type Thinking struct {
	BudgetTokens int64  `json:"budget_tokens,omitzero"` // >1024 and less than max_tokens
	Type         string `json:"type,omitzero"`          // "enabled", "disabled"
}

type ToolChoice struct {
	Type string `json:"type,omitzero"` // "auto", "any", "tool", "none"

	// Type == "auto", "any", "tool"
	DisableParallelToolUse bool `json:"disable_parallel_tool_use,omitzero"` // default false

	// Type == "tool"
	Name string `json:"name,omitzero"`
}

// https://docs.anthropic.com/en/api/messages#body-tools
type Tool struct {
	Type string `json:"type,omitzero"` // "custom", "computer_20241022", "computer_20250124", "bash_20241022", "bash_20250124", "text_editor_20241022", "text_editor_20250124"
	// Type == "custom"
	Description string             `json:"description,omitzero"`
	InputSchema *jsonschema.Schema `json:"input_schema,omitzero"`

	// Type == "custom": tool name
	// Type == "computer_20241022", "computer_20250124": "computer"
	// Type == "bash_20241022", "bash_20250124": "bash"
	// Type == "text_editor_20241022", "text_editor_20250124": "str_replace_editor"
	Name string `json:"name,omitzero"`

	// Type == "custom", "computer_20241022", "computer_20250124", "bash_20241022", "bash_20250124", "text_editor_20241022", "text_editor_20250124"
	CacheControl struct {
		Type string `json:"type,omitzero"` // "ephemeral"
	} `json:"cache_control,omitzero"`

	// Type == "computer_20241022", "computer_20250124"
	DisplayNumber   int64 `json:"display_number,omitzero"`
	DisplayHeightPX int64 `json:"display_height_px,omitzero"`
	DisplayWidthPX  int64 `json:"display_width_px,omitzero"`
}

type CompletionResponse struct {
	Message             // Role is always "assistant"
	ID           string `json:"id"`
	Model        string `json:"model"`
	StopReason   string `json:"stop_reason"` // "end_turn", "tool_use", "stop_sequence", "max_tokens"
	StopSequence string `json:"stop_sequence"`
	Type         string `json:"type"` // "message"
	Usage        struct {
		InputTokens              int64 `json:"input_tokens"`
		OutputTokens             int64 `json:"output_tokens"`
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	} `json:"usage"`
}

func (c *CompletionResponse) ToResult() (genai.CompletionResult, error) {
	out := genai.CompletionResult{
		Usage: genai.Usage{
			InputTokens:  c.Usage.InputTokens,
			OutputTokens: c.Usage.OutputTokens,
		},
	}
	err := c.To(&out.Message)
	return out, err
}

// https://docs.anthropic.com/en/api/messages-streaming
// Each stream uses the following event flow:
//   - message_start: contains a Message object with empty content.
//   - A series of content blocks, each of which have a content_block_start, one
//     or more content_block_delta events, and a content_block_stop event. Each
//     content block will have an index that corresponds to its index in the final
//     Message content array.
//   - One or more message_delta events, indicating top-level changes to the
//     final Message object.
//   - A final message_stop event.
type CompletionStreamChunkResponse struct {
	Type string `json:"type"` // // "message_start", "content_block_start", "content_block_delta", "mesage_delta", "message_stop"

	// Type == "message_start"
	Message struct {
		ID           string   `json:"id"`
		Type         string   `json:"type"` // "message"
		Role         string   `json:"role"`
		Model        string   `json:"model"`
		Content      []string `json:"content"`
		StopReason   string   `json:"stop_reason"`
		StopSequence string   `json:"stop_sequence"`
		Usage        struct {
			InputTokens              int64 `json:"input_tokens"`
			OutputTokens             int64 `json:"output_tokens"`
			CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
		} `json:"usage"`
	} `json:"message"`

	Index int64 `json:"index"`

	// Type == "content_block_start"
	ContentBlock struct {
		Type string `json:"type"` // "text", "tool_use", "thinking"

		// Type == "text"
		Text string `json:"text"`

		// Type == "tool_use"
		ID    string `json:"id"`
		Name  string `json:"name"`
		Input any    `json:"input"`
	} `json:"content_block"`

	// Type == "content_block_delta"
	Delta struct {
		Type string `json:"type"` // "text_delta", "input_json_delta", "thinking_delta", "signature_delta", ""

		// Type == "text_delta"
		Text string `json:"text"`

		// Type == "input_json_delta"
		PartialJSON string `json:"partial_json"`

		// Type == "thinking_delta"
		Thinking string `json:"thinking"`

		// Type == "signature_delta"
		Signature []byte `json:"signature"`

		// Type == ""
		StopReason   string `json:"stop_reason"` // "end_turn", "tool_use", "stop_sequence", "max_tokens"
		StopSequence string `json:"stop_sequence"`
	} `json:"delta"`
	Usage struct {
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
}

//

type errorResponse struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// Client implements the REST JSON based API.
type Client struct {
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	model string
}

// New creates a new client to talk to the Anthropic platform API.
//
// If apiKey is not provided, it tries to load it from the ANTHROPIC_API_KEY environment variable.
// If none is found, it returns an error.
// Get an API key at https://console.anthropic.com/settings/keys
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.anthropic.com/en/docs/about-claude/models/all-models
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("ANTHROPIC_API_KEY"); apiKey == "" {
			return nil, errors.New("anthropic API key is required; get one at " + apiKeyURL)
		}
	}
	// Anthropic doesn't support HTTP POST compression.
	h := http.Header{"x-api-key": {apiKey}, "anthropic-version": {"2023-06-01"}}
	return &Client{model: model, Client: httpjson.Client{DefaultHeader: h}}, nil
}

func (c *Client) Completion(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.CompletionResult, error) {
	// https://docs.anthropic.com/en/api/messages
	rpcin := CompletionRequest{Model: c.model}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genai.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.CompletionResult{}, err
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.post(ctx, "https://api.anthropic.com/v1/messages", in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) error {
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks)
	})
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	return err
}

func processStreamPackets(ch <-chan CompletionStreamChunkResponse, chunks chan<- genai.MessageFragment) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		word := ""
		switch pkt.Type {
		case "message_start":
			switch pkt.Message.Role {
			case "assistant":
			default:
				return fmt.Errorf("unexpected role %q", pkt.Message.Role)
			}
		case "content_block_start":
			word = pkt.ContentBlock.Text
		case "content_block_delta":
			word = pkt.Delta.Text
		}
		// slog.DebugContext(ctx, "anthropic", "word", word, "duration", time.Since(start).Round(time.Millisecond))
		if word != "" {
			chunks <- genai.MessageFragment{TextFragment: word}
		}
	}
	return nil
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	resp, err := c.Client.PostRequest(ctx, "https://api.anthropic.com/v1/messages", nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	for r := bufio.NewReader(resp.Body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) != 0 {
			if err := parseStreamLine(line, out); err != nil {
				return err
			}
		}
	}
}

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
	// https://docs.anthropic.com/en/api/messages-streaming
	// and
	// https://developer.mozilla.org/en-US/docs/Web/API/Server-sent%5Fevents/Using%5Fserver-sent%5Fevents
	const dataPrefix = "data: "
	switch {
	case bytes.HasPrefix(line, []byte(dataPrefix)):
		suffix := string(line[len(dataPrefix):])
		if suffix == "[DONE]" {
			return nil
		}
		d := json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		d.UseNumber()
		msg := CompletionStreamChunkResponse{}
		if err := d.Decode(&msg); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		out <- msg
	case bytes.HasPrefix(line, []byte("event:")):
		// Ignore for now.
	default:
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		d.UseNumber()
		er := errorResponse{}
		if err := d.Decode(&er); err == nil {
			return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
	}
	return nil
}

type Model struct {
	CreatedAt   time.Time `json:"created_at"`
	DisplayName string    `json:"display_name"`
	ID          string    `json:"id"`
	Type        string    `json:"type"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s: %s (%s)", m.ID, m.DisplayName, m.CreatedAt.Format("2006-01-02"))
}

func (m *Model) Context() int64 {
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://docs.anthropic.com/en/api/models-list
	var out struct {
		Data    []Model `json:"data"`
		FirstID string  `json:"first_id"`
		HasMore bool    `json:"has_more"`
		LastID  string  `json:"last_id"`
	}
	err := c.Client.Get(ctx, "https://api.anthropic.com/v1/models?limit=1000", nil, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(out.Data))
	for i := range out.Data {
		models[i] = &out.Data[i]
	}
	return models, err
}

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	// Anthropic doesn't HTTP POST support compression.
	resp, err := c.Client.PostRequest(ctx, url, nil, in)
	if err != nil {
		return err
	}
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: error %s: %s. You can get a new API key at %s", herr, er.Error.Type, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error %s: %s", herr, er.Error.Type, er.Error.Message)
		}
		return fmt.Errorf("error %s: %s", er.Error.Type, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "anthropic", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "anthropic", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://console.anthropic.com/settings/keys"

var (
	_ genai.CompletionProvider = &Client{}
	_ genai.ModelProvider      = &Client{}
)
