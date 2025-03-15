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

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
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

func (c *CompletionRequest) Init(msgs []genaiapi.Message, opts any) error {
	var errs []error
	if opts != nil {
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			c.MaxToks = v.MaxTokens
			c.Temperature = v.Temperature
			if v.Seed != 0 {
				errs = append(errs, errors.New("anthropic doesn't support seed"))
			}
			c.TopP = v.TopP
			c.TopK = v.TopK
			c.StopSequences = v.Stop
			if v.ReplyAsJSON || !v.JSONSchema.IsZero() {
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
					c.Tools[i].InputSchema = t.Parameters
					// Unclear if this has any impact: c.Tools[i].CacheControl.Type = "ephemeral"
				}
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}
	if c.MaxToks == 0 {
		// TODO: Query the model. Anthropic requires a value! Use the lowest common denominator for 3.5+.
		c.MaxToks = 8192
	}

	if err := genaiapi.ValidateMessages(msgs); err != nil {
		errs = append(errs, err)
	} else {
		c.Messages = make([]Message, 0, len(msgs))
		for i, m := range msgs {
			switch m.Role {
			case genaiapi.System:
				// System prompt is passed differently.
				c.System = m.Text
				continue
			case genaiapi.User, genaiapi.Assistant:
				c.Messages = append(c.Messages, Message{})
				if err := c.Messages[len(c.Messages)-1].From(m); err != nil {
					errs = append(errs, fmt.Errorf("message %d: %w", i, err))
				}
			default:
				errs = append(errs, fmt.Errorf("message %d: unsupported role %q", i, m.Role))
				continue
			}
		}
	}
	return errors.Join(errs...)
}

// https://docs.anthropic.com/en/api/messages
type Message struct {
	Role    string    `json:"role"`
	Content []Content `json:"content"`
}

func (msg *Message) From(m genaiapi.Message) error {
	msg.Role = string(m.Role)
	// TODO: An Anthropic message can contain multiple content blocks.
	msg.Content = []Content{{}}
	switch m.Type {
	case genaiapi.Text:
		msg.Content[0].Type = "text"
		msg.Content[0].Text = m.Text
	case genaiapi.Document:
		mimeType, data, err := internal.ParseDocument(&m, 10*1024*1024)
		if err != nil {
			return err
		}
		// Anthropic require a mime-type to determine if image or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", m.URL)
		}
		msg.Content[0].CacheControl.Type = "ephemeral"
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			msg.Content[0].Type = "image"
			if m.URL != "" {
				msg.Content[0].Source.Type = "url"
				msg.Content[0].Source.URL = m.URL
			} else {
				msg.Content[0].Source.MediaType = mimeType
				msg.Content[0].Source.Type = "base64"
				msg.Content[0].Source.Data = data
			}
		case mimeType == "application/pdf":
			msg.Content[0].Type = "document"
			if m.URL != "" {
				msg.Content[0].Source.Type = "url"
				msg.Content[0].Source.URL = m.URL
			} else {
				msg.Content[0].Source.MediaType = mimeType
				msg.Content[0].Source.Type = "base64"
				msg.Content[0].Source.Data = data
			}
		default:
			return fmt.Errorf("unsupported content mime-type %s", mimeType)
		}
	default:
		return fmt.Errorf("unsupported content type %s", m.Type)
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
	Description string              `json:"description,omitzero"`
	InputSchema genaiapi.JSONSchema `json:"input_schema,omitzero"`

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
	Content      []Content `json:"content"`
	ID           string    `json:"id"`
	Model        string    `json:"model"`
	Role         string    `json:"role"`        // Always "assistant"
	StopReason   string    `json:"stop_reason"` // "end_turn", "tool_use", "stop_sequence", "max_tokens"
	StopSequence string    `json:"stop_sequence"`
	Type         string    `json:"type"` // "message"
	Usage        struct {
		InputTokens              int64 `json:"input_tokens"`
		OutputTokens             int64 `json:"output_tokens"`
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
	} `json:"usage"`
}

func (c *CompletionResponse) ToResult() (genaiapi.CompletionResult, error) {
	out := genaiapi.CompletionResult{}
	out.InputTokens = c.Usage.InputTokens
	out.OutputTokens = c.Usage.OutputTokens
	if len(c.Content) != 1 {
		return out, fmt.Errorf("expected 1 choice, got %#v", c.Content)
	}
	switch c.Content[0].Type {
	case "text":
		out.Type = genaiapi.Text
		out.Text = c.Content[0].Text
	case "tool_use":
		out.Type = genaiapi.ToolCalls
		raw, err := json.Marshal(c.Content[0].Input)
		if err != nil {
			return out, fmt.Errorf("failed to marshal input: %w; for tool call: %#v", err, c.Content[0])
		}
		out.ToolCalls = []genaiapi.ToolCall{{
			ID:        c.Content[0].ID,
			Name:      c.Content[0].Name,
			Arguments: string(raw),
		}}
	default:
		return out, fmt.Errorf("unsupported content type %q", c.Content[0].Type)
	}
	switch role := c.Role; role {
	case "system", "assistant", "user":
		out.Role = genaiapi.Role(role)
	default:
		return out, fmt.Errorf("unsupported role %q", role)
	}
	return out, nil
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
	apiKey string
	model  string
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
	return &Client{apiKey: apiKey, model: model}, nil
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (genaiapi.CompletionResult, error) {
	// https://docs.anthropic.com/en/api/messages
	rpcin := CompletionRequest{Model: c.model}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genaiapi.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genaiapi.CompletionResult{}, err
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

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, chunks chan<- genaiapi.MessageChunk) error {
	in := CompletionRequest{Model: c.model}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	start := time.Now()
	go func() {
		lastRole := genaiapi.System
		for pkg := range ch {
			word := ""
			switch pkg.Type {
			case "message_start":
				switch pkg.Message.Role {
				case "system", "assistant", "user":
					lastRole = genaiapi.Role(pkg.Message.Role)
				case "":
				default:
					cancel()
					// We need to empty the channel to avoid blocking the goroutine.
					for range ch {
					}
					end <- fmt.Errorf("unexpected role %q", pkg.Message.Role)
					return
				}
			case "content_block_start":
				word = pkg.ContentBlock.Text
			case "content_block_delta":
				word = pkg.Delta.Text
			}
			slog.DebugContext(ctx, "anthropic", "word", word, "duration", time.Since(start).Round(time.Millisecond))
			if word != "" {
				chunks <- genaiapi.MessageChunk{Role: lastRole, Type: genaiapi.Text, Text: word}
			}
		}
		end <- nil
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	<-end
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	h := make(http.Header)
	h.Set("x-api-key", c.apiKey)
	h.Set("anthropic-version", "2023-06-01")
	// Anthropic doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, "https://api.anthropic.com/v1/messages", h, in)
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

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://docs.anthropic.com/en/api/models-list
	h := make(http.Header)
	h.Set("x-api-key", c.apiKey)
	h.Set("anthropic-version", "2023-06-01")
	var out struct {
		Data    []Model `json:"data"`
		FirstID string  `json:"first_id"`
		HasMore bool    `json:"has_more"`
		LastID  string  `json:"last_id"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://api.anthropic.com/v1/models?limit=1000", h, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out.Data))
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
	h := make(http.Header)
	h.Set("x-api-key", c.apiKey)
	h.Set("anthropic-version", "2023-06-01")
	// Anthropic doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
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
	_ genaiapi.CompletionProvider = &Client{}
	_ genaiapi.ModelProvider      = &Client{}
)
