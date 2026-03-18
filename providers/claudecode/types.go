// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package claudecode

import "encoding/json"

// NDJSON message types for the Claude Code CLI stream-json protocol.
//
// Each line on stdout is one of these types, discriminated by the "type" field.
// Each line on stdin is an inputMsg.

// inputMsg is written to stdin to send a user message.
type inputMsg struct {
	Type    string       `json:"type"`    // always "user"
	Message inputContent `json:"message"` // the user content
}

// inputContent holds the user message payload.
// Content is a plain string for text-only messages, or []inputBlock for
// multi-modal messages that include images.
type inputContent struct {
	Role    string `json:"role"`    // always "user"
	Content any    `json:"content"` // string or []inputBlock
}

// inputBlock is a single content block inside a multi-modal user message.
type inputBlock struct {
	Type   string       `json:"type"`             // "text" or "image"
	Text   string       `json:"text,omitempty"`   // Type == "text"
	Source *inputSource `json:"source,omitempty"` // Type == "image"
}

// inputSource describes the data source for an image content block.
type inputSource struct {
	Type      string `json:"type"`                 // "base64" or "url"
	MediaType string `json:"media_type,omitempty"` // e.g. "image/png"
	Data      string `json:"data,omitempty"`       // base64-encoded bytes
	URL       string `json:"url,omitempty"`        // Type == "url"
}

// baseMsg is used to discriminate the type of each stdout line.
type baseMsg struct {
	Type    string `json:"type"`
	Subtype string `json:"subtype,omitempty"`
}

// systemInitMsg is the first message emitted on stdout, type="system" subtype="init".
type systemInitMsg struct {
	Type      string   `json:"type"`
	Subtype   string   `json:"subtype"`
	SessionID string   `json:"session_id"`
	Cwd       string   `json:"cwd,omitzero"`
	Model     string   `json:"model"`
	Tools     []string `json:"tools"`
	Version   string   `json:"claude_code_version"`
}

// assistantMsg carries a complete assistant response, type="assistant".
type assistantMsg struct {
	Type    string     `json:"type"`
	Message apiMessage `json:"message"`
}

// apiMessage is the inner message from the model.
type apiMessage struct {
	Content    []contentBlock `json:"content"`
	StopReason string         `json:"stop_reason,omitzero"`
	Usage      msgUsage       `json:"usage"`
}

// contentBlock is a single block within an apiMessage.
// Type is one of "text", "thinking", "tool_use".
// Only "text" is surfaced in v1; others become Opaque.
type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`

	// tool_use fields (not surfaced in v1)
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// thinking fields (not surfaced in v1)
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
}

// resultMsg is the terminal message emitted when the session ends, type="result".
type resultMsg struct {
	Type          string   `json:"type"`
	Subtype       string   `json:"subtype"`
	IsError       bool     `json:"is_error"`
	Result        string   `json:"result"`
	Errors        []string `json:"errors"`
	StopReason    string   `json:"stop_reason,omitzero"`
	SessionID     string   `json:"session_id"`
	Usage         msgUsage `json:"usage"`
	TotalCostUSD  float64  `json:"total_cost_usd"`
	DurationMs    int64    `json:"duration_ms,omitzero"`
	DurationAPIMs int64    `json:"duration_api_ms,omitzero"`
	NumTurns      int      `json:"num_turns,omitzero"`
}

// msgUsage holds token counts from the model.
type msgUsage struct {
	InputTokens              int64  `json:"input_tokens"`
	OutputTokens             int64  `json:"output_tokens"`
	CacheCreationInputTokens int64  `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int64  `json:"cache_read_input_tokens"`
	ServiceTier              string `json:"service_tier"`
}

// streamEventMsg carries a partial streaming update, type="stream_event".
// Only emitted when --include-partial-messages is set.
type streamEventMsg struct {
	Type  string      `json:"type"`
	Event streamEvent `json:"event"`
}

// streamEvent holds the event data for a single streaming update.
type streamEvent struct {
	Type         string          `json:"type"`                    // "content_block_start", "content_block_delta", "content_block_stop", "message_start", "message_stop", "error", etc.
	Index        int             `json:"index"`                   // content block index
	Delta        *streamDelta    `json:"delta,omitempty"`         // present for content_block_delta
	ContentBlock json.RawMessage `json:"content_block,omitempty"` // present for content_block_start
}

// streamDelta is the payload inside a content_block_delta event.
type streamDelta struct {
	Type        string `json:"type"` // "text_delta", "thinking_delta", "input_json_delta", "signature_delta"
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
}
