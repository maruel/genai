// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genaiapi

import "context"

// ChatProvider is the generic interface to interact with a LLM backend.
type ChatProvider interface {
	Completion(ctx context.Context, msgs []Message, maxtoks, seed int, temperature float64) (string, error)
	CompletionStream(ctx context.Context, msgs []Message, maxtoks, seed int, temperature float64, words chan<- string) (string, error)
	CompletionContent(ctx context.Context, msgs []Message, maxtoks, seed int, temperature float64, mime string, content []byte) (string, error)
}

// Role is one of the LLM known roles.
type Role string

// LLM known roles. Not all systems support all roles.
const (
	System         Role = "system"
	User           Role = "user"
	Assistant      Role = "assistant"
	AvailableTools Role = "available_tools"
	ToolCall       Role = "tool_call"
	ToolCallResult Role = "tool_call_result"
)

// Message is a message to send to the LLM as part of the exchange.
type Message struct {
	Role    Role   `json:"role"`
	Content string `json:"content"`
}
