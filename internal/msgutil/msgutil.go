// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package msgutil provides shared message utilities for subprocess-based
// providers.
package msgutil

import (
	"encoding/json"
	"errors"
	"io"

	"github.com/maruel/genai"
)

// ExtractOpaqueID extracts a string value from Reply.Opaque by walking messages
// in reverse order.
//
// It is used by subprocess-based providers to retrieve session or thread IDs
// stored in previous replies.
func ExtractOpaqueID(msgs genai.Messages, key string) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		for j := range msgs[i].Replies {
			if id, ok := msgs[i].Replies[j].Opaque[key].(string); ok && id != "" {
				return id
			}
		}
	}
	return ""
}

// LastUserMsg returns the last user message in msgs.
//
// It is used by subprocess-based providers that send only the final user
// message to a subprocess, relying on session resumption for context.
func LastUserMsg(msgs genai.Messages) (genai.Message, error) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role() == "user" {
			if len(msgs[i].Requests) == 0 {
				return genai.Message{}, errors.New("last user message has no content")
			}
			return msgs[i], nil
		}
	}
	return genai.Message{}, errors.New("no user message found in msgs")
}

// WriteNDJSON marshals v as JSON and writes it followed by a newline.
//
// It is used by subprocess-based providers that communicate via NDJSON over
// stdin/stdout.
func WriteNDJSON(w io.Writer, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = w.Write(data)
	return err
}
