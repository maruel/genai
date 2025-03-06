// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import "context"

type Backend interface {
	Query(ctx context.Context, systemPrompt, query string) (string, error)
	QueryContent(ctx context.Context, systemPrompt, query, mime string, content []byte) (string, error)
}
