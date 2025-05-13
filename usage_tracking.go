// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"context"
	"sync"
)

// UsageTrackingChatProvider wraps a ChatProvider and accumulates Usage values
// across multiple requests to track total token consumption.
type UsageTrackingChatProvider struct {
	Provider ChatProvider

	mu         sync.Mutex
	accumUsage Usage
}

// Chat implements the ChatProvider interface and accumulates usage statistics.
func (p *UsageTrackingChatProvider) Chat(ctx context.Context, msgs Messages, opts Validatable) (ChatResult, error) {
	result, err := p.Provider.Chat(ctx, msgs, opts)
	p.mu.Lock()
	p.accumUsage.InputTokens += result.Usage.InputTokens
	p.accumUsage.InputCachedTokens += result.Usage.InputCachedTokens
	p.accumUsage.OutputTokens += result.Usage.OutputTokens
	p.mu.Unlock()
	return result, err
}

// ChatStream implements the ChatProvider interface and accumulates usage statistics.
func (p *UsageTrackingChatProvider) ChatStream(ctx context.Context, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Usage, error) {
	// Call the wrapped provider and accumulate usage statistics
	usage, err := p.Provider.ChatStream(ctx, msgs, opts, replies)
	p.mu.Lock()
	p.accumUsage.InputTokens += usage.InputTokens
	p.accumUsage.InputCachedTokens += usage.InputCachedTokens
	p.accumUsage.OutputTokens += usage.OutputTokens
	p.mu.Unlock()
	return usage, err
}

// GetAccumulatedUsage returns the current accumulated usage values.
func (p *UsageTrackingChatProvider) GetAccumulatedUsage() Usage {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.accumUsage
}
