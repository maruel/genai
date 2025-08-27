// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package adapters includes multiple adapters to convert one ProviderFoo interface into another one.
package adapters

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"sync"

	"github.com/maruel/genai"
)

// GenSyncWithToolCallLoop runs a conversation with the LLM, handling tool calls in a loop until there are no
// more tool calls.
//
// It calls the provided Provider.GenSync() method, processes any tool calls using Message.DoToolCalls(),
// and continues the conversation in a loop until the LLM's response has no more tool calls.
//
// Warning: If opts.ToolCallRequest == ToolCallRequired, it will be mutated to ToolCallAny after the first
// tool call.
//
// It returns the messages to accumulate to the thread. The last message is the LLM's response.
func GenSyncWithToolCallLoop(ctx context.Context, p genai.Provider, msgs genai.Messages, opts ...genai.Options) (genai.Messages, genai.Usage, error) {
	usage := genai.Usage{}
	var out genai.Messages
	workMsgs := make(genai.Messages, len(msgs))
	copy(workMsgs, msgs)
	var chatOpts *genai.OptionsText
	for _, opt := range opts {
		ok := false
		if chatOpts, ok = opt.(*genai.OptionsText); ok {
			break
		}
	}
	if chatOpts == nil {
		return out, usage, errors.New("no tools found")
	}
	tools := chatOpts.Tools
	for {
		res, err := p.GenSync(ctx, workMsgs, opts...)
		usage.InputTokens += res.Usage.InputTokens
		usage.InputCachedTokens += res.Usage.InputCachedTokens
		usage.OutputTokens += res.Usage.OutputTokens
		usage.FinishReason = res.Usage.FinishReason
		usage.Limits = res.Usage.Limits
		if err != nil {
			return out, usage, err
		}
		out = append(out, res.Message)
		workMsgs = append(workMsgs, res.Message)
		if !slices.ContainsFunc(res.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
			return out, usage, nil
		}
		tr, err := res.DoToolCalls(ctx, tools)
		if err != nil {
			return out, usage, err
		}
		if tr.IsZero() {
			return out, usage, fmt.Errorf("expected tool call to return a result or an error")
		}
		out = append(out, tr)
		workMsgs = append(workMsgs, tr)
		if chatOpts.ToolCallRequest == genai.ToolCallRequired {
			chatOpts.ToolCallRequest = genai.ToolCallAny
		}
	}
}

// GenStreamWithToolCallLoop runs a conversation loop with an LLM that handles tool calls via streaming
// until there are no more. It will repeatedly call GenStream(), collect fragments into a complete message,
// process tool calls with DoToolCalls(), and continue the conversation until the LLM response
// has no more tool calls.
//
// The function will return early if any error occurs. The returned Messages will include
// all the messages including the original ones, the LLM's responses, and the tool call result
// messages.
//
// Warning: If opts.ToolCallRequest == ToolCallRequired, it will be mutated to ToolCallAny after the first
// tool call.
//
// No need to process the tool calls or accumulate the ReplyFragment.
func GenStreamWithToolCallLoop(ctx context.Context, p genai.Provider, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Messages, genai.Usage, error)) {
	var out genai.Messages
	usage := genai.Usage{}
	var finalErr error

	fnFragments := func(yield func(genai.ReplyFragment) bool) {
		workMsgs := slices.Clone(msgs)
		var chatOpts *genai.OptionsText
		for _, opt := range opts {
			ok := false
			if chatOpts, ok = opt.(*genai.OptionsText); ok {
				break
			}
		}
		if chatOpts == nil {
			finalErr = errors.New("no tools found")
			return
		}
		tools := chatOpts.Tools
		for {
			fragments, finish := p.GenStream(ctx, workMsgs, opts...)
			send := true
			for f := range fragments {
				if err := f.Validate(); err != nil {
					finalErr = err
					send = false
				}
				if send && !yield(f) {
					send = false
				}
			}
			res, err := finish()
			usage.InputTokens += res.Usage.InputTokens
			usage.InputCachedTokens += res.Usage.InputCachedTokens
			usage.OutputTokens += res.Usage.OutputTokens
			usage.FinishReason = res.Usage.FinishReason
			usage.Limits = res.Usage.Limits
			if err != nil {
				finalErr = err
				return
			}
			out = append(out, res.Message)
			workMsgs = append(workMsgs, res.Message)
			if !slices.ContainsFunc(res.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
				return
			}
			tr, err := res.DoToolCalls(ctx, tools)
			if err != nil {
				finalErr = err
				return
			}
			if tr.IsZero() {
				finalErr = fmt.Errorf("expected tool call to return a result or an error")
				return
			}
			out = append(out, tr)
			workMsgs = append(workMsgs, tr)
			if chatOpts.ToolCallRequest == genai.ToolCallRequired {
				chatOpts.ToolCallRequest = genai.ToolCallAny
			}
		}
	}
	fnFinish := func() (genai.Messages, genai.Usage, error) {
		return out, usage, finalErr
	}
	return fnFragments, fnFinish
}

//

// ProviderIgnoreUnsupported wraps a Provider to ignore UnsupportedContinuableError.
type ProviderIgnoreUnsupported struct {
	genai.Provider
}

func (c *ProviderIgnoreUnsupported) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	res, err := c.Provider.GenSync(ctx, msgs, opts...)
	var uce *genai.UnsupportedContinuableError
	if errors.As(err, &uce) {
		err = nil
	}
	return res, err
}

func (c *ProviderIgnoreUnsupported) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	fragments, finish := c.Provider.GenStream(ctx, msgs, opts...)
	return fragments, func() (genai.Result, error) {
		res, err := finish()
		var uce *genai.UnsupportedContinuableError
		if errors.As(err, &uce) {
			err = nil
		}
		return res, err
	}
}

func (c *ProviderIgnoreUnsupported) Unwrap() genai.Provider {
	return c.Provider
}

//

// ProviderUsage wraps a Provider and accumulates Usage values
// across multiple requests to track total token consumption.
type ProviderUsage struct {
	genai.Provider

	mu         sync.Mutex
	accumUsage genai.Usage
}

// GenSync implements the Provider interface and accumulates usage statistics.
func (c *ProviderUsage) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	res, err := c.Provider.GenSync(ctx, msgs, opts...)
	c.mu.Lock()
	c.accumUsage.InputTokens += res.Usage.InputTokens
	c.accumUsage.InputCachedTokens += res.Usage.InputCachedTokens
	c.accumUsage.OutputTokens += res.Usage.OutputTokens
	c.accumUsage.Limits = res.Usage.Limits
	c.mu.Unlock()
	return res, err
}

// GenStream implements the Provider interface and accumulates usage statistics.
func (c *ProviderUsage) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	// Call the wrapped provider and accumulate usage statistics
	fragments, finish := c.Provider.GenStream(ctx, msgs, opts...)
	return fragments, func() (genai.Result, error) {
		res, err := finish()
		c.mu.Lock()
		c.accumUsage.InputTokens += res.Usage.InputTokens
		c.accumUsage.InputCachedTokens += res.Usage.InputCachedTokens
		c.accumUsage.OutputTokens += res.Usage.OutputTokens
		c.accumUsage.Limits = res.Usage.Limits
		c.mu.Unlock()
		return res, err
	}
}

// GetAccumulatedUsage returns the current accumulated usage values.
func (c *ProviderUsage) GetAccumulatedUsage() genai.Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.accumUsage
}

func (c *ProviderUsage) Unwrap() genai.Provider {
	return c.Provider
}

//

// ProviderAppend wraps a Provider and appends a Request before processing when the messages end with a
// user message.
//
// Useful to inject a "/think" for Qwen3 models.
type ProviderAppend struct {
	genai.Provider

	Append genai.Request
}

func (c *ProviderAppend) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	if len(msgs[len(msgs)-1].Requests) != 0 {
		msgs = slices.Clone(msgs)
		msgs[len(msgs)-1].Requests = slices.Clone(msgs[len(msgs)-1].Requests)
		msgs[len(msgs)-1].Requests = append(msgs[len(msgs)-1].Requests, c.Append)
	}
	return c.Provider.GenSync(ctx, msgs, opts...)
}

func (c *ProviderAppend) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.ReplyFragment], func() (genai.Result, error)) {
	if i := len(msgs) - 1; len(msgs[i].Requests) != 0 {
		msgs = slices.Clone(msgs)
		msgs[i].Requests = append(slices.Clone(msgs[i].Requests), c.Append)
	}
	return c.Provider.GenStream(ctx, msgs, opts...)
}

func (c *ProviderAppend) Unwrap() genai.Provider {
	return c.Provider
}
