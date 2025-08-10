// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package adapters includes multiple adapters to convert one ProviderFoo interface into another one.
package adapters

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"golang.org/x/sync/errgroup"
)

// GenSyncWithToolCallLoop runs a conversation with the LLM, handling tool calls in a loop until there are no
// more tool calls.
//
// It calls the provided ProviderGen.GenSync() method, processes any tool calls using Message.DoToolCalls(),
// and continues the conversation in a loop until the LLM's response has no more tool calls.
//
// Warning: If opts.ToolCallRequest == ToolCallRequired, it will be mutated to ToolCallAny after the first
// tool call.
//
// It returns the messages to accumulate to the thread. The last message is the LLM's response.
func GenSyncWithToolCallLoop(ctx context.Context, p genai.ProviderGen, msgs genai.Messages, opts genai.Options) (genai.Messages, genai.Usage, error) {
	usage := genai.Usage{}
	var out genai.Messages
	workMsgs := make(genai.Messages, len(msgs))
	copy(workMsgs, msgs)
	chatOpts, ok := opts.(*genai.OptionsText)
	if !ok || len(chatOpts.Tools) == 0 {
		return out, usage, errors.New("no tools found")
	}
	tools := chatOpts.Tools
	for {
		result, err := p.GenSync(ctx, workMsgs, opts)
		usage.InputTokens += result.InputTokens
		usage.InputCachedTokens += result.InputCachedTokens
		usage.OutputTokens += result.OutputTokens
		usage.FinishReason = result.FinishReason
		if err != nil {
			return out, usage, err
		}
		out = append(out, result.Message)
		workMsgs = append(workMsgs, result.Message)
		if len(result.ToolCalls) == 0 {
			return out, usage, nil
		}
		tr, err := result.DoToolCalls(ctx, tools)
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
// No need to process the tool calls or accumulate the ContentFragment.
func GenStreamWithToolCallLoop(ctx context.Context, p genai.ProviderGen, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Messages, genai.Usage, error) {
	usage := genai.Usage{}
	var out genai.Messages
	workMsgs := make(genai.Messages, len(msgs))
	copy(workMsgs, msgs)
	chatOpts, ok := opts.(*genai.OptionsText)
	if !ok || len(chatOpts.Tools) == 0 {
		return out, usage, errors.New("no tools found")
	}
	tools := chatOpts.Tools
	for {
		internalReplies := make(chan genai.ContentFragment)
		reply := genai.Message{}
		eg := errgroup.Group{}
		eg.Go(func() error {
			// TODO: Process tools as they arrive for maximum asynchronicity.
			for f := range internalReplies {
				replies <- f
				if err := reply.Accumulate(f); err != nil {
					for range internalReplies {
						// Drain channel
					}
					return err
				}
			}
			return nil
		})
		result, err := p.GenStream(ctx, workMsgs, internalReplies, opts)
		close(internalReplies)
		usage.InputTokens += result.InputTokens
		usage.InputCachedTokens += result.InputCachedTokens
		usage.OutputTokens += result.OutputTokens
		usage.FinishReason = result.FinishReason
		// Note: We already have the complete message in result.Message, but we accumulate separately
		// to preserve backward compatibility with existing behavior and tests
		if err3 := eg.Wait(); err == nil {
			err = err3
		}
		if err != nil {
			return out, usage, err
		}
		out = append(out, reply)
		workMsgs = append(workMsgs, reply)
		if len(reply.ToolCalls) == 0 {
			return out, usage, nil
		}
		tr, err := reply.DoToolCalls(ctx, tools)
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

//

// ProviderGenIgnoreUnsupported wraps a ProviderGen to ignore UnsupportedContinuableError.
type ProviderGenIgnoreUnsupported struct {
	genai.ProviderGen
}

func (c *ProviderGenIgnoreUnsupported) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	res, err := c.ProviderGen.GenSync(ctx, msgs, opts)
	var uce *genai.UnsupportedContinuableError
	if errors.As(err, &uce) {
		err = nil
	}
	return res, err
}

func (c *ProviderGenIgnoreUnsupported) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	res, err := c.ProviderGen.GenStream(ctx, msgs, chunks, opts)
	var uce *genai.UnsupportedContinuableError
	if errors.As(err, &uce) {
		err = nil
	}
	return res, err
}

func (c *ProviderGenIgnoreUnsupported) Unwrap() genai.Provider {
	return c.ProviderGen
}

//

// ProviderGenDocIgnoreUnsupported wraps a ProviderGenDoc to ignore UnsupportedContinuableError.
type ProviderGenDocIgnoreUnsupported struct {
	genai.ProviderGenDoc
}

func (c *ProviderGenDocIgnoreUnsupported) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	res, err := c.ProviderGenDoc.GenDoc(ctx, msg, opts)
	var uce *genai.UnsupportedContinuableError
	if errors.As(err, &uce) {
		err = nil
	}
	return res, err
}

func (c *ProviderGenDocIgnoreUnsupported) Unwrap() genai.Provider {
	return c.ProviderGenDoc
}

//

// ProviderGenDocToGen converts a ProviderGenDoc, e.g. a provider only generating audio, images, or videos into a ProviderGen.
type ProviderGenDocToGen struct {
	genai.ProviderGenDoc
}

func (c *ProviderGenDocToGen) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	if len(msgs) != 1 {
		return genai.Result{}, errors.New("must pass exactly one Message")
	}
	return c.GenDoc(ctx, msgs[0], opts)
}

func (c *ProviderGenDocToGen) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if len(msgs) != 1 {
		return genai.Result{}, errors.New("must pass exactly one Message")
	}
	return base.SimulateStream(ctx, c, msgs, chunks, opts)
}

func (c *ProviderGenDocToGen) Unwrap() genai.Provider {
	return c.ProviderGenDoc
}

//

// ProviderGenUsage wraps a ProviderGen and accumulates Usage values
// across multiple requests to track total token consumption.
type ProviderGenUsage struct {
	genai.ProviderGen

	mu         sync.Mutex
	accumUsage genai.Usage
}

// GenSync implements the ProviderGen interface and accumulates usage statistics.
func (c *ProviderGenUsage) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	result, err := c.ProviderGen.GenSync(ctx, msgs, opts)
	c.mu.Lock()
	c.accumUsage.InputTokens += result.InputTokens
	c.accumUsage.InputCachedTokens += result.InputCachedTokens
	c.accumUsage.OutputTokens += result.OutputTokens
	c.mu.Unlock()
	return result, err
}

// GenStream implements the ProviderGen interface and accumulates usage statistics.
func (c *ProviderGenUsage) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	// Call the wrapped provider and accumulate usage statistics
	result, err := c.ProviderGen.GenStream(ctx, msgs, replies, opts)
	c.mu.Lock()
	c.accumUsage.InputTokens += result.InputTokens
	c.accumUsage.InputCachedTokens += result.InputCachedTokens
	c.accumUsage.OutputTokens += result.OutputTokens
	c.mu.Unlock()
	return result, err
}

// GetAccumulatedUsage returns the current accumulated usage values.
func (c *ProviderGenUsage) GetAccumulatedUsage() genai.Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.accumUsage
}

func (c *ProviderGenUsage) Unwrap() genai.Provider {
	return c.ProviderGen
}

//

// ProviderGenAppend wraps a ProviderGen and appends a Message before processing.
//
// Useful to inject a "/think" for Qwen3 models.
type ProviderGenAppend struct {
	genai.ProviderGen

	Append genai.Message
}

func (c *ProviderGenAppend) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	if len(msgs[len(msgs)-1].ToolCallResults) == 0 {
		msgs = append(slices.Clone(msgs), c.Append)
	}
	return c.ProviderGen.GenSync(ctx, msgs, opts)
}

func (c *ProviderGenAppend) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if len(msgs[len(msgs)-1].ToolCallResults) == 0 {
		msgs = append(slices.Clone(msgs), c.Append)
	}
	return c.ProviderGen.GenStream(ctx, msgs, replies, opts)
}

func (c *ProviderGenAppend) Unwrap() genai.Provider {
	return c.ProviderGen
}
