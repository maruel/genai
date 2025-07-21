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
	"strings"
	"sync"
	"unicode"

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

// ProviderGenThinking wraps a ProviderGen and processes its output to extract thinking blocks.
//
// It looks for content within tags ("<TagName>" and "</TagName>") and places it in Thinking Content blocks
// instead of Text.
type ProviderGenThinking struct {
	genai.ProviderGen

	// TagName is the name of the tag to use for thinking content. Normally "think" or "thinking".
	TagName string

	// SkipJSON specifies to skip parsing when JSON is requested.
	SkipJSON bool

	_ struct{}
}

// GenSync implements the ProviderGen interface by delegating to the wrapped provider
// and processing the result to extract thinking blocks.
func (c *ProviderGenThinking) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	result, err := c.ProviderGen.GenSync(ctx, msgs, opts)
	// When replying in JSON, the thinking tokens are "denied" by the engine.
	if o, ok := opts.(*genai.OptionsText); !ok || !c.SkipJSON || (!o.ReplyAsJSON && o.DecodeAs == nil) {
		if err2 := c.processThinkingMessage(&result.Message); err == nil {
			err = err2
		}
	}
	return result, err
}

// GenStream implements the ProviderGen interface for streaming by delegating to the wrapped provider
// and processing each fragment to extract thinking blocks.
// If no thinking tags are present, the first part of the message is assumed to be thinking.
func (c *ProviderGenThinking) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if c.SkipJSON {
		if o, ok := opts.(*genai.OptionsText); ok && (o.ReplyAsJSON || o.DecodeAs != nil) {
			// When replying in JSON, the thinking tokens are "denied" by the engine.
			return c.ProviderGen.GenStream(ctx, msgs, replies, opts)
		}
	}

	internalReplies := make(chan genai.ContentFragment)
	accumulated := genai.Message{}
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		state := start
		for f := range internalReplies {
			var err2 error
			state, err2 = c.processPacket(state, replies, &accumulated, f)
			if err2 != nil {
				for range internalReplies {
					// Drain channel
				}
				return err2
			}
		}
		return nil
	})
	result, err := c.ProviderGen.GenStream(ctx, msgs, internalReplies, opts)
	close(internalReplies)
	if err3 := eg.Wait(); err == nil {
		err = err3
	}
	// Use the accumulated contents from our processed message, which has correctly
	// transformed TextFragments to ThinkingFragments according to the state machine
	if len(accumulated.Contents) > 0 {
		result.Contents = accumulated.Contents
	}
	if err != nil {
		return result, err
	}
	return result, nil
}

func (c *ProviderGenThinking) processPacket(state tagProcessingState, replies chan<- genai.ContentFragment, accumulated *genai.Message, f genai.ContentFragment) (tagProcessingState, error) {
	if f.ThinkingFragment != "" {
		return state, fmt.Errorf("got unexpected thinking fragment: %q; do not use ProviderGenThinking with an explicit thinking CoT model", f.ThinkingFragment)
	}
	// Mutate the fragment then send it.
	switch state {
	case start:
		// Ignore whitespace until text or thinking tag is seen.
		f.ThinkingFragment = strings.TrimLeftFunc(f.TextFragment, unicode.IsSpace)
		f.TextFragment = ""
		// The tokens always have a trailing "\n". When streaming, the trailing "\n" will likely be sent as a
		// separate event. This requires a small state machine to keep track of that.
		tagStart := "<" + c.TagName + ">"
		if tStart := strings.Index(f.ThinkingFragment, tagStart); tStart != -1 {
			if tStart != 0 {
				return state, fmt.Errorf("unexpected prefix before thinking tag: %q", f.TextFragment)
			}
			f.ThinkingFragment = strings.TrimLeftFunc(f.ThinkingFragment[len(tagStart):], unicode.IsSpace)
		}
		if f.ThinkingFragment != "" {
			// Some model do not always send tagStart.
			state = thinkingTextSeen
		}
	case startTagSeen:
		// Ignore whitespace until text is seen.
		f.ThinkingFragment = f.TextFragment
		f.TextFragment = ""
		if buf := strings.TrimLeftFunc(f.ThinkingFragment, unicode.IsSpace); buf != "" {
			state = thinkingTextSeen
			f.ThinkingFragment = buf
		}
	case thinkingTextSeen:
		f.ThinkingFragment = f.TextFragment
		f.TextFragment = ""
		tagEnd := "</" + c.TagName + ">"
		if tEnd := strings.Index(f.ThinkingFragment, tagEnd); tEnd != -1 {
			state = endTagSeen
			after := f.ThinkingFragment[tEnd+len(tagEnd):]
			if tEnd != 0 {
				// Unlikely case where we need to flush out the remainder.
				f.ThinkingFragment = f.ThinkingFragment[:tEnd]
				f.TextFragment = ""
				replies <- f
				if err := accumulated.Accumulate(f); err != nil {
					return state, err
				}
			}
			f.TextFragment = after
			f.ThinkingFragment = ""
			if buf := strings.TrimLeftFunc(f.TextFragment, unicode.IsSpace); buf != "" {
				state = textSeen
				f.TextFragment = buf
			}
		}
	case endTagSeen:
		// Ignore whitespace until text is seen.
		if buf := strings.TrimLeftFunc(f.TextFragment, unicode.IsSpace); buf != "" {
			state = textSeen
			f.TextFragment = buf
		}
	case textSeen:
	default:
		return state, errors.New("internal error in ProviderGenThinking.GenStream()")
	}
	replies <- f
	err := accumulated.Accumulate(f)
	return state, err
}

func (c *ProviderGenThinking) processThinkingMessage(m *genai.Message) error {
	if len(m.Contents) == 0 {
		// It can be a function call.
		return nil
	}

	// Check if one of the contents is already a Thinking block
	for _, c := range m.Contents {
		if c.Thinking != "" {
			return fmt.Errorf("got unexpected thinking content: %q; do not use ProviderGenThinking with an explicit thinking CoT model", c.Thinking)
		}
	}

	text := m.AsText()
	if text == "" {
		// Maybe an image.
		return nil
	}

	tagStart := "<" + c.TagName + ">"
	tagEnd := "</" + c.TagName + ">"
	tStart := strings.Index(text, tagStart)
	if tStart == -1 {
		return nil // No thinking tag found, nothing to do
	}
	if prefix := text[:tStart]; strings.TrimSpace(prefix) != "" {
		return fmt.Errorf("failed to parse thinking tokens")
	}
	// Zap the text.
	for i := range m.Contents {
		m.Contents[i].Text = ""
	}
	// Remove whitespace after the starting tag.
	textAfterStartTag := strings.TrimLeftFunc(text[tStart+len(tagStart):], unicode.IsSpace)
	if tEnd := strings.Index(textAfterStartTag, tagEnd); tEnd != -1 {
		thinkingContent := textAfterStartTag[:tEnd]
		remainingText := strings.TrimLeftFunc(textAfterStartTag[tEnd+len(tagEnd):], unicode.IsSpace)
		m.Contents[0].Thinking = thinkingContent
		if len(m.Contents) == 1 {
			m.Contents = append(m.Contents, genai.Content{})
		}
		m.Contents[len(m.Contents)-1].Text = remainingText
	} else {
		// This happens when MaxTokens is used or another reason which cut the stream off before the end tag is seen.
		// Consider everything thinking.
		// We do not return an error so the user can process the data.
		m.Contents[0].Thinking = textAfterStartTag
	}
	return nil
}

func (c *ProviderGenThinking) Unwrap() genai.Provider {
	return c.ProviderGen
}

type tagProcessingState int

const (
	start tagProcessingState = iota
	startTagSeen
	thinkingTextSeen
	endTagSeen
	textSeen
)
