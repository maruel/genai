// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"unicode"

	"golang.org/x/sync/errgroup"
)

// ChatWithToolCallLoop runs a conversation with the LLM, handling tool calls in a loop until there are no
// more tool calls.
//
// It calls the provided ProviderChat.Chat() method, processes any tool calls using Message.DoToolCalls(),
// and continues the conversation in a loop until the LLM's response has no more tool calls.
//
// Warning: If opts.ToolCallRequest == ToolCallRequired, it will be mutated to ToolCallAny after the first
// tool call.
//
// It returns the messages to accumulate to the thread. The last message is the LLM's response.
func ChatWithToolCallLoop(ctx context.Context, provider ProviderChat, msgs Messages, opts Validatable) (Messages, Usage, error) {
	usage := Usage{}
	var out Messages
	workMsgs := make(Messages, len(msgs))
	copy(workMsgs, msgs)
	chatOpts, ok := opts.(*ChatOptions)
	if !ok || len(chatOpts.Tools) == 0 {
		return out, usage, errors.New("no tools found")
	}
	tools := chatOpts.Tools
	for {
		result, err := provider.Chat(ctx, workMsgs, opts)
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
		if chatOpts.ToolCallRequest == ToolCallRequired {
			chatOpts.ToolCallRequest = ToolCallAny
		}
	}
}

// ChatStreamWithToolCallLoop runs a conversation loop with an LLM that handles tool calls via streaming
// until there are no more. It will repeatedly call ChatStream(), collect fragments into a complete message,
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
// No need to process the tool calls or accumulate the MessageFragment.
func ChatStreamWithToolCallLoop(ctx context.Context, provider ProviderChat, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Messages, Usage, error) {
	usage := Usage{}
	var out Messages
	workMsgs := make(Messages, len(msgs))
	copy(workMsgs, msgs)
	chatOpts, ok := opts.(*ChatOptions)
	if !ok || len(chatOpts.Tools) == 0 {
		return out, usage, errors.New("no tools found")
	}
	tools := chatOpts.Tools
	for {
		internalReplies := make(chan MessageFragment)
		reply := Message{}
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
		result, err := provider.ChatStream(ctx, workMsgs, opts, internalReplies)
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
		if chatOpts.ToolCallRequest == ToolCallRequired {
			chatOpts.ToolCallRequest = ToolCallAny
		}
	}
}

//

// ProviderChatUsage wraps a ProviderChat and accumulates Usage values
// across multiple requests to track total token consumption.
type ProviderChatUsage struct {
	ProviderChat

	mu         sync.Mutex
	accumUsage Usage
}

// Chat implements the ProviderChat interface and accumulates usage statistics.
func (c *ProviderChatUsage) Chat(ctx context.Context, msgs Messages, opts Validatable) (Result, error) {
	result, err := c.ProviderChat.Chat(ctx, msgs, opts)
	c.mu.Lock()
	c.accumUsage.InputTokens += result.InputTokens
	c.accumUsage.InputCachedTokens += result.InputCachedTokens
	c.accumUsage.OutputTokens += result.OutputTokens
	c.mu.Unlock()
	return result, err
}

// ChatStream implements the ProviderChat interface and accumulates usage statistics.
func (c *ProviderChatUsage) ChatStream(ctx context.Context, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Result, error) {
	// Call the wrapped provider and accumulate usage statistics
	result, err := c.ProviderChat.ChatStream(ctx, msgs, opts, replies)
	c.mu.Lock()
	c.accumUsage.InputTokens += result.InputTokens
	c.accumUsage.InputCachedTokens += result.InputCachedTokens
	c.accumUsage.OutputTokens += result.OutputTokens
	c.mu.Unlock()
	return result, err
}

// GetAccumulatedUsage returns the current accumulated usage values.
func (c *ProviderChatUsage) GetAccumulatedUsage() Usage {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.accumUsage
}

func (c *ProviderChatUsage) Unwrap() Provider {
	return c.ProviderChat
}

//

// ProviderChatThinking wraps a ProviderChat and processes its output to extract thinking blocks.
//
// It looks for content within tags ("<TagName>" and "</TagName>") and places it in Thinking Content blocks
// instead of Text.
type ProviderChatThinking struct {
	ProviderChat

	// TagName is the name of the tag to use for thinking content. Normally "think" or "thinking".
	TagName string

	// SkipJSON specifies to skip parsing when JSON is requested.
	SkipJSON bool

	_ struct{}
}

// Chat implements the ProviderChat interface by delegating to the wrapped provider
// and processing the result to extract thinking blocks.
func (c *ProviderChatThinking) Chat(ctx context.Context, msgs Messages, opts Validatable) (Result, error) {
	result, err := c.ProviderChat.Chat(ctx, msgs, opts)
	// When replying in JSON, the thinking tokens are "denied" by the engine.
	if o, ok := opts.(*ChatOptions); !ok || !c.SkipJSON || (!o.ReplyAsJSON && o.DecodeAs == nil) {
		if err2 := c.processThinkingMessage(&result.Message); err == nil {
			err = err2
		}
	}
	return result, err
}

// ChatStream implements the ProviderChat interface for streaming by delegating to the wrapped provider
// and processing each fragment to extract thinking blocks.
// If no thinking tags are present, the first part of the message is assumed to be thinking.
func (c *ProviderChatThinking) ChatStream(ctx context.Context, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Result, error) {
	if c.SkipJSON {
		if o, ok := opts.(*ChatOptions); ok && (o.ReplyAsJSON || o.DecodeAs != nil) {
			// When replying in JSON, the thinking tokens are "denied" by the engine.
			return c.ProviderChat.ChatStream(ctx, msgs, opts, replies)
		}
	}

	internalReplies := make(chan MessageFragment)
	// The tokens always have a trailing "\n". When streaming, the trailing "\n" will likely be sent as a
	// separate event. This requires a small state machine to keep track of that.
	tagStart := "<" + c.TagName + ">"
	tagEnd := "</" + c.TagName + ">"

	accumulated := Message{}
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		state := start
		for f := range internalReplies {
			if f.ThinkingFragment != "" {
				return fmt.Errorf("got unexpected thinking fragment: %q; do not use ProviderChatThinking with an explicit thinking CoT model", f.ThinkingFragment)
			}
			// Mutate the fragment then send it.
			switch state {
			case start:
				// Ignore whitespace until text or thinking tag is seen.
				f.ThinkingFragment = strings.TrimLeftFunc(f.TextFragment, unicode.IsSpace)
				f.TextFragment = ""
				if tStart := strings.Index(f.ThinkingFragment, tagStart); tStart != -1 {
					if tStart != 0 {
						return fmt.Errorf("unexpected prefix before thinking tag: %q", f.TextFragment)
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
				if tEnd := strings.Index(f.ThinkingFragment, tagEnd); tEnd != -1 {
					state = endTagSeen
					after := f.ThinkingFragment[tEnd+len(tagEnd):]
					if tEnd != 0 {
						// Unlikely case where we need to flush out the remainder.
						f.ThinkingFragment = f.ThinkingFragment[:tEnd]
						f.TextFragment = ""
						replies <- f
						if err := accumulated.Accumulate(f); err != nil {
							return err
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
				panic("internal error")
			}
			replies <- f
			if err := accumulated.Accumulate(f); err != nil {
				return err
			}
		}
		return nil
	})
	result, err := c.ProviderChat.ChatStream(ctx, msgs, opts, internalReplies)
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

func (c *ProviderChatThinking) processThinkingMessage(m *Message) error {
	if len(m.Contents) == 0 {
		// It can be a function call.
		return nil
	}

	// Check if one of the contents is already a Thinking block
	for _, c := range m.Contents {
		if c.Thinking != "" {
			return fmt.Errorf("got unexpected thinking content: %q; do not use ProviderChatThinking with an explicit thinking CoT model", c.Thinking)
		}
	}
	if len(m.Contents) > 1 {
		return fmt.Errorf("got multiple content blocks; to be implemented; %#v", m)
	}

	text := m.AsText()
	if text == "" {
		// Maybe an image.
		return nil
	}

	tagStart := "<" + c.TagName + ">"
	tagEnd := "</" + c.TagName + ">"
	tStart := strings.Index(text, tagStart)
	if tStart != -1 {
		if prefix := text[:tStart]; strings.TrimSpace(prefix) != "" {
			return fmt.Errorf("failed to parse thinking tokens")
		}
		// Remove whitespace after the starting tag.
		text = strings.TrimLeftFunc(text[tStart+len(tagStart):], unicode.IsSpace)
	}
	if tEnd := strings.Index(text, tagEnd); tEnd != -1 {
		// Remove whitespace after the ending tag.
		m.Contents[0].Text = strings.TrimLeftFunc(text[tEnd+len(tagEnd):], unicode.IsSpace)
		m.Contents = append([]Content{{Thinking: text[:tEnd]}}, m.Contents...)
	} else if tStart != -1 {
		// This happens when MaxTokens is used or another reason which cut the stream off before the end tag is seen.
		// Consider everything thinking.
		m.Contents[0].Thinking = text
		m.Contents[0].Text = ""
	}
	return nil
}

func (c *ProviderChatThinking) Unwrap() Provider {
	return c.ProviderChat
}

type tagProcessingState int

const (
	start tagProcessingState = iota
	startTagSeen
	thinkingTextSeen
	endTagSeen
	textSeen
)
