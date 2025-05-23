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
// It calls the provided ChatProvider.Chat() method, processes any tool calls using Message.DoToolCalls(),
// and continues the conversation in a loop until the LLM's response has no more tool calls.
//
// Warning: If opts.ToolCallRequest == ToolCallRequired, it will be mutated to ToolCallAny after the first
// tool call.
//
// It returns the messages to accumulate to the thread. The last message is the LLM's response.
func ChatWithToolCallLoop(ctx context.Context, provider ChatProvider, msgs Messages, opts Validatable) (Messages, Usage, error) {
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
		tr, err := result.DoToolCalls(tools)
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
func ChatStreamWithToolCallLoop(ctx context.Context, provider ChatProvider, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Messages, Usage, error) {
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
		var loopMsgs Messages
		eg := errgroup.Group{}
		eg.Go(func() error {
			for f := range internalReplies {
				replies <- f
				var err2 error
				if loopMsgs, err2 = f.Accumulate(loopMsgs); err2 != nil {
					for range internalReplies {
					}
					return err2
				}
			}
			return nil
		})
		u, err := provider.ChatStream(ctx, workMsgs, opts, internalReplies)
		usage.InputTokens += u.InputTokens
		usage.InputCachedTokens += u.InputCachedTokens
		usage.OutputTokens += u.OutputTokens
		usage.FinishReason = u.FinishReason
		if err3 := eg.Wait(); err == nil {
			err = err3
		}
		if err != nil {
			return out, usage, err
		}
		out = append(out, loopMsgs...)
		workMsgs = append(workMsgs, loopMsgs...)
		hadTool := false
		for i := range loopMsgs {
			if len(loopMsgs[i].ToolCalls) == 0 {
				continue
			}
			hadTool = true
			tr, err := loopMsgs[i].DoToolCalls(tools)
			if err != nil {
				return out, usage, err
			}
			if tr.IsZero() {
				return out, usage, fmt.Errorf("expected tool call to return a result or an error")
			}
			out = append(out, tr)
			workMsgs = append(workMsgs, tr)
		}
		if !hadTool {
			return out, usage, nil
		}
		if chatOpts.ToolCallRequest == ToolCallRequired {
			chatOpts.ToolCallRequest = ToolCallAny
		}
	}
}

//

// ChatProviderUsage wraps a ChatProvider and accumulates Usage values
// across multiple requests to track total token consumption.
type ChatProviderUsage struct {
	Provider ChatProvider

	mu         sync.Mutex
	accumUsage Usage
}

// Chat implements the ChatProvider interface and accumulates usage statistics.
func (p *ChatProviderUsage) Chat(ctx context.Context, msgs Messages, opts Validatable) (ChatResult, error) {
	result, err := p.Provider.Chat(ctx, msgs, opts)
	p.mu.Lock()
	p.accumUsage.InputTokens += result.InputTokens
	p.accumUsage.InputCachedTokens += result.InputCachedTokens
	p.accumUsage.OutputTokens += result.OutputTokens
	p.mu.Unlock()
	return result, err
}

// ChatStream implements the ChatProvider interface and accumulates usage statistics.
func (p *ChatProviderUsage) ChatStream(ctx context.Context, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Usage, error) {
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
func (p *ChatProviderUsage) GetAccumulatedUsage() Usage {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.accumUsage
}

//

// ChatProviderThinking wraps a ChatProvider and processes its output to extract thinking blocks.
//
// It looks for content within tags ("<TagName>" and "</TagName>") and places it in Thinking Content blocks
// instead of Text.
type ChatProviderThinking struct {
	// Provider is the underlying ChatProvider.
	Provider ChatProvider

	// TagName is the name of the tag to use for thinking content. Normally "think" or "thinking".
	TagName string

	_ struct{}
}

// Chat implements the ChatProvider interface by delegating to the wrapped provider
// and processing the result to extract thinking blocks.
func (tp *ChatProviderThinking) Chat(ctx context.Context, msgs Messages, opts Validatable) (ChatResult, error) {
	result, err := tp.Provider.Chat(ctx, msgs, opts)
	if err != nil {
		return result, err
	}
	if len(result.Contents) == 0 {
		// It can be a function call.
		return result, nil
	}
	if len(result.Contents) > 1 {
		return result, fmt.Errorf("multiple content block: %#v", result.Contents)
	}
	text := result.Contents[0].Text
	if text == "" {
		// Maybe an image.
		return result, nil
	}

	tagStart := "<" + tp.TagName + ">"
	tagEnd := "</" + tp.TagName + ">"
	if tStart := strings.Index(text, tagStart); tStart != -1 {
		if prefix := text[:tStart]; strings.TrimSpace(prefix) != "" {
			return result, fmt.Errorf("failed to parse thinking tokens")
		}
		// Remove whitespace after the starting tag.
		text = strings.TrimLeftFunc(text[tStart+len(tagStart):], unicode.IsSpace)
	}
	if tEnd := strings.Index(text, tagEnd); tEnd != -1 {
		// Remove whitespace after the ending tag.
		result.Contents[0].Text = strings.TrimLeftFunc(text[tEnd+len(tagEnd):], unicode.IsSpace)
		result.Contents = append([]Content{{Thinking: text[:tEnd]}}, result.Contents...)
	}
	return result, nil
}

// ChatStream implements the ChatProvider interface for streaming by delegating to the wrapped provider
// and processing each fragment to extract thinking blocks.
// If no thinking tags are present, the first part of the message is assumed to be thinking.
func (tp *ChatProviderThinking) ChatStream(ctx context.Context, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Usage, error) {
	internalReplies := make(chan MessageFragment)
	// The tokens always have a trailing "\n". When streaming, the trailing "\n" will likely be sent as a
	// separate event. This requires a small state machine to keep track of that.
	tagStart := "<" + tp.TagName + ">"
	tagEnd := "</" + tp.TagName + ">"

	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		state := start
		for f := range internalReplies {
			switch state {
			case start:
				// Ignore whitespace until text or thinking tag is seen.
				buf := strings.TrimLeftFunc(f.TextFragment, unicode.IsSpace)
				if tStart := strings.Index(buf, tagStart); tStart != -1 {
					if tStart != 0 {
						return fmt.Errorf("unexpected prefix before thinking tag: %q", f.TextFragment)
					}
					state = startTagSeen
					if buf = strings.TrimLeftFunc(buf[len(tagStart):], unicode.IsSpace); buf != "" {
						state = thinkingTextSeen
						f.ThinkingFragment = buf
						f.TextFragment = ""
						replies <- f
					}
					continue
				}
				if buf != "" {
					// Some model do not always send tagStart.
					state = thinkingTextSeen
					f.ThinkingFragment = buf
					f.TextFragment = ""
					replies <- f
				}
			case startTagSeen:
				// Ignore whitespace until text is seen.
				if buf := strings.TrimLeftFunc(f.TextFragment, unicode.IsSpace); buf != "" {
					state = thinkingTextSeen
					f.ThinkingFragment = buf
					f.TextFragment = ""
					replies <- f
				}
			case thinkingTextSeen:
				if tEnd := strings.Index(f.TextFragment, tagEnd); tEnd != -1 {
					state = endTagSeen
					buf := f.TextFragment
					if tEnd != 0 {
						f.ThinkingFragment = buf[:tEnd]
						f.TextFragment = ""
						replies <- f
					}
					if buf = strings.TrimLeftFunc(buf[tEnd+len(tagEnd):], unicode.IsSpace); buf != "" {
						state = textSeen
						f.TextFragment = buf
						replies <- f
					}
					continue
				}
				f.ThinkingFragment = f.TextFragment
				f.TextFragment = ""
				replies <- f
			case endTagSeen:
				// Ignore whitespace until text is seen.
				if buf := strings.TrimLeftFunc(f.TextFragment, unicode.IsSpace); buf != "" {
					state = textSeen
					f.TextFragment = buf
					replies <- f
				}
			case textSeen:
				replies <- f
			}
		}
		return nil
	})
	usage, err := tp.Provider.ChatStream(ctx, msgs, opts, internalReplies)
	close(internalReplies)
	if err3 := eg.Wait(); err == nil {
		err = err3
	}
	return usage, err
}

type tagProcessingState int

const (
	start tagProcessingState = iota
	startTagSeen
	thinkingTextSeen
	endTagSeen
	textSeen
)
