// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package adapters

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"slices"
	"strings"
	"unicode"

	"github.com/maruel/genai"
)

// WrapReasoning wraps a Provider and processes its output to extract reasoning blocks ONLY if needed.
func WrapReasoning(c genai.Provider) genai.Provider {
	id := c.ModelID()
	for _, sc := range c.Scoreboard().Scenarios {
		// Some models like qwen-3-235b-a22b-thinking-2507 do not use ReasoningTokenStart.
		if slices.Contains(sc.Models, id) && sc.ReasoningTokenEnd != "" {
			return &ProviderReasoning{
				Provider:            c,
				ReasoningTokenStart: sc.ReasoningTokenStart,
				ReasoningTokenEnd:   sc.ReasoningTokenEnd,
			}
		}
	}
	return c
}

// ProviderReasoning wraps a Provider and processes its output to extract reasoning blocks.
//
// It looks for content within tags ReasoningTokenStart and ReasoningTokenEnd and places it in Reasoning
// Content blocks instead of Text.
//
// It requires the starting reasoning tag. Otherwise, the content is assumed to be text. This is necessary for
// JSON formatted responses.
type ProviderReasoning struct {
	genai.Provider

	// ReasoningTokenStart is the start reasoning token. It is often "<think>\n" but there are cases when it can
	// be never output, like "qwen-3-235b-a22b-thinking-2507".
	ReasoningTokenStart string
	// ReasoningTokenEnd is the end reasoning token, where the explicit answer lies after. It is often
	// "\n</think>\n".
	ReasoningTokenEnd string

	_ struct{}
}

// GenSync implements the Provider interface by delegating to the wrapped provider
// and processing the result to extract reasoning blocks.
func (c *ProviderReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	result, err := c.Provider.GenSync(ctx, msgs, opts...)
	if err2 := c.processReasoningMessage(&result.Message); err == nil {
		err = err2
	}
	return result, err
}

// GenStream implements the Provider interface for streaming by delegating to the wrapped provider
// and processing each fragment to extract reasoning blocks.
// If no reasoning tags are present, the first part of the message is assumed to be reasoning.
func (c *ProviderReasoning) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	accumulated := genai.Message{}
	var finalErr error
	fragments, finish := c.Provider.GenStream(ctx, msgs, opts...)
	fnFragments := func(yield func(genai.Reply) bool) {
		state := start
		if c.ReasoningTokenStart == "" {
			// Simulate that the reasoning tag was seen.
			state = startTagSeen
		}
		for f := range fragments {
			var replies []genai.Reply
			var err2 error
			replies, state, err2 = c.processPacket(state, &accumulated, f)
			if finalErr == nil {
				finalErr = err2
			}
			for _, r := range replies {
				if !yield(r) {
					return
				}
			}
			if finalErr != nil {
				return
			}
		}
	}
	fnFinish := func() (genai.Result, error) {
		res, err := finish()
		// Use the accumulated contents from our processed message, which has correctly
		// transformed Text's to Reasoning's according to the state machine.
		if len(accumulated.Replies) > 0 {
			res.Replies = accumulated.Replies
		}
		if finalErr != nil {
			return res, finalErr
		}
		return res, err
	}
	return fnFragments, fnFinish
}

// processPacket is the streaming version of message fragment processing.
func (c *ProviderReasoning) processPacket(state tagProcessingState, accumulated *genai.Message, f genai.Reply) ([]genai.Reply, tagProcessingState, error) {
	var replies []genai.Reply
	if f.Reasoning != "" {
		return replies, state, fmt.Errorf("got unexpected reasoning fragment: %q; do not use ProviderReasoning with an explicit reasoning CoT model", f.Reasoning)
	}
	// Mutate the fragment then send it.
	switch state {
	case start:
		// Ignore whitespace until text or reasoning tag is seen.
		t := strings.TrimLeftFunc(f.Text, unicode.IsSpace)
		// The tokens always have a trailing "\n". When streaming, the trailing "\n" will likely be sent as a
		// separate event. This requires a small state machine to keep track of that.
		if tStart := strings.Index(t, c.ReasoningTokenStart); tStart != -1 {
			if tStart != 0 {
				return replies, state, fmt.Errorf("unexpected prefix before reasoning tag: %q", t[:len(c.ReasoningTokenStart)+1])
			}
			f.Reasoning = strings.TrimLeftFunc(t[len(c.ReasoningTokenStart):], unicode.IsSpace)
			f.Text = ""
			state = thinkingTextSeen
		} else if t != "" {
			// This response does not contain reasoning text, it could be JSON or something else.
			state = textSeen
		} else {
			f.Text = ""
		}
	case startTagSeen:
		// Ignore whitespace until text is seen.
		f.Reasoning = f.Text
		f.Text = ""
		if buf := strings.TrimLeftFunc(f.Reasoning, unicode.IsSpace); buf != "" {
			state = thinkingTextSeen
			f.Reasoning = buf
		}
	case thinkingTextSeen:
		f.Reasoning = f.Text
		f.Text = ""
		if tEnd := strings.Index(f.Reasoning, c.ReasoningTokenEnd); tEnd != -1 {
			state = endTagSeen
			after := f.Reasoning[tEnd+len(c.ReasoningTokenEnd):]
			if tEnd != 0 {
				// Unlikely case where we need to flush out the remainder.
				f.Reasoning = f.Reasoning[:tEnd]
				f.Text = ""
				replies = append(replies, f)
				if err := accumulated.Accumulate(f); err != nil {
					return replies, state, err
				}
			}
			f.Text = after
			f.Reasoning = ""
			if buf := strings.TrimLeftFunc(f.Text, unicode.IsSpace); buf != "" {
				state = textSeen
				f.Text = buf
			}
		}
	case endTagSeen:
		// Ignore whitespace until text is seen.
		if buf := strings.TrimLeftFunc(f.Text, unicode.IsSpace); buf != "" {
			state = textSeen
			f.Text = buf
		}
	case textSeen:
	default:
		return replies, state, errors.New("internal error in ProviderReasoning.GenStream()")
	}
	replies = append(replies, f)
	err := accumulated.Accumulate(f)
	return replies, state, err
}

// processReasoningMessage is the non-streaming version of message fragment processing. It's a bit faster since
// it can slice things directly.
func (c *ProviderReasoning) processReasoningMessage(m *genai.Message) error {
	if len(m.Replies) == 0 {
		// It can be a function call.
		return nil
	}

	// Check if one of the contents is already a Reasoning block
	for _, c := range m.Replies {
		if c.Reasoning != "" {
			return fmt.Errorf("got unexpected reasoning content: %q; do not use ProviderReasoning with an explicit reasoning CoT model", c.Reasoning)
		}
	}

	text := m.String()
	if text == "" {
		// Maybe an image.
		return nil
	}

	tStart := 0
	if c.ReasoningTokenStart != "" {
		tStart = strings.Index(text, c.ReasoningTokenStart)
		if tStart == -1 {
			// This response does not contain reasoning text, it could be JSON or something else.
			return nil
		}
		if prefix := text[:tStart]; strings.TrimSpace(prefix) != "" {
			return fmt.Errorf("unexpected prefix before reasoning tag: %q", prefix)
		}
	} else {
		// Check if there's an end tag. Otherwise it was not reasoning at all.
		if !strings.Contains(text, c.ReasoningTokenEnd) {
			return nil
		}
	}
	// Zap the text.
	for i := range m.Replies {
		m.Replies[i].Text = ""
	}
	// Remove whitespace after the starting tag.
	textAfterStartTag := strings.TrimLeftFunc(text[tStart+len(c.ReasoningTokenStart):], unicode.IsSpace)
	if tEnd := strings.Index(textAfterStartTag, c.ReasoningTokenEnd); tEnd != -1 {
		thinkingContent := textAfterStartTag[:tEnd]
		remainingText := strings.TrimLeftFunc(textAfterStartTag[tEnd+len(c.ReasoningTokenEnd):], unicode.IsSpace)
		m.Replies[0].Reasoning = thinkingContent
		if len(m.Replies) == 1 {
			m.Replies = append(m.Replies, genai.Reply{})
		}
		m.Replies[len(m.Replies)-1].Text = remainingText
	} else {
		// This happens when MaxTokens is used or another reason which cut the stream off before the end tag is seen.
		// Consider everything reasoning.
		// We do not return an error so the user can process the data.
		m.Replies[0].Reasoning = textAfterStartTag
	}
	return nil
}

func (c *ProviderReasoning) Unwrap() genai.Provider {
	return c.Provider
}

type tagProcessingState int

const (
	start tagProcessingState = iota
	startTagSeen
	thinkingTextSeen
	endTagSeen
	textSeen
)
