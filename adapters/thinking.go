// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package adapters

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"unicode"

	"github.com/maruel/genai"
	"golang.org/x/sync/errgroup"
)

// WrapThinking wraps a ProviderGen and processes its output to extract thinking blocks ONLY if needed.
func WrapThinking(c genai.ProviderGen) genai.ProviderGen {
	if s, ok := c.(genai.ProviderScoreboard); ok {
		id := c.ModelID()
		for _, sc := range s.Scoreboard().Scenarios {
			if slices.Contains(sc.Models, id) && sc.ThinkingTokenStart != "" {
				return &ProviderGenThinking{ProviderGen: c, ThinkingTokenStart: sc.ThinkingTokenStart, ThinkingTokenEnd: sc.ThinkingTokenEnd}
			}
		}
	}
	return c
}

// ProviderGenThinking wraps a ProviderGen and processes its output to extract thinking blocks.
//
// It looks for content within tags ThinkingTokenStart and ThinkingTokenEnd and places it in Thinking Content
// blocks instead of Text.
//
// It requires the starting thinking tag. Otherwise, the content is assumed to be text. This is necessary for
// JSON formatted responses.
type ProviderGenThinking struct {
	genai.ProviderGen

	// ThinkingTokenStart is the start thinking token. It is often "<think>\n".
	ThinkingTokenStart string
	// ThinkingTokenEnd is the end thinking token, where the explicit answer lies after. It is often "\n</think>\n".
	ThinkingTokenEnd string

	_ struct{}
}

// GenSync implements the ProviderGen interface by delegating to the wrapped provider
// and processing the result to extract thinking blocks.
func (c *ProviderGenThinking) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	result, err := c.ProviderGen.GenSync(ctx, msgs, opts)
	if err2 := c.processThinkingMessage(&result.Message); err == nil {
		err = err2
	}
	return result, err
}

// GenStream implements the ProviderGen interface for streaming by delegating to the wrapped provider
// and processing each fragment to extract thinking blocks.
// If no thinking tags are present, the first part of the message is assumed to be thinking.
func (c *ProviderGenThinking) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
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

// processPacket is the streaming version of message fragment processing.
func (c *ProviderGenThinking) processPacket(state tagProcessingState, replies chan<- genai.ContentFragment, accumulated *genai.Message, f genai.ContentFragment) (tagProcessingState, error) {
	if f.ThinkingFragment != "" {
		return state, fmt.Errorf("got unexpected thinking fragment: %q; do not use ProviderGenThinking with an explicit thinking CoT model", f.ThinkingFragment)
	}
	// Mutate the fragment then send it.
	switch state {
	case start:
		// Ignore whitespace until text or thinking tag is seen.
		t := strings.TrimLeftFunc(f.TextFragment, unicode.IsSpace)
		// The tokens always have a trailing "\n". When streaming, the trailing "\n" will likely be sent as a
		// separate event. This requires a small state machine to keep track of that.
		if tStart := strings.Index(t, c.ThinkingTokenStart); tStart != -1 {
			if tStart != 0 {
				return state, fmt.Errorf("unexpected prefix before thinking tag: %q", t[:len(c.ThinkingTokenStart)+1])
			}
			f.ThinkingFragment = strings.TrimLeftFunc(t[len(c.ThinkingTokenStart):], unicode.IsSpace)
			f.TextFragment = ""
			state = thinkingTextSeen
		} else if t != "" {
			// This response does not contain thinking text, it could be JSON or something else.
			state = textSeen
		} else {
			f.TextFragment = ""
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
		if tEnd := strings.Index(f.ThinkingFragment, c.ThinkingTokenEnd); tEnd != -1 {
			state = endTagSeen
			after := f.ThinkingFragment[tEnd+len(c.ThinkingTokenEnd):]
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

// processThinkingMessage is the non-streaming version of message fragment processing. It's a bit faster since
// it can slice things directly.
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

	tStart := strings.Index(text, c.ThinkingTokenStart)
	if tStart == -1 {
		// This response does not contain thinking text, it could be JSON or something else.
		return nil
	}
	if prefix := text[:tStart]; strings.TrimSpace(prefix) != "" {
		return fmt.Errorf("unexpected prefix before thinking tag: %q", prefix)
	}
	// Zap the text.
	for i := range m.Contents {
		m.Contents[i].Text = ""
	}
	// Remove whitespace after the starting tag.
	textAfterStartTag := strings.TrimLeftFunc(text[tStart+len(c.ThinkingTokenStart):], unicode.IsSpace)
	if tEnd := strings.Index(textAfterStartTag, c.ThinkingTokenEnd); tEnd != -1 {
		thinkingContent := textAfterStartTag[:tEnd]
		remainingText := strings.TrimLeftFunc(textAfterStartTag[tEnd+len(c.ThinkingTokenEnd):], unicode.IsSpace)
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
