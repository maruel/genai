// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"context"
	"fmt"
	"strings"
	"unicode"

	"golang.org/x/sync/errgroup"
)

// ThinkingChatProvider wraps a ChatProvider and processes its output to extract thinking blocks.
//
// It looks for content within tags ("<TagName>" and "</TagName>") and places it in Thinking Content blocks
// instead of Text.
type ThinkingChatProvider struct {
	// Provider is the underlying ChatProvider.
	Provider ChatProvider

	// TagName is the name of the tag to use for thinking content. Normally "think" or "thinking".
	TagName string

	_ struct{}
}

// Chat implements the ChatProvider interface by delegating to the wrapped provider
// and processing the result to extract thinking blocks.
func (tp *ThinkingChatProvider) Chat(ctx context.Context, msgs Messages, opts Validatable) (ChatResult, error) {
	result, err := tp.Provider.Chat(ctx, msgs, opts)
	if err != nil {
		return result, err
	}
	if len(result.Contents) != 1 {
		// This is extremely unlikely this will happen.
		return result, fmt.Errorf("implement when there's no or multiple content blocks")
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
func (tp *ThinkingChatProvider) ChatStream(ctx context.Context, msgs Messages, opts Validatable, replies chan<- MessageFragment) (Usage, error) {
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
