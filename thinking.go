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
		// gotText is set once we got any form of text. This is to discard leading whitespace, which sometimes
		// happens before the initial thinking tag.
		gotText := false
		// gotTag is true when a tag is found until non-whitespace is found.
		gotTag := false
		foundEndTag := false
		sentText := false
		for f := range internalReplies {
			buf := f.TextFragment
			if buf != "" && !foundEndTag {
				if !gotText || gotTag {
					buf = strings.TrimLeftFunc(buf, unicode.IsSpace)
				}
				if strings.TrimSpace(buf) != "" {
					gotText = false
					gotTag = false
				}
				if tStart := strings.Index(buf, tagStart); tStart != -1 {
					gotTag = true
					// Was there an empty prefix?
					if strings.TrimSpace(buf[:tStart]) != "" || sentText {
						// Empty the channel.
						for range internalReplies {
						}
						return fmt.Errorf("ignoring prefix before thinking tag: %q", buf[:tStart])
					}
					// Process thinking after the tag, if any. Generally there's none.
					buf = strings.TrimLeftFunc(buf[tStart+len(tagStart):], unicode.IsSpace)
				}
				if tEnd := strings.Index(buf, tagEnd); tEnd != -1 {
					gotTag = true
					replies <- MessageFragment{ThinkingFragment: buf[:tEnd]}
					foundEndTag = true
					// Process text after the tag, if any. Generally there's none.
					f.TextFragment = strings.TrimLeftFunc(buf[tEnd+len(tagEnd):], unicode.IsSpace)
				}
			}
			if !foundEndTag {
				f.TextFragment = ""
				f.ThinkingFragment = buf
				if !sentText && strings.TrimSpace(buf) != "" {
					sentText = true
				}
			}
			if !f.IsZero() {
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
