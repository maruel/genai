// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral

import (
	"context"
	"errors"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"golang.org/x/sync/errgroup"
)

// MagistralThinking wraps a Mistral ProviderGen and processes its output to extract thinking blocks.
//
// It looks for tag "<think>" and then "\n\\boxed{" and "}".
//
// It requires the starting thinking tag. Otherwise, the content is assumed to be text. This is necessary for
// JSON formatted responses.
type MagistralThinking struct {
	genai.ProviderGen

	_ struct{}
}

// GenSync implements the ProviderGen interface by delegating to the wrapped provider
// and processing the result to extract thinking blocks.
func (c *MagistralThinking) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	t := adapters.ProviderGenThinking{ProviderGen: c.ProviderGen, TagName: "think"}
	result, err := t.GenSync(ctx, msgs, opts)
	if err2 := c.processBoxed(&result.Message); err == nil {
		err = err2
	}
	return result, err
}

// GenStream implements the ProviderGen interface for streaming by delegating to the wrapped provider
// and processing each fragment to extract thinking blocks.
// If no thinking tags are present, the first part of the message is assumed to be thinking.
func (c *MagistralThinking) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	t := adapters.ProviderGenThinking{ProviderGen: c.ProviderGen, TagName: "think"}
	internalReplies := make(chan genai.ContentFragment)
	accumulated := genai.Message{}
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		state := noTextSeen
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
	result, err := t.GenStream(ctx, msgs, internalReplies, opts)
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
func (c *MagistralThinking) processPacket(state tagProcessingState, replies chan<- genai.ContentFragment, accumulated *genai.Message, f genai.ContentFragment) (tagProcessingState, error) {
	// Mutate the fragment then send it.
	switch state {
	case noTextSeen:
		if f.TextFragment == "" {
			return state, nil
		}
		if strings.HasPrefix(f.TextFragment, "@boxed{") {
			f.TextFragment = f.TextFragment[len("@boxed{"):]
			state = boxedSeen
		}
	case boxedSeen:
	default:
		return state, errors.New("internal error in ProviderGenThinking.GenStream()")
	}
	replies <- f
	err := accumulated.Accumulate(f)
	return state, err
}

// processBoxed removes @boxed{} if found.
func (c *MagistralThinking) processBoxed(m *genai.Message) error {
	// Assumes there's only one text content.
	for _, c := range m.Contents {
		if c.Text != "" {
			if strings.HasPrefix(c.Text, "@boxed{") && strings.HasSuffix(c.Text, "}") {
				c.Text = c.Text[len("@boxed{") : len(c.Text)-len("}")]
			}
		}
	}
	return nil
}

func (c *MagistralThinking) Unwrap() genai.Provider {
	return c.ProviderGen
}

type tagProcessingState int

const (
	noTextSeen tagProcessingState = iota
	boxedSeen
)
