// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package adapter includes multiple adapters to convert one ProviderFoo interface into another one.
package adapter

import (
	"context"
	"errors"

	"github.com/maruel/genai"
	"github.com/maruel/genai/provider"
)

// GenDocToGen converts a ProviderGenDoc, e.g. a provider only generating audio, images, or videos into a ProviderGen.
type GenDocToGen struct {
	genai.ProviderGenDoc
}

func (c *GenDocToGen) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	if len(msgs) != 1 {
		return genai.Result{}, errors.New("must pass exactly one Message")
	}
	return c.GenDoc(ctx, msgs[0], opts)
}

func (c *GenDocToGen) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if len(msgs) != 1 {
		return genai.Result{}, errors.New("must pass exactly one Message")
	}
	return provider.SimulateStream(ctx, c, msgs, chunks, opts)
}
