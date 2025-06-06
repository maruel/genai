// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package provider is awesome sauce to reduce code duplication across most providers.
package provider

import (
	"context"
	"errors"
	"fmt"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/bb"
)

//

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
	return SimulateStream(ctx, c, msgs, chunks, opts)
}

//

// ListModelsResponse is an interface for responses that contain model data.
type ListModelsResponse interface {
	// ToModels converts the provider-specific models to a slice of genai.Model
	ToModels() []genai.Model
}

// ListModels is a generic function that implements the common pattern for listing models across providers.
// It makes an HTTP GET request to the specified URL and converts the response to a slice of genai.Model.
func ListModels[PErrorResponse fmt.Stringer, R ListModelsResponse](ctx context.Context, c *Base[PErrorResponse], url string) ([]genai.Model, error) {
	var resp R
	if err := c.DoRequest(ctx, "GET", url, nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// SimulateStream simulates GenStream for APIs that do not support streaming.
func SimulateStream(ctx context.Context, c genai.ProviderGen, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	res, err := c.GenSync(ctx, msgs, opts)
	if err == nil {
		for i := range res.Contents {
			if url := res.Contents[i].URL; url != "" {
				chunks <- genai.ContentFragment{
					Filename: res.Contents[i].Filename,
					URL:      res.Contents[i].URL,
				}
			} else if d := res.Contents[i].Document; d != nil {
				chunks <- genai.ContentFragment{
					Filename:         res.Contents[i].Filename,
					DocumentFragment: res.Contents[i].Document.(*bb.BytesBuffer).D,
				}
			} else {
				return res, fmt.Errorf("expected ContentFragment with URL or Document, got %#v", res.Contents)
			}
		}
	}
	return res, err
}
