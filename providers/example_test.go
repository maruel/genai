// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package providers_test

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"maps"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers"
)

func Example_all_ListModel() {
	ctx := context.Background()
	for name, cfg := range providers.All {
		c, err := cfg.Factory(ctx, genai.ProviderOptionModel(genai.ModelNone))
		if err != nil {
			continue
		}
		models, err := c.ListModels(context.Background())
		var ent *base.ErrNotSupported
		if errors.As(err, &ent) {
			continue
		}
		fmt.Printf("%s:\n", name)
		if err != nil {
			fmt.Printf("  Failed to get models: %v\n", err)
		}
		for _, model := range models {
			fmt.Printf("- %s\n", model)
		}
	}
}

func Example_all_Provider() {
	ctx := context.Background()
	for name, cfg := range providers.All {
		c, err := cfg.Factory(ctx, genai.ProviderOptionModel(genai.ModelCheap))
		if err != nil {
			log.Fatal(err)
		}
		msgs := genai.Messages{
			genai.NewTextMessage("Tell a story in 10 words."),
		}
		// Include options with some unsupported features to demonstrate UnsupportedContinuableError
		opts := &genai.GenOptionsText{
			TopK:      50, // Not all providers support this
			MaxTokens: 512,
		}
		response, err := c.GenSync(context.Background(), msgs, opts)
		if err != nil {
			fmt.Printf("- %s: %v\n", name, err)
		} else {
			fmt.Printf("- %s: %v\n", name, response)
		}
	}
}

func Example_all_Full() {
	// This example includes:
	// - Processing <think> tokens for explicit Chain-of-Thoughts models (e.g. Qwen3).
	var names []string
	ctx := context.Background()
	for name := range providers.Available(ctx) {
		names = append(names, name)
	}
	sort.Strings(names)
	s := strings.Join(names, ", ")
	if s == "" {
		s = "set environment variables, e.g. `OPENAI_API_KEY`"
	}
	provider := flag.String("provider", "", "provider to use, "+s)
	model := flag.String("model", "", "model to use; "+genai.ModelCheap+", "+genai.ModelGood+" (default) or "+genai.ModelSOTA+" for automatic model selection")
	remote := flag.String("remote", "", "url to use, e.g. when using ollama or llama-server on another host")
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		log.Fatal("provide a query")
	}
	var opts []genai.ProviderOption
	if *model != "" {
		opts = append(opts, genai.ProviderOptionModel(*model))
	}
	if *remote != "" {
		opts = append(opts, genai.ProviderOptionRemote(*remote))
	}
	p, err := LoadProvider(ctx, *provider, opts...)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := p.GenSync(ctx, genai.Messages{genai.NewTextMessage(query)})
	if err != nil {
		log.Fatalf("failed to use provider %q: %s", *provider, err)
	}
	fmt.Printf("%s\n", resp.String())
}

// LoadProvider loads a provider.
func LoadProvider(ctx context.Context, provider string, opts ...genai.ProviderOption) (genai.Provider, error) {
	if provider == "" {
		return nil, fmt.Errorf("no provider specified")
	}
	cfg := providers.All[provider]
	if cfg.Factory == nil {
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
	c, err := cfg.Factory(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to provider %q: %w", provider, err)
	}
	// Wrap the provider with an adapter to process "<think>" tokens automatically ONLY if needed.
	return adapters.WrapReasoning(c), nil
}

// LoadDefaultProvider loads a provider if there's exactly one available.
func LoadDefaultProvider(ctx context.Context) (genai.Provider, error) {
	avail := providers.Available(ctx)
	if len(avail) == 1 {
		for _, cfg := range avail {
			return cfg.Factory(ctx, genai.ProviderOptionModel(genai.ModelNone))
		}
	}
	if len(avail) == 0 {
		return nil, errors.New("no provider available; please set environment variables or specify a provider and API keys or remote URL")
	}
	return nil, fmt.Errorf("multiple providers available, select one of: %s", strings.Join(slices.Sorted(maps.Keys(avail)), ", "))
}

func Example_available() {
	// Automatically select the provider available if there's only one. Asserts that the provider implements
	// Provider.
	ctx := context.Background()
	c, err := LoadDefaultProvider(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
	}
	msgs := genai.Messages{genai.NewTextMessage("Provide a life tip that sounds good but is actually a bad idea.")}
	opts := genai.GenOptionsText{Seed: 42}
	resp, err := c.GenSync(ctx, msgs, &opts)
	if err != nil {
		var ent *base.ErrNotSupported
		if errors.As(err, &ent) && slices.Contains(ent.Options, "GenOptionsText.Seed") {
			opts.Seed = 0
			if resp, err = c.GenSync(ctx, msgs, &opts); err != nil {
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
				return
			}
		}
	}
	fmt.Printf("%s\n", resp.String())
}
