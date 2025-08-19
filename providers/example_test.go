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
	"os"
	"sort"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers"
)

func Example_all_ProvidersModel() {
	for _, name := range GetProvidersModel() {
		c, err := providers.All[name](&genai.OptionsProvider{Model: genai.ModelNone}, nil)
		if err != nil {
			log.Fatal(err)
		}
		// c is guaranteed to implement ProviderModel.
		models, err := c.(genai.ProviderModel).ListModels(context.Background())
		fmt.Printf("%s:\n", name)
		if err != nil {
			fmt.Printf("  Failed to get models: %v\n", err)
		}
		for _, model := range models {
			fmt.Printf("- %s\n", model)
		}
	}
}

// GetProvidersModel returns all the providers that support listing models.
//
// It's really just a simple loop that iterates over each item in All and checks if it implements
// genai.ProviderModel.
//
// Test:
//   - `c` if you want to determine if the functionality is potentially available, even if there's no known
//     API key available at the moment.
//   - `err` if you want to determine if the functionality is available in the current context, i.e.
//     environment variables FOO_API_KEY are set.
func GetProvidersModel() []string {
	var names []string
	for name, f := range providers.All {
		c, _ := f(&genai.OptionsProvider{Model: genai.ModelNone}, nil)
		if c == nil {
			continue
		}
		if _, ok := c.(genai.ProviderModel); ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

func Example_all_ProviderGen() {
	for name, f := range providers.All {
		c, err := f(&genai.OptionsProvider{Model: genai.ModelCheap}, nil)
		if err != nil {
			log.Fatal(err)
		}
		p, ok := c.(genai.ProviderGen)
		if !ok {
			if pd, ok := c.(genai.ProviderGenDoc); ok {
				// Use an adapter to make the document generator behave in a generic way.
				p = &adapters.ProviderGenDocToGen{ProviderGenDoc: pd}
			} else {
				continue
			}
		}
		msgs := genai.Messages{
			genai.NewTextMessage("Tell a story in 10 words."),
		}
		// Include options with some unsupported features to demonstrate UnsupportedContinuableError
		opts := &genai.OptionsText{
			TopK:      50, // Not all providers support this
			MaxTokens: 512,
		}
		response, err := p.GenSync(context.Background(), msgs, opts)
		if err != nil {
			if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
				fmt.Printf("- %s (ignored args: %s): %v\n", name, strings.Join(uce.Unsupported, ","), response)
			} else {
				fmt.Printf("- %s: %v\n", name, err)
			}
		} else {
			fmt.Printf("- %s: %v\n", name, response)
		}
	}
}

func Example_all_Full() {
	// This example includes:
	// - Making sure the provider implements genai.ProviderGen.
	// - Processing <think> tokens for explicit Chain-of-Thoughts models (e.g. Qwen3).
	var names []string
	for name := range providers.Available() {
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
	p, err := LoadProvider(*provider, &genai.OptionsProvider{Model: *model, Remote: *remote})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := p.GenSync(context.Background(), genai.Messages{genai.NewTextMessage(query)}, nil)
	if err != nil {
		log.Fatalf("failed to use provider %q: %s", *provider, err)
	}
	fmt.Printf("%s\n", resp.String())
}

// LoadProvider loads a provider.
func LoadProvider(provider string, opts *genai.OptionsProvider) (genai.ProviderGen, error) {
	if provider == "" {
		return nil, fmt.Errorf("no provider specified")
	}
	f := providers.All[provider]
	if f == nil {
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
	c, err := f(opts, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to provider %q: %w", provider, err)
	}
	p, ok := c.(genai.ProviderGen)
	if !ok {
		return nil, fmt.Errorf("provider %q doesn't implement genai.ProviderGen", provider)
	}
	// Wrap the provider with an adapter to process "<think>" tokens automatically ONLY if needed.
	p = adapters.WrapThinking(p)
	return p, nil
}

// LoadDefaultProviderGen loads a provider if there's exactly one available.
func LoadDefaultProviderGen() (genai.ProviderGen, error) {
	avail := providers.Available()
	if len(avail) == 1 {
		for name, f := range avail {
			c, err := f(&genai.OptionsProvider{Model: genai.ModelNone}, nil)
			if err != nil {
				return nil, err
			}
			p, ok := c.(genai.ProviderGen)
			if !ok {
				return nil, fmt.Errorf("provider %q doesn't implement genai.ProviderGen", name)
			}
			return p, nil
		}
	}
	if len(avail) == 0 {
		return nil, errors.New("no provider available; please set environment variables or specify a provider and API keys or remote URL")
	}
	names := make([]string, 0, len(avail))
	for name := range avail {
		names = append(names, name)
	}
	sort.Strings(names)
	return nil, fmt.Errorf("multiple providers available, select one of: %s", strings.Join(names, ", "))
}

func Example_available() {
	// Automatically select the provider available if there's only one. Asserts that the provider implements
	// ProviderGen.
	c, err := LoadDefaultProviderGen()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
	}
	// Wrap the provider with an adapter to ignore errors caused by unsupported features. Even if Seed is not
	// supported, no error will be returned.
	c = &adapters.ProviderGenIgnoreUnsupported{ProviderGen: c}
	msgs := genai.Messages{genai.NewTextMessage("Provide a life tip that sounds good but is actually a bad idea.")}
	resp, err := c.GenSync(context.Background(), msgs, &genai.OptionsText{Seed: 42})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
	}
	fmt.Printf("%s\n", resp.String())
}

func Example_all_GetProvidersGenAsync() {
	for _, name := range GetProvidersGenAsync() {
		fmt.Printf("%s\n", name)
	}
	// Output:
	// anthropic
	// bfl
	// gemini
	// openai
	// openaichat
}

// GetProvidersGenAsync returns all the providers that support asynchronous (batch) operations.
//
// It's really just a simple loop that iterates over each item in All and checks if it implements
// genai.ProviderGenAsync.
//
// Test:
//   - `c` if you want to determine if the functionality is potentially available, even if there's no known
//     API key available at the moment.
//   - `err` if you want to determine if the functionality is available in the current context, i.e.
//     environment variables FOO_API_KEY are set.
func GetProvidersGenAsync() []string {
	var names []string
	for name, f := range providers.All {
		c, _ := f(&genai.OptionsProvider{Model: genai.ModelNone}, nil)
		if c == nil {
			continue
		}
		if _, ok := c.(genai.ProviderGenAsync); ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}
