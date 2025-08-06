// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package providers_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers"
)

func Example_all_ProvidersModel() {
	for _, name := range GetProvidersModel() {
		c, err := providers.All[name](&genai.OptionsProvider{Model: base.NoModel}, nil)
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
		c, _ := f(&genai.OptionsProvider{Model: base.NoModel}, nil)
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
	for _, name := range GetProvidersGen() {
		c, err := providers.All[name](&genai.OptionsProvider{Model: base.PreferredCheap}, nil)
		if err != nil {
			log.Fatal(err)
		}
		p, ok := c.(genai.ProviderGen)
		if !ok {
			// Use an adapter to make the document generator behave in a generic way.
			p = &adapters.ProviderGenDocToGen{ProviderGenDoc: c.(genai.ProviderGenDoc)}
		}
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Tell a story in 10 words."),
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

func LoadProviderGen() (genai.ProviderGen, error) {
	avail := providers.Available()
	if len(avail) == 1 {
		for name, f := range avail {
			c, err := f(&genai.OptionsProvider{Model: base.NoModel}, nil)
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

func ExampleAvailable() {
	// Automatically select the provider available if there's only one. Asserts that the provider implements
	// ProviderGen.
	_, err := LoadProviderGen()
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
	}
}

// GetProvidersGen returns all the providers that support generating messages.
//
// It's really just a simple loop that iterates over each item in All and checks if it implements
// genai.ProviderGen (general) or genai.ProviderGenDoc (specialized for non-text generation).
//
// Test:
//   - `c` if you want to determine if the functionality is potentially available, even if there's no known
//     API key available at the moment.
//   - `err` if you want to determine if the functionality is available in the current context, i.e.
//     environment variables FOO_API_KEY are set.
func GetProvidersGen() []string {
	var names []string
	for name, f := range providers.All {
		c, _ := f(&genai.OptionsProvider{Model: base.NoModel}, nil)
		if c == nil {
			continue
		}
		if _, ok := c.(genai.ProviderGen); ok {
			names = append(names, name)
		} else if _, ok := c.(genai.ProviderGenDoc); ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}

func Example_all_GetProvidersGenAsync() {
	for _, name := range GetProvidersGenAsync() {
		fmt.Printf("%s\n", name)
	}
	// Output:
	// anthropic
	// bfl
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
		c, _ := f(&genai.OptionsProvider{Model: base.NoModel}, nil)
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
