// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package providers_test

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers"
)

func Example_all_ProvidersModel() {
	for _, name := range GetProvidersModel() {
		c, err := providers.All[name]("", nil)
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
func GetProvidersModel() []string {
	var names []string
	for name, f := range providers.All {
		c, err := f("", nil)
		if err != nil {
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
		c, err := providers.All[name](base.PreferredCheap, nil)
		if err != nil {
			log.Fatal(err)
		}
		// c is guaranteed to implement ProviderGen.
		p := c.(genai.ProviderGen)
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

// GetProvidersGen returns all the providers that support generating messages.
//
// It's really just a simple loop that iterates over each item in All and checks if it implements
// genai.ProviderGen.
func GetProvidersGen() []string {
	var names []string
	for name, f := range providers.All {
		c, err := f("", nil)
		if err != nil {
			continue
		}
		if _, ok := c.(genai.ProviderGen); ok {
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
}

// GetProvidersGenAsync returns all the providers that support asynchronous (batch) operations.
//
// It's really just a simple loop that iterates over each item in All and checks if it implements
// genai.ProviderGenAsync.
func GetProvidersGenAsync() []string {
	var names []string
	for name, f := range providers.All {
		c, err := f("", nil)
		if err != nil {
			continue
		}
		if _, ok := c.(genai.ProviderGenAsync); ok {
			names = append(names, name)
		}
	}
	sort.Strings(names)
	return names
}
