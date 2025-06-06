// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package providers_test

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers"
)

func ExampleAll_providerModel() {
	for name, factory := range providers.All {
		c, err := factory("")
		if err != nil {
			log.Fatal(err)
		}
		p, ok := c.(genai.ProviderModel)
		if !ok {
			continue
		}
		models, err := p.ListModels(context.Background())
		fmt.Printf("%s:\n", name)
		if err != nil {
			fmt.Printf("  Failed to get models: %v\n", err)
		}
		for _, model := range models {
			fmt.Printf("- %s\n", model)
		}
	}
}

func ExampleAll_providerGen() {
	for name, factory := range providers.All {
		c, err := factory("")
		if err != nil {
			log.Fatal(err)
		}
		p, ok := c.(genai.ProviderGen)
		if !ok {
			continue
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
