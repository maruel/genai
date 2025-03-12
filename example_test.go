// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai_test

import (
	"context"
	"fmt"
	"os"

	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/genai/openai"
	"github.com/maruel/genai/perplexity"
)

func Example_modelProvider() {
	// Pro-tip: Using os.Stderr so if you modify this file and append a "// Output: foo"
	// at the end of this function, "go test" will run the code and stream the
	// output to you.

	// Notably missing: "llamacpp" and "perplexity".
	providers := map[string]genaiapi.ModelProvider{
		"anthropic":   &anthropic.Client{ApiKey: os.Getenv("ANTHROPIC_API_KEY")},
		"cerebras":    &cerebras.Client{ApiKey: os.Getenv("CEREBRAS_API_KEY")},
		"cloudflare":  &cloudflare.Client{AccountID: os.Getenv("CLOUDFLARE_ACCOUNT_ID"), ApiKey: os.Getenv("CLOUDFLARE_API_KEY")},
		"cohere":      &cohere.Client{ApiKey: os.Getenv("COHERE_API_KEY")},
		"deepseek":    &deepseek.Client{ApiKey: os.Getenv("DEEPSEEK_API_KEY")},
		"gemini":      &gemini.Client{ApiKey: os.Getenv("GEMINI_API_KEY")},
		"groq":        &groq.Client{ApiKey: os.Getenv("GROQ_API_KEY")},
		"huggingface": &huggingface.Client{},
		"mistral":     &mistral.Client{ApiKey: os.Getenv("MISTRAL_API_KEY")},
		"openai":      &openai.Client{ApiKey: os.Getenv("OPENAI_API_KEY")},
	}
	for name, p := range providers {
		models, err := p.ListModels(context.Background())
		fmt.Fprintf(os.Stderr, "%s:\n", name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Failed to get models: %v\n", err)
		}
		if len(models) == 0 {
			continue
		}
		for _, model := range models {
			fmt.Fprintf(os.Stderr, "- %s\n", model)
		}
	}
}

func Example_completionProvider() {
	// Pro-tip: Using os.Stderr so if you modify this file and append a "// Output: foo"
	// at the end of this function, "go test" will run the code and stream the
	// output to you.

	providers := map[string]genaiapi.CompletionProvider{
		"anthropic": &anthropic.Client{
			ApiKey: os.Getenv("ANTHROPIC_API_KEY"),
			// https://docs.anthropic.com/en/docs/about-claude/models/all-models
			Model: "claude-3-7-sonnet-latest",
		},
		"cerebras": &cerebras.Client{
			ApiKey: os.Getenv("CEREBRAS_API_KEY"),
			Model:  "llama-3.3-70b",
		},
		"cloudflare": &cloudflare.Client{
			AccountID: os.Getenv("CLOUDFLARE_ACCOUNT_ID"),
			ApiKey:    os.Getenv("CLOUDFLARE_API_KEY"),
			// https://developers.cloudflare.com/workers-ai/models/
			Model: "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
		},
		"cohere": &cohere.Client{
			ApiKey: os.Getenv("COHERE_API_KEY"),
			// https://docs.cohere.com/v2/docs/models
			Model: "command-r-plus",
		},
		"deepseek": &deepseek.Client{
			ApiKey: os.Getenv("DEEPSEEK_API_KEY"),
			Model:  "deepseek-reasoner",
		},
		"gemini": &gemini.Client{
			ApiKey: os.Getenv("GEMINI_API_KEY"),
			// https://ai.google.dev/gemini-api/docs/models/gemini
			Model: "gemini-2.0-flash",
		},
		"groq": &groq.Client{
			ApiKey: os.Getenv("GROQ_API_KEY"),
			// https://console.groq.com/docs/models
			Model: "qwen-qwq-32b",
		},
		"huggingface": &huggingface.Client{
			// https://huggingface.co/models?inference=warm&sort=trending
			Model: "Qwen/QwQ-32B",
		},
		"llamacpp": &llamacpp.Client{
			BaseURL: "http://localhost:8080",
		},
		"mistral": &mistral.Client{
			ApiKey: os.Getenv("MISTRAL_API_KEY"),
			// https://docs.mistral.ai/getting-started/models/models_overview/
			Model: "mistral-large-latest",
		},
		"openai": &openai.Client{
			ApiKey: os.Getenv("OPENAI_API_KEY"),
			// https://platform.openai.com/docs/api-reference/models
			Model: "o3-mini",
		},
		"perplexity": &perplexity.Client{
			ApiKey: os.Getenv("PERPLEXITY_API_KEY"),
		},
	}
	for name, provider := range providers {
		msgs := []genaiapi.Message{
			{
				Role: genaiapi.User,
				Type: genaiapi.Text,
				Text: "Tell a story in 10 words.",
			},
		}
		response, err := provider.Completion(context.Background(), msgs, nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "- %s: %v\n", name, err)
		} else {
			fmt.Fprintf(os.Stderr, "- %s: %s\n", name, response)
		}
	}
}
