// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai_test

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/genai/openai"
	"github.com/maruel/genai/perplexity"
	"github.com/maruel/genai/togetherai"
)

func ExampleModelProvider() {
	// Pro-tip: Using os.Stderr so if you modify this file and append a "// Output: foo"
	// at the end of this function, "go test" will run the code and stream the
	// output to you.

	modelProviders := map[string]genai.ModelProvider{}
	if c, err := anthropic.New("", ""); err == nil {
		modelProviders["anthropic"] = c
	}
	if c, err := cerebras.New("", ""); err == nil {
		modelProviders["cerebras"] = c
	}
	if c, err := cloudflare.New("", "", ""); err == nil {
		modelProviders["cloudflare"] = c
	}
	if c, err := cohere.New("", ""); err == nil {
		modelProviders["cohere"] = c
	}
	if c, err := deepseek.New("", ""); err == nil {
		modelProviders["deepseek"] = c
	}
	if c, err := gemini.New("", ""); err == nil {
		modelProviders["gemini"] = c
	}
	if c, err := groq.New("", ""); err == nil {
		modelProviders["groq"] = c
	}
	if c, err := huggingface.New("", ""); err == nil {
		modelProviders["huggingface"] = c
	}
	// llamapcpp doesn't implement ModelProvider.
	if c, err := mistral.New("", ""); err == nil {
		modelProviders["mistral"] = c
	}
	if c, err := openai.New("", ""); err == nil {
		modelProviders["openai"] = c
	}
	// perplexity doesn't implement ModelProvider.
	if c, err := togetherai.New("", ""); err == nil {
		modelProviders["togetherai"] = c
	}

	for name, p := range modelProviders {
		models, err := p.ListModels(context.Background())
		fmt.Fprintf(os.Stderr, "%s:\n", name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  Failed to get models: %v\n", err)
		}
		for _, model := range models {
			fmt.Fprintf(os.Stderr, "- %s\n", model)
		}
	}
}

func ExampleToolCall_Call() {
	// Define a tool that adds two numbers
	tool := genai.ToolDef{
		Name:        "add",
		Description: "Add two numbers together",
		InputsAs: &struct {
			A int `json:"a"`
			B int `json:"b"`
		}{},
		Callback: func(input *struct {
			A int `json:"a"`
			B int `json:"b"`
		},
		) string {
			return fmt.Sprintf("%d + %d = %d", input.A, input.B, input.A+input.B)
		},
	}

	// Create a tool call that would come from an LLM
	toolCall := genai.ToolCall{
		ID:        "call1",
		Name:      "add",
		Arguments: `{"a": 5, "b": 3}`,
	}

	// Invoke the tool with the arguments
	result, err := toolCall.Call(&tool)
	if err != nil {
		fmt.Printf("Error calling tool: %v\n", err)
		return
	}

	// Print the result
	fmt.Println(result)
	// Output: 5 + 3 = 8
}

func ExampleChatProvider() {
	// Pro-tip: Using os.Stderr so if you modify this file and append a "// Output: foo"
	// at the end of this function, "go test" will run the code and stream the
	// output to you.
	completionProviders := map[string]genai.ChatProvider{}
	// https://docs.anthropic.com/en/docs/about-claude/models/all-models
	if c, err := anthropic.New("", "claude-3-7-sonnet-latest"); err == nil {
		completionProviders["anthropic"] = c
	}
	if c, err := cerebras.New("", "llama-3.3-70b"); err == nil {
		completionProviders["cerebras"] = c
	}
	// https://developers.cloudflare.com/workers-ai/models/
	if c, err := cloudflare.New("", "", "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"); err == nil {
		completionProviders["cloudflare"] = c
	}
	// https://docs.cohere.com/v2/docs/models
	if c, err := cohere.New("", "command-r-plus"); err == nil {
		completionProviders["cohere"] = c
	}
	if c, err := deepseek.New("", "deepseek-reasoner"); err == nil {
		completionProviders["deepseek"] = c
	}
	// https://ai.google.dev/gemini-api/docs/models/gemini
	if c, err := gemini.New("", "gemini-2.0-flash"); err == nil {
		completionProviders["gemini"] = c
	}
	// https://console.groq.com/docs/models
	if c, err := groq.New("", "qwen-qwq-32b"); err == nil {
		completionProviders["groq"] = c
	}
	// https://huggingface.co/models?inference=warm&sort=trending
	if c, err := huggingface.New("", "Qwen/QwQ-32B"); err == nil {
		completionProviders["huggingface"] = c
	}
	if false {
		// See llamacpp/llamacppsrv to see how to run a local server.
		if c, err := llamacpp.New("http://localhost:8080", nil); err == nil {
			completionProviders["llamacpp"] = c
		}
	}
	// https://docs.mistral.ai/getting-started/models/models_overview/
	if c, err := mistral.New("", "mistral-large-latest"); err == nil {
		completionProviders["mistral"] = c
	}
	// https://platform.openai.com/docs/api-reference/models
	if c, err := openai.New("", "o3-mini"); err == nil {
		completionProviders["openai"] = c
	}
	if c, err := perplexity.New("", "sonar"); err == nil {
		completionProviders["perplexity"] = c
	}
	if c, err := togetherai.New("", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"); err == nil {
		completionProviders["togetherai"] = c
	}

	for name, provider := range completionProviders {
		msgs := genai.Messages{
			genai.NewTextMessage(genai.User, "Tell a story in 10 words."),
		}
		// Include options with some unsupported features to demonstrate UnsupportedContinuableError
		opts := &genai.ChatOptions{
			TopK:      50, // Not all providers support this
			MaxTokens: 512,
		}
		response, err := provider.Chat(context.Background(), msgs, opts)
		if err != nil {
			if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
				fmt.Fprintf(os.Stderr, "- %s (ignored args: %s): %v\n", name, strings.Join(uce.Unsupported, ","), response)
			} else {
				fmt.Fprintf(os.Stderr, "- %s: %v\n", name, err)
			}
		} else {
			fmt.Fprintf(os.Stderr, "- %s: %v\n", name, response)
		}
	}
}
