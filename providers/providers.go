// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package providers is the root of all known providers.
package providers

import (
	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/genai/providers/bfl"
	"github.com/maruel/genai/providers/cerebras"
	"github.com/maruel/genai/providers/cloudflare"
	"github.com/maruel/genai/providers/cohere"
	"github.com/maruel/genai/providers/deepseek"
	"github.com/maruel/genai/providers/gemini"
	"github.com/maruel/genai/providers/groq"
	"github.com/maruel/genai/providers/huggingface"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/mistral"
	"github.com/maruel/genai/providers/ollama"
	"github.com/maruel/genai/providers/openai"
	"github.com/maruel/genai/providers/perplexity"
	"github.com/maruel/genai/providers/pollinations"
	"github.com/maruel/genai/providers/togetherai"
)

// All is a easy way to propose the user to load any of the supported provider.
//
// It assumes the user has set the API key as an environment variable.
//
// The caveat for llamaccp is that the model is in fact the base URL.
var All = map[string]func(model string) (genai.Provider, error){
	"anthropic":    func(model string) (genai.Provider, error) { return anthropic.New("", model, nil) },
	"bfl":          func(model string) (genai.Provider, error) { return bfl.New("", model, nil) },
	"cerebras":     func(model string) (genai.Provider, error) { return cerebras.New("", model, nil) },
	"cloudflare":   func(model string) (genai.Provider, error) { return cloudflare.New("", "", model, nil) },
	"cohere":       func(model string) (genai.Provider, error) { return cohere.New("", model, nil) },
	"deepseek":     func(model string) (genai.Provider, error) { return deepseek.New("", model, nil) },
	"gemini":       func(model string) (genai.Provider, error) { return gemini.New("", model, nil) },
	"groq":         func(model string) (genai.Provider, error) { return groq.New("", model, nil) },
	"huggingface":  func(model string) (genai.Provider, error) { return huggingface.New("", model, nil) },
	"llamacpp":     func(model string) (genai.Provider, error) { return llamacpp.New(model, nil, nil) },
	"mistral":      func(model string) (genai.Provider, error) { return mistral.New("", model, nil) },
	"ollama":       func(model string) (genai.Provider, error) { return ollama.New("", model, nil) },
	"openai":       func(model string) (genai.Provider, error) { return openai.New("", model, nil) },
	"perplexity":   func(model string) (genai.Provider, error) { return perplexity.New("", model, nil) },
	"pollinations": func(model string) (genai.Provider, error) { return pollinations.New("", model, nil) },
	"togetherai":   func(model string) (genai.Provider, error) { return togetherai.New("", model, nil) },
}
