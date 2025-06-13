// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package providers is the root of all known providers.
package providers

import (
	"net/http"

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
// # Caveats
//
// - cloudflare: the account ID must be set as an environment variable.
// - llamaccp: the model is in fact the base URL.
// - openaicompatible: it is not included since it needs both the base URL and the model name.
var All = map[string]func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error){
	"anthropic": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return anthropic.New("", model, wrapper)
	},
	"bfl": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return bfl.New("", model, wrapper)
	},
	"cerebras": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return cerebras.New("", model, wrapper)
	},
	"cloudflare": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return cloudflare.New("", "", model, wrapper)
	},
	"cohere": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return cohere.New("", model, wrapper)
	},
	"deepseek": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return deepseek.New("", model, wrapper)
	},
	"gemini": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return gemini.New("", model, wrapper)
	},
	"groq": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return groq.New("", model, wrapper)
	},
	"huggingface": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return huggingface.New("", model, wrapper)
	},
	"llamacpp": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return llamacpp.New(model, nil, wrapper)
	},
	"mistral": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return mistral.New("", model, wrapper)
	},
	"ollama": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return ollama.New("", model, wrapper)
	},
	"openai": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return openai.New("", model, wrapper)
	},
	"perplexity": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return perplexity.New("", model, wrapper)
	},
	"pollinations": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return pollinations.New("", model, wrapper)
	},
	"togetherai": func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return togetherai.New("", model, wrapper)
	},
}
