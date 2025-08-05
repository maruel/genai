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
	"github.com/maruel/genai/providers/openai/openaichat"
	"github.com/maruel/genai/providers/openai/openairesponses"
	"github.com/maruel/genai/providers/openaicompatible"
	"github.com/maruel/genai/providers/perplexity"
	"github.com/maruel/genai/providers/pollinations"
	"github.com/maruel/genai/providers/togetherai"
)

// All is a easy way to propose the user to load any of the supported provider.
var All = map[string]func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error){
	"anthropic": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return anthropic.New(opts, wrapper)
	},
	"bfl": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return bfl.New(opts, wrapper)
	},
	"cerebras": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return cerebras.New(opts, wrapper)
	},
	"cloudflare": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return cloudflare.New(opts, wrapper)
	},
	"cohere": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return cohere.New(opts, wrapper)
	},
	"deepseek": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return deepseek.New(opts, wrapper)
	},
	"gemini": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return gemini.New(opts, wrapper)
	},
	"groq": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return groq.New(opts, wrapper)
	},
	"huggingface": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return huggingface.New(opts, wrapper)
	},
	"llamacpp": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return llamacpp.New(opts, wrapper)
	},
	"mistral": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return mistral.New(opts, wrapper)
	},
	"ollama": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return ollama.New(opts, wrapper)
	},
	"openai": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return openai.New(opts, wrapper)
	},
	"openaichat": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return openaichat.New(opts, wrapper)
	},
	"openairesponses": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return openairesponses.New(opts, wrapper)
	},
	"openaicompatible": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return openaicompatible.New(opts, wrapper)
	},
	"perplexity": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return perplexity.New(opts, wrapper)
	},
	"pollinations": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return pollinations.New(opts, wrapper)
	},
	"togetherai": func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return togetherai.New(opts, wrapper)
	},
}
