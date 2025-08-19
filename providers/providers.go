// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package providers is the root of all known providers.
package providers

import (
	"context"
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
var All = map[string]func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error){
	"anthropic": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := anthropic.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"bfl": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := bfl.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"cerebras": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := cerebras.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"cloudflare": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := cloudflare.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"cohere": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := cohere.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"deepseek": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := deepseek.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"gemini": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := gemini.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"groq": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := groq.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"huggingface": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := huggingface.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"llamacpp": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := llamacpp.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"mistral": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := mistral.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"ollama": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := ollama.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"openai": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := openai.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"openaichat": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := openaichat.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"openairesponses": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := openairesponses.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"openaicompatible": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := openaicompatible.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"perplexity": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := perplexity.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"pollinations": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := pollinations.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"togetherai": func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := togetherai.New(opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
}

// Available returns the factories that are valid.
func Available() map[string]func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	avail := map[string]func(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error){}
	for name, f := range All {
		if c, err := f(&genai.ProviderOptions{Model: genai.ModelNone}, nil); err == nil {
			if p, ok := c.(genai.ProviderPing); ok {
				if err = p.Ping(context.Background()); err != nil {
					continue
				}
			}
			avail[name] = f
		}
	}
	return avail
}
