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
	"github.com/maruel/genai/providers/openaichat"
	"github.com/maruel/genai/providers/openaicompatible"
	"github.com/maruel/genai/providers/openairesponses"
	"github.com/maruel/genai/providers/perplexity"
	"github.com/maruel/genai/providers/pollinations"
	"github.com/maruel/genai/providers/togetherai"
)

// All is a easy way to propose the user to load any of the supported provider.
//
// The keys are aliases and there can be duplicate aliases. As of now, "openai" links to "openaichat". Use
// Provider.Name to get the real provider name.
var All = map[string]func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error){
	"anthropic": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := anthropic.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"bfl": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := bfl.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"cerebras": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := cerebras.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"cloudflare": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := cloudflare.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"cohere": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := cohere.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"deepseek": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := deepseek.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"gemini": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := gemini.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"groq": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := groq.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"huggingface": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := huggingface.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"llamacpp": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := llamacpp.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"mistral": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := mistral.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"ollama": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := ollama.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	// Create an alias for "openai" that refers to openaichat.
	"openai": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := openaichat.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"openaichat": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := openaichat.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"openairesponses": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := openairesponses.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"openaicompatible": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := openaicompatible.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"perplexity": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := perplexity.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"pollinations": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := pollinations.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
	"togetherai": func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		p, err := togetherai.New(ctx, opts, wrapper)
		if p == nil {
			return nil, err
		}
		return p, err
	},
}

// Available returns the factories that are valid.
func Available(ctx context.Context) map[string]func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	avail := map[string]func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error){}
	for name, f := range All {
		if c, err := f(ctx, &genai.ProviderOptions{Model: genai.ModelNone}, nil); err == nil {
			if p, ok := c.(genai.ProviderPing); ok {
				if err = p.Ping(ctx); err != nil {
					continue
				}
			}
			avail[name] = f
		}
	}
	return avail
}
