// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package providers is the root of all standard providers.
//
// It contains a registry of all known providers.
package providers

import (
	"context"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/genai/providers/baseten"
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

// Config is a registry entry.
type Config struct {
	APIKeyEnvVar string
	Alias        string
	Factory      func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error)
}

// All is a easy way to propose the user to load any of the supported provider.
//
// The keys are aliases and there can be duplicate aliases. "openai" links to "openairesponses". Use
// Provider.Name to get the real provider name.
var All = map[string]Config{
	"anthropic": {
		APIKeyEnvVar: "ANTHROPIC_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := anthropic.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"baseten": {
		APIKeyEnvVar: "BASETEN_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := baseten.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"bfl": {
		APIKeyEnvVar: "BFL_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := bfl.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"cerebras": {
		APIKeyEnvVar: "CEREBRAS_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := cerebras.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"cloudflare": {
		APIKeyEnvVar: "CLOUDFLARE_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := cloudflare.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"cohere": {
		APIKeyEnvVar: "COHERE_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := cohere.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"deepseek": {
		APIKeyEnvVar: "DEEPSEEK_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := deepseek.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"gemini": {
		APIKeyEnvVar: "GEMINI_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := gemini.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"groq": {
		APIKeyEnvVar: "GROQ_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := groq.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"huggingface": {
		APIKeyEnvVar: "HUGGINGFACE_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := huggingface.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"llamacpp": {
		APIKeyEnvVar: "",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := llamacpp.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"mistral": {
		APIKeyEnvVar: "MISTRAL_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := mistral.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"ollama": {
		APIKeyEnvVar: "",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := ollama.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"openai": {
		APIKeyEnvVar: "OPENAI_API_KEY",
		Alias:        "openairesponses",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := openairesponses.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"openaichat": {
		APIKeyEnvVar: "OPENAI_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := openaichat.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"openairesponses": {
		APIKeyEnvVar: "OPENAI_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := openairesponses.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"openaicompatible": {
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := openaicompatible.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"perplexity": {
		APIKeyEnvVar: "PERPLEXITY_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := perplexity.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"pollinations": {
		APIKeyEnvVar: "POLLINATIONS_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := pollinations.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
	"togetherai": {
		APIKeyEnvVar: "TOGETHER_API_KEY",
		Factory: func(ctx context.Context, opts ...genai.ProviderOption) (genai.Provider, error) {
			p, err := togetherai.New(ctx, opts...)
			if p == nil {
				return nil, err
			}
			return p, err
		},
	},
}

// Available returns the factories that are valid.
func Available(ctx context.Context) map[string]Config {
	avail := map[string]Config{}
	for name, cfg := range All {
		if c, err := cfg.Factory(ctx); err == nil {
			if p, ok := c.(genai.ProviderPing); ok {
				if err = p.Ping(ctx); err != nil {
					continue
				}
			}
			avail[name] = cfg
		}
	}
	return avail
}
