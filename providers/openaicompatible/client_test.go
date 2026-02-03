// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openaicompatible_test

import (
	"errors"
	"net/http"
	"os"
	"slices"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openaicompatible"
	"github.com/maruel/roundtrippers"
)

// Testing is very different here as we test various providers to see if they work with this generic provider.
func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})

	getClient := func(t *testing.T, provider string) genai.Provider {
		t.Parallel()
		p := providers[provider]
		apiKey := os.Getenv(p.envAPIKey)
		if apiKey == "" {
			apiKey = "<insert_api_key_here>"
		}
		wrapper := func(h http.RoundTripper) http.RoundTripper {
			return &roundtrippers.Header{
				Header:    p.header(apiKey),
				Transport: testRecorder.Record(t, h),
			}
		}
		c, err := openaicompatible.New(t.Context(), &genai.ProviderOptions{Remote: p.chatURL, Model: p.model}, wrapper)
		if err != nil {
			t.Fatal(err)
		}
		return c
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, "openai"))
	})

	// Note: Skipping Preferred test as openaicompatible is a generic provider
	// without its own preferred models in the scoreboard.

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, "openai")
		})
	})

	t.Run("GenSync_simple", func(t *testing.T) {
		for name := range providers {
			t.Run(name, func(t *testing.T) {
				c := getClient(t, name)
				msgs := genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
				opts := genai.GenOptionsText{Temperature: 0.01, MaxTokens: 2000, Seed: 1}
				ctx := t.Context()
				resp, err := c.GenSync(ctx, msgs, &opts)
				if err != nil {
					var ent *base.ErrNotSupported
					if !errors.As(err, &ent) || !slices.Contains(ent.Options, "GenOptionsText.Seed") {
						t.Fatal(err)
					}
					// Try again without seed.
					opts.Seed = 0
					if resp, err = c.GenSync(ctx, msgs, &opts); err != nil {
						t.Fatal(err)
					}
				}
				t.Logf("Raw response: %#v", resp)
				if len(resp.Replies) == 0 {
					t.Fatal("missing response")
				}
				internaltest.ValidateWordResponse(t, resp, "hello")
			})
		}
	})

	t.Run("GenStream_simple", func(t *testing.T) {
		for name := range providers {
			t.Run(name, func(t *testing.T) {
				c := getClient(t, name)
				msgs := genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
				opts := genai.GenOptionsText{Temperature: 0.01, MaxTokens: 2000, Seed: 1}
				fragments, finish := c.GenStream(t.Context(), msgs, &opts)
				for f := range fragments {
					t.Logf("Packet: %#v", f)
				}
				res, err := finish()
				if err != nil {
					var ent *base.ErrNotSupported
					if !errors.As(err, &ent) || !slices.Contains(ent.Options, "GenOptionsText.Seed") {
						t.Fatal(err)
					}
					// Try again without seed.
					opts.Seed = 0
					fragments, finish := c.GenStream(t.Context(), msgs, &opts)
					for f := range fragments {
						t.Logf("Packet: %#v", f)
					}
					if res, err = finish(); err != nil {
						t.Fatal(err)
					}
				}
				t.Logf("Raw response: %#v", res)
				if len(res.Replies) == 0 {
					t.Fatal("missing response")
				}
				internaltest.ValidateWordResponse(t, res, "hello")
			})
		}
	})
}

type provider struct {
	envAPIKey string
	chatURL   string
	header    func(apiKey string) http.Header
	model     string
}

// Keep in sync with ExampleProviderGen_all in ../example_test.go.
var providers = map[string]provider{
	"anthropic": {
		envAPIKey: "ANTHROPIC_API_KEY",
		chatURL:   "https://api.anthropic.com/v1/messages",
		header: func(apiKey string) http.Header {
			return http.Header{"x-api-key": {apiKey}, "anthropic-version": {"2023-06-01"}}
		},
		model: "claude-3-haiku-20240307",
	},
	"cerebras": {
		envAPIKey: "CEREBRAS_API_KEY",
		chatURL:   "https://api.cerebras.ai/v1/chat/completions",
		header: func(apiKey string) http.Header {
			return http.Header{"Authorization": {"Bearer " + apiKey}}
		},
		model: "llama-3.1-8b",
	},
	// "cloudflare": {
	// 	envAPIKey:       "CLOUDFLARE_API_KEY",
	// 	envAccountIDKey: "CLOUDFLARE_ACCOUNT_ID",
	// 	chatURL:         "https://api.cloudflare.com/client/v4/accounts/" + accountID + "/ai/run/" + model,
	// 	header: func(apiKey string) http.Header {
	// 		return http.Header{"Authorization": {"Bearer " + apiKey}}
	// 	},
	// 	model: "@cf/meta/llama-3.2-3b-instruct",
	// },
	"cohere": {
		envAPIKey: "COHERE_API_KEY",
		chatURL:   "https://api.cohere.com/v2/chat",
		header: func(apiKey string) http.Header {
			return http.Header{"Authorization": {"Bearer " + apiKey}}
		},
		model: "command-r7b-12-2024",
	},
	"deepseek": {
		envAPIKey: "DEEPSEEK_API_KEY",
		chatURL:   "https://api.deepseek.com/chat/completions",
		header: func(apiKey string) http.Header {
			return http.Header{"Authorization": {"Bearer " + apiKey}}
		},
		model: "deepseek-chat",
	},
	"groq": {
		envAPIKey: "GROQ_API_KEY",
		chatURL:   "https://api.groq.com/openai/v1/chat/completions",
		header: func(apiKey string) http.Header {
			return http.Header{"Authorization": {"Bearer " + apiKey}}
		},
		model: "llama3-8b-8192",
	},
	// "huggingface": {
	// 	envAPIKey: "HUGGINGFACE_API_KEY",
	// 	// chatURL:   "https://router.huggingface.co/hf-inference/models/" + model + "/v1/chat/completions",
	// 	header: func(apiKey string) http.Header {
	// 		return http.Header{"Authorization": {"Bearer " + apiKey}}
	// 	},
	// 	model: "meta-llama/Llama-3.3-70B-Instruct",
	// },
	"mistral": {
		envAPIKey: "MISTRAL_API_KEY",
		chatURL:   "https://api.mistral.ai/v1/chat/completions",
		header: func(apiKey string) http.Header {
			return http.Header{"Authorization": {"Bearer " + apiKey}}
		},
		model: "ministral-3b-latest",
	},
	"openai": {
		envAPIKey: "OPENAI_API_KEY",
		chatURL:   "https://api.openai.com/v1/chat/completions",
		header: func(apiKey string) http.Header {
			return http.Header{"Authorization": {"Bearer " + apiKey}}
		},
		model: "gpt-4.1-nano",
	},
	// "perplexity": {
	// 	envAPIKey: "PERPLEXITY_API_KEY",
	// 	chatURL:   "https://api.perplexity.ai/chat/completions",
	// 	header: func(apiKey string) http.Header {
	// 		return http.Header{"Authorization": {"Bearer " + apiKey}}
	// 	},
	// 	model:         "sonar",
	// 	thinkingStart: "<think>",
	// },
	"togetherai": {
		envAPIKey: "TOGETHER_API_KEY",
		chatURL:   "https://api.together.xyz/v1/chat/completions",
		header: func(apiKey string) http.Header {
			return http.Header{"Authorization": {"Bearer " + apiKey}}
		},
		model: "meta-llama/Llama-3.2-3B-Instruct-Turbo",
	},
}
