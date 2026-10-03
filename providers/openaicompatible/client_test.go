// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenAI-compatible provider client.

package openaicompatible_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openaicompatible"
)

// Testing is very different here as we test various providers to see if they work with this generic provider.
func TestClient(t *testing.T) {
	t.Run("ToolRoundTrip", testClientTools)
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})

	getClient := func(t *testing.T, provider string) genai.Provider {
		t.Parallel()
		p := providers[provider]
		apiKey := internaltest.GetEnv(p.envAPIKey)
		if apiKey == "" {
			apiKey = "<insert_api_key_here>"
		}
		wrapper := func(h http.RoundTripper) http.RoundTripper {
			return &roundtrippers.Header{
				Header:    p.header(apiKey),
				Transport: testRecorder.Record(t, h),
			}
		}
		c, err := openaicompatible.New(t.Context(), genai.ProviderOptionTransportWrapper(wrapper), genai.ProviderOptionRemote(p.chatURL), genai.ProviderOptionModel(p.model))
		if c != nil {
			internaltest.CleanupCloser(t, c)
		}
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
				opts := genai.GenOptionText{Temperature: 0.01, MaxTokens: 2000}
				ctx := t.Context()
				resp, err := c.GenSync(ctx, msgs, &opts, genai.GenOptionSeed(1))
				if err != nil {
					if ent, ok := errors.AsType[*base.ErrNotSupported](err); !ok || !slices.Contains(ent.Options, "GenOptionSeed") {
						t.Fatal(err)
					}
					// Try again without seed.
					if resp, err = c.GenSync(ctx, msgs, &opts); err != nil {
						t.Fatal(err)
					}
				}
				t.Logf("Raw response: %#v", resp)
				if len(resp.Replies) == 0 {
					t.Fatal("missing response")
				}
				internaltest.ValidateWordResponse(t, &resp, "hello")
			})
		}
	})

	t.Run("GenStream_simple", func(t *testing.T) {
		for name := range providers {
			t.Run(name, func(t *testing.T) {
				c := getClient(t, name)
				msgs := genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
				opts := genai.GenOptionText{Temperature: 0.01, MaxTokens: 2000}
				fragments, finish := c.GenStream(t.Context(), msgs, &opts, genai.GenOptionSeed(1))
				for f := range fragments {
					t.Logf("Packet: %#v", f)
				}
				res, err := finish()
				if err != nil {
					if ent, ok := errors.AsType[*base.ErrNotSupported](err); !ok || !slices.Contains(ent.Options, "GenOptionSeed") {
						t.Fatal(err)
					}
					// Try again without seed.
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
				internaltest.ValidateWordResponse(t, &res, "hello")
			})
		}
	})
}

func testClientTools(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			calls := 0
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var req openaicompatible.ChatRequest
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				calls++
				if req.ToolChoice != "auto" || len(req.Tools) != 1 || req.Tools[0].Function.Name != "weather" {
					t.Errorf("tools: %#v", req)
				}
				body := `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"a","type":"function","function":{"name":"weather","arguments":""}}]},"finish_reason":"stop"}],"extension":true}`
				if req.Stream {
					body = `{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"a","function":{"name":"weather","arguments":""}}]},"finish_reason":"stop"}],"extension":true}`
				}
				if calls == 2 {
					if len(req.Messages) != 3 || len(req.Messages[1].ToolCalls) != 1 || req.Messages[2].Role != "tool" || req.Messages[2].ToolCallID != "a" || len(req.Messages[2].Content) != 1 || req.Messages[2].Content[0].Text != "sunny" {
						t.Errorf("tool history: %#v", req.Messages)
					}
					body = `{"choices":[{"message":{"role":"assistant","content":"sunny"},"finish_reason":"stop"}]}`
					if req.Stream {
						body = `{"choices":[{"delta":{"content":"sunny"},"finish_reason":"stop"}]}`
					}
				}
				if req.Stream {
					w.Header().Set("Content-Type", "text/event-stream")
					body = "data: " + body + "\n\ndata: [DONE]\n\n"
				} else {
					w.Header().Set("Content-Type", "application/json")
				}
				if _, err := w.Write([]byte(body)); err != nil {
					t.Error(err)
				}
			}))
			t.Cleanup(srv.Close)
			c, err := openaicompatible.New(t.Context(), genai.ProviderOptionRemote(srv.URL))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			opts := &genai.GenOptionTools{Tools: []genai.ToolDef{{Name: "weather", Description: "Get weather", Callback: func(context.Context, *struct{}) (string, error) { return "sunny", nil }}}}
			msgs := make(genai.Messages, 1, 3)
			msgs[0] = genai.NewTextMessage("Weather?")
			generate := func() (genai.Result, error) {
				if !stream {
					return c.GenSync(t.Context(), msgs, opts)
				}
				fragments, finish := c.GenStream(t.Context(), msgs, opts)
				for range fragments {
				}
				return finish()
			}
			res, err := generate()
			if err != nil {
				t.Fatal(err)
			}
			if res.Usage.FinishReason != genai.FinishedToolCalls {
				t.Fatalf("finish: %q", res.Usage.FinishReason)
			}
			result, err := res.DoToolCalls(t.Context(), opts.Tools)
			if err != nil {
				t.Fatal(err)
			}
			msgs = append(msgs, res.Message, result)
			res, err = generate()
			if err != nil {
				t.Fatal(err)
			}
			if calls != 2 || len(res.Replies) != 1 || res.Replies[0].Text != "sunny" {
				t.Fatalf("result: %#v; requests: %d", res, calls)
			}
		})
	}
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
