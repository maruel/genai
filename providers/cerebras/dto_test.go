// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Cerebras provider DTOs.

package cerebras_test

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cerebras"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestReasoningFormat(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			err := cerebras.ReasoningFormat("chatty").Validate()
			if err == nil {
				t.Fatal("Validate() succeeded, want error")
			}
		})
	})
}

func TestServiceTier(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			err := cerebras.ServiceTier("turbo").Validate()
			if err == nil {
				t.Fatal("Validate() succeeded, want error")
			}
		})
	})
}

func TestToolChoice(t *testing.T) {
	t.Run("MarshalJSON", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			err := (cerebras.ToolChoice{Mode: cerebras.ToolChoiceRequired, Function: "lookup"}).Validate()
			if err == nil {
				t.Fatal("Validate() succeeded, want error")
			}
		})
	})
}

func TestPredictionContent(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, content := range []cerebras.PredictionContent{
				{},
				{Type: cerebras.ContentImageURL, Text: "known"},
			} {
				if err := content.Validate(); err == nil {
					t.Error("Validate() succeeded, want error")
				}
			}
		})
	})
}

func TestPrediction(t *testing.T) {
	t.Run("MarshalJSON", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			data, err := json.Marshal(cerebras.Prediction{Content: []cerebras.PredictionContent{{
				Type: cerebras.ContentText,
				Text: "known",
			}}})
			if err != nil {
				t.Fatal(err)
			}
			const want = `{"type":"content","content":[{"type":"text","text":"known"}]}`
			if string(data) != want {
				t.Errorf("JSON = %s, want %s", data, want)
			}
		})
		t.Run("error", func(t *testing.T) {
			err := (cerebras.Prediction{Text: "text", Content: []cerebras.PredictionContent{{Text: "content"}}}).Validate()
			if err == nil {
				t.Fatal("Validate() succeeded, want error")
			}
		})
		t.Run("unsupportedContentType", func(t *testing.T) {
			err := (cerebras.Prediction{Content: []cerebras.PredictionContent{{Type: "image", Text: "known"}}}).Validate()
			if err == nil {
				t.Fatal("Validate() succeeded, want error")
			}
		})
	})
}

func TestChatRequestPredictionContent(t *testing.T) {
	var got cerebras.ChatRequest
	err := got.Init(genai.Messages{{Requests: []genai.Request{{Text: "hello"}}}}, "gemma-4-31b", &cerebras.GenOption{
		Prediction: cerebras.Prediction{Content: []cerebras.PredictionContent{{Text: "known"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Prediction.Content[0].Type != cerebras.ContentText {
		t.Errorf("Prediction.Content[0].Type = %q, want %q", got.Prediction.Content[0].Type, cerebras.ContentText)
	}
}

func TestQueueThreshold(t *testing.T) {
	t.Run("GenSync", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			var got string
			c, err := cerebras.New(t.Context(),
				genai.ProviderOptionAPIKey("api-key"),
				genai.ProviderOptionModel("gemma-4-31b"),
				cerebras.ProviderOptionQueueThreshold(100*time.Millisecond),
				genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
					return roundTripperFunc(func(r *http.Request) (*http.Response, error) {
						got = r.Header.Get("queue_threshold")
						return &http.Response{
							StatusCode: http.StatusOK,
							Header:     http.Header{"Content-Type": {"application/json"}},
							Body:       io.NopCloser(strings.NewReader(`{"id":"id","model":"gemma-4-31b","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hello"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)),
							Request:    r,
						}, nil
					})
				}),
			)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.GenSync(t.Context(), genai.Messages{{Requests: []genai.Request{{Text: "hello"}}}}, &cerebras.GenOption{
				QueueThreshold: 50 * time.Millisecond,
			}); err != nil {
				t.Fatal(err)
			}
			if got != "50" {
				t.Errorf("queue_threshold header = %q, want 50", got)
			}
			got = ""
			if _, err := c.GenSync(t.Context(), genai.Messages{{Requests: []genai.Request{{Text: "hello"}}}}); err != nil {
				t.Fatal(err)
			}
			if got != "100" {
				t.Errorf("queue_threshold header = %q, want 100", got)
			}
		})
		t.Run("providerOptionError", func(t *testing.T) {
			cl, err := cerebras.New(t.Context(), cerebras.ProviderOptionQueueThreshold(49*time.Millisecond))
			if cl != nil {
				internaltest.CleanupCloser(t, cl)
			}
			if err == nil {
				t.Fatal("New() succeeded, want error")
			}
		})
	})
}

func TestReasoningEffort(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			err := cerebras.ReasoningEffort("maximum").Validate()
			if err == nil {
				t.Fatal("Validate() succeeded, want error")
			}
		})
	})
}

func TestChatRequest(t *testing.T) {
	t.Run("Init", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("providerOptions", func(t *testing.T) {
				var got cerebras.ChatRequest
				err := got.Init(genai.Messages{{Requests: []genai.Request{{Text: "hello"}}}}, "gpt-oss-120b", &cerebras.GenOption{
					Prediction:      cerebras.Prediction{Text: "known output"},
					PromptCacheKey:  "conversation-1",
					ReasoningFormat: cerebras.ReasoningFormatParsed,
					ServiceTier:     cerebras.ServiceTierFlex,
					ToolChoice:      cerebras.ToolChoice{Function: "lookup"},
				})
				if err != nil {
					t.Fatal(err)
				}
				data, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				for _, want := range []string{
					`"prediction":{"type":"content","content":"known output"}`,
					`"prompt_cache_key":"conversation-1"`,
					`"reasoning_format":"parsed"`,
					`"service_tier":"flex"`,
					`"tool_choice":{"type":"function","function":{"name":"lookup"}}`,
				} {
					if !strings.Contains(string(data), want) {
						t.Errorf("ChatRequest JSON = %s, want %s", data, want)
					}
				}
			})
			t.Run("reasoningEffort", func(t *testing.T) {
				var got cerebras.ChatRequest
				err := got.Init(genai.Messages{{Requests: []genai.Request{{Text: "hello"}}}}, "gemma-4-31b", &cerebras.GenOption{
					ReasoningEffort: cerebras.ReasoningEffortHigh,
				})
				if err != nil {
					t.Fatal(err)
				}
				if got.ReasoningEffort != cerebras.ReasoningEffortHigh {
					t.Errorf("ReasoningEffort = %q, want %q", got.ReasoningEffort, cerebras.ReasoningEffortHigh)
				}
				data, err := json.Marshal(got)
				if err != nil {
					t.Fatal(err)
				}
				if !strings.Contains(string(data), `"reasoning_effort":"high"`) {
					t.Errorf("ChatRequest JSON = %s, want reasoning_effort=high", data)
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			t.Run("promptCacheKey", func(t *testing.T) {
				var got cerebras.ChatRequest
				err := got.Init(genai.Messages{{Requests: []genai.Request{{Text: "hello"}}}}, "gemma-4-31b", &cerebras.GenOption{
					PromptCacheKey: strings.Repeat("a", 1025),
				})
				if err == nil {
					t.Fatal("Init succeeded, want error")
				}
			})
			t.Run("tooManyImagesAcrossMessages", func(t *testing.T) {
				msgs := make(genai.Messages, 2)
				for i := range msgs {
					msgs[i].Requests = make([]genai.Request, 5)
					for j := range msgs[i].Requests {
						msgs[i].Requests[j].Doc = genai.Doc{Filename: "input.png", Src: strings.NewReader("png")}
					}
				}
				var got cerebras.ChatRequest
				err := got.Init(msgs, "gemma-4-31b")
				if err == nil {
					t.Fatal("Init succeeded, want error")
				}
			})
		})
	})
}
