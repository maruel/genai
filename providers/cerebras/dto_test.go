// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Cerebras provider DTOs.

package cerebras_test

import (
	"encoding/json"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/cerebras"
)

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

func TestDTOFieldsHaveJSONTag(t *testing.T) {
	seen := map[reflect.Type]struct{}{}
	for _, typ := range []reflect.Type{
		reflect.TypeFor[cerebras.ToolChoice](),
		reflect.TypeFor[cerebras.PredictionContent](),
		reflect.TypeFor[cerebras.Prediction](),
		reflect.TypeFor[cerebras.ChatRequest](),
		reflect.TypeFor[cerebras.Message](),
		reflect.TypeFor[cerebras.Content](),
		reflect.TypeFor[cerebras.Tool](),
		reflect.TypeFor[cerebras.ToolCall](),
		reflect.TypeFor[cerebras.ChatResponse](),
		reflect.TypeFor[cerebras.ChatStreamChunkResponse](),
		reflect.TypeFor[cerebras.Logprobs](),
		reflect.TypeFor[cerebras.Usage](),
		reflect.TypeFor[cerebras.Model](),
		reflect.TypeFor[cerebras.ModelsResponse](),
		reflect.TypeFor[cerebras.ErrorResponse](),
	} {
		assertDTOFieldsHaveJSONTag(t, typ, seen)
	}
}

func assertDTOFieldsHaveJSONTag(t *testing.T, typ reflect.Type, seen map[reflect.Type]struct{}) {
	if _, ok := seen[typ]; ok {
		return
	}
	seen[typ] = struct{}{}
	for i := range typ.NumField() {
		f := typ.Field(i)
		if tag, ok := f.Tag.Lookup("json"); !ok || tag == "-" {
			t.Errorf("%s.%s has no JSON field tag", typ, f.Name)
		}
		if f.Type.Kind() == reflect.Struct && f.Type.Name() == "" {
			assertDTOFieldsHaveJSONTag(t, f.Type, seen)
		}
	}
}

func TestReasoningFormat(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, format := range []cerebras.ReasoningFormat{
				"",
				cerebras.ReasoningFormatParsed,
				cerebras.ReasoningFormatTextParsed,
				cerebras.ReasoningFormatRaw,
				cerebras.ReasoningFormatHidden,
				cerebras.ReasoningFormatNone,
			} {
				if err := format.Validate(); err != nil {
					t.Errorf("Validate(%q): %v", format, err)
				}
			}
		})
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
		t.Run("valid", func(t *testing.T) {
			for _, tier := range []cerebras.ServiceTier{
				"",
				cerebras.ServiceTierAuto,
				cerebras.ServiceTierDefault,
				cerebras.ServiceTierFlex,
				cerebras.ServiceTierPriority,
			} {
				if err := tier.Validate(); err != nil {
					t.Errorf("Validate(%q): %v", tier, err)
				}
			}
		})
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
		t.Run("valid", func(t *testing.T) {
			data, err := json.Marshal(cerebras.ToolChoice{Function: "lookup"})
			if err != nil {
				t.Fatal(err)
			}
			const want = `{"type":"function","function":{"name":"lookup"}}`
			if string(data) != want {
				t.Errorf("JSON = %s, want %s", data, want)
			}
		})
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
		t.Run("valid", func(t *testing.T) {
			for _, content := range []cerebras.PredictionContent{
				{Text: "known"},
				{Type: cerebras.ContentText, Text: "known"},
			} {
				if err := content.Validate(); err != nil {
					t.Errorf("Validate() = %v", err)
				}
			}
		})
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
			_, err := cerebras.New(t.Context(), cerebras.ProviderOptionQueueThreshold(49*time.Millisecond))
			if err == nil {
				t.Fatal("New() succeeded, want error")
			}
		})
		t.Run("genOptionError", func(t *testing.T) {
			var request cerebras.ChatRequest
			err := request.Init(genai.Messages{{Requests: []genai.Request{{Text: "hello"}}}}, "gemma-4-31b", &cerebras.GenOption{
				QueueThreshold: 49 * time.Millisecond,
			})
			if err == nil {
				t.Fatal("Init() succeeded, want error")
			}
		})
	})
}

func TestReasoningEffort(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, effort := range []cerebras.ReasoningEffort{
				"",
				cerebras.ReasoningEffortNone,
				cerebras.ReasoningEffortLow,
				cerebras.ReasoningEffortMedium,
				cerebras.ReasoningEffortHigh,
			} {
				if err := effort.Validate(); err != nil {
					t.Errorf("Validate(%q): %v", effort, err)
				}
			}
		})
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
			t.Run("multipleRequests", func(t *testing.T) {
				var got cerebras.ChatRequest
				err := got.Init(genai.Messages{{
					Requests: []genai.Request{
						{Text: "What is in this image?"},
						{Doc: genai.Doc{Filename: "input.png", Src: strings.NewReader("png")}},
					},
				}}, "gemma-4-31b")
				if err != nil {
					t.Fatal(err)
				}
				if len(got.Messages) != 1 {
					t.Fatalf("len(Messages) = %d, want 1", len(got.Messages))
				}
				data, err := json.Marshal(got.Messages[0])
				if err != nil {
					t.Fatal(err)
				}
				const want = `{"role":"user","content":[{"type":"text","text":"What is in this image?"},{"type":"image_url","image_url":{"url":"data:image/png;base64,cG5n"}}]}`
				if string(data) != want {
					t.Errorf("Message = %s, want %s", data, want)
				}
			})
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
			t.Run("text", func(t *testing.T) {
				var got cerebras.ChatRequest
				err := got.Init(genai.Messages{{Requests: []genai.Request{{Text: "hello"}}}}, "gemma-4-31b")
				if err != nil {
					t.Fatal(err)
				}
				data, err := json.Marshal(got.Messages[0])
				if err != nil {
					t.Fatal(err)
				}
				const want = `{"role":"user","content":[{"type":"text","text":"hello"}]}`
				if string(data) != want {
					t.Errorf("Message = %s, want %s", data, want)
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
			t.Run("remoteImage", func(t *testing.T) {
				var got cerebras.ChatRequest
				err := got.Init(genai.Messages{{Requests: []genai.Request{{
					Doc: genai.Doc{Filename: "input.png", URL: "https://example.com/input.png"},
				}}}}, "gemma-4-31b")
				if err == nil {
					t.Fatal("Init succeeded, want error")
				}
				const want = "message #0: request #0: cerebras requires image documents to be provided inline, not as a URL"
				if err.Error() != want {
					t.Errorf("Init error = %q, want %q", err, want)
				}
			})
			t.Run("unsupportedImage", func(t *testing.T) {
				var got cerebras.ChatRequest
				err := got.Init(genai.Messages{{Requests: []genai.Request{{
					Doc: genai.Doc{Filename: "input.gif", Src: strings.NewReader("gif")},
				}}}}, "gemma-4-31b")
				if err == nil {
					t.Fatal("Init succeeded, want error")
				}
				const want = "message #0: request #0: unsupported image MIME type \"image/gif\"; Cerebras supports image/jpeg and image/png"
				if err.Error() != want {
					t.Errorf("Init error = %q, want %q", err, want)
				}
			})
			t.Run("tooManyImages", func(t *testing.T) {
				reqs := make([]genai.Request, 6)
				for i := range reqs {
					reqs[i].Doc = genai.Doc{Filename: "input.png", Src: strings.NewReader("png")}
				}
				var got cerebras.ChatRequest
				err := got.Init(genai.Messages{{Requests: reqs}}, "gemma-4-31b")
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
