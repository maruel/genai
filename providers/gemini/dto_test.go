// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Gemini provider json schema.

package gemini

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/google/go-cmp/cmp"

	"github.com/maruel/genai"
)

func TestSchema(t *testing.T) {
	t.Run("FromJSONSchema", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			data := []struct {
				name string
				in   string
				want Schema
			}{
				{
					name: "nullable via anyOf with null second",
					in:   `{"anyOf":[{"type":"string"},{"type":"null"}]}`,
					want: Schema{Type: TypeString, Nullable: true},
				},
				{
					name: "nullable via anyOf with null first",
					in:   `{"anyOf":[{"type":"null"},{"type":"integer"}]}`,
					want: Schema{Type: TypeInteger, Nullable: true},
				},
				{
					name: "nullable via type array null first",
					in:   `{"type":["null","integer"]}`,
					want: Schema{Type: TypeInteger, Nullable: true},
				},
				{
					name: "anyOf non-nullable union",
					in:   `{"anyOf":[{"type":"string"},{"type":"integer"}]}`,
					want: Schema{
						AnyOf: []*Schema{
							{Type: TypeString},
							{Type: TypeInteger},
						},
					},
				},
				{
					name: "integer enum large value",
					in:   `{"type":"integer","enum":[9007199254740993]}`,
					want: Schema{Type: TypeInteger, Enum: []string{"9007199254740993"}},
				},
				{
					name: "no type field",
					in:   `{"description":"schemaless"}`,
					want: Schema{Description: "schemaless"},
				},
			}
			for _, line := range data {
				t.Run(line.name, func(t *testing.T) {
					s := Schema{}
					if err := s.FromJSONSchema(genai.JSONSchema(line.in)); err != nil {
						t.Fatal(err)
					}
					if diff := cmp.Diff(line.want, s); diff != "" {
						t.Errorf("Schema mismatch (-want +got):\n%s", diff)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			data := []struct {
				name string
				in   string
				want string
			}{
				{
					name: "invalid json",
					in:   `not json`,
					want: "invalid JSON schema: invalid character 'o' in literal null (expecting 'u')",
				},
				{
					name: "unsupported type in property",
					in:   `{"type":"object","properties":{"a":{"type":"bad"}}}`,
					want: `property "a": unsupported JSON Schema type: "bad"`,
				},
				{
					name: "unsupported type in items",
					in:   `{"type":"array","items":{"type":"bad"}}`,
					want: `items: unsupported JSON Schema type: "bad"`,
				},
				{
					name: "unsupported type in anyOf",
					in:   `{"anyOf":[{"type":"string"},{"type":"bad"}]}`,
					want: `anyOf[1]: unsupported JSON Schema type: "bad"`,
				},
				{
					name: "boolean enum value",
					in:   `{"type":"string","enum":["A",true]}`,
					want: `enum[1]: unsupported type bool, must be string or number`,
				},
			}
			for _, line := range data {
				t.Run(line.name, func(t *testing.T) {
					s := Schema{}
					if err := s.FromJSONSchema(genai.JSONSchema(line.in)); err == nil {
						t.Fatal("expected error")
					} else if got := err.Error(); got != line.want {
						t.Errorf("got error %q, want %q", got, line.want)
					}
				})
			}
		})
	})
}

func TestChatRequest(t *testing.T) {
	t.Run("Init", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			t.Run("options", func(t *testing.T) {
				var in ChatRequest
				if err := in.Init(genai.Messages{genai.NewTextMessage("hello")}, "gemini-3.8-flash", &genai.GenOptionText{Temperature: -1}); err == nil {
					t.Fatal("accepted invalid options")
				}
			})
		})
	})
}

func TestSignatureOnlyStreamReplay(t *testing.T) {
	// A text chunk followed by an empty-text signature chunk is returned by
	// Gemini's stream API. The signature must stay a separate ordered part.
	raw := []string{
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"Task 7 is waiting."}]}}]}`,
		`{"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"c2lnbmF0dXJl"}]}}]}`,
	}
	chunks := make([]ChatStreamChunkResponse, len(raw))
	for i, s := range raw {
		if err := json.Unmarshal([]byte(s), &chunks[i]); err != nil {
			t.Fatal(err)
		}
	}
	fragments, finish := ProcessStream(func(yield func(ChatStreamChunkResponse) bool) {
		for _, c := range chunks {
			if !yield(c) {
				return
			}
		}
	})
	m := genai.Message{}
	for f := range fragments {
		if err := m.Accumulate(&f); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := finish(); err != nil {
		t.Fatal(err)
	}
	before, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if len(m.Replies) != 2 || m.Replies[0].Text != "Task 7 is waiting." || len(m.Replies[0].Opaque) != 0 || m.Replies[1].Text != "" {
		t.Fatalf("assembled replies lost part boundaries: %#v", m)
	}
	req := ChatRequest{}
	if err := req.Init(genai.Messages{genai.NewTextMessage("Check task 7."), m}, "gemini-flash-lite-latest"); err != nil {
		t.Fatal(err)
	}
	want := []Part{{Text: "Task 7 is waiting."}, {ThoughtSignature: []byte("signature")}}
	if diff := cmp.Diff(want, req.Contents[1].Parts); diff != "" {
		t.Fatal(diff)
	}
	after, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(before, after) {
		t.Fatal("request mutated returned message")
	}
	syncMsg := genai.Message{}
	if err := chunks[1].Candidates[0].Content.To(&syncMsg); err != nil {
		t.Fatal(err)
	}
	if diff := cmp.Diff(m.Replies[1:], syncMsg.Replies); diff != "" {
		t.Fatal(diff)
	}
}

func TestPart(t *testing.T) {
	t.Run("FromReply", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("thinking_signature", func(t *testing.T) {
				var got Part
				if err := got.FromReply(&genai.Reply{Reasoning: "thinking", Opaque: map[string]any{"signature": []byte("signed")}}); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(Part{Text: "thinking", Thought: true, ThoughtSignature: []byte("signed")}, got); diff != "" {
					t.Fatal(diff)
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				opaque map[string]any
			}{
				{"extra", map[string]any{"signature": []byte("signature"), "extra": true}},
				{"absent", nil},
			} {
				t.Run(tc.name, func(t *testing.T) {
					var p Part
					if err := p.FromReply(&genai.Reply{Opaque: tc.opaque}); err == nil {
						t.Fatal("accepted invalid metadata-only reply")
					}
				})
			}
		})
	})
}

func TestEmbedContentConfig(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, tc := range []embeddingConfigCase{
				{"invalid task", EmbedContentConfig{TaskType: "bad"}},
				{"title without retrieval document", EmbedContentConfig{Title: "Cats"}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					if err := tc.in.Validate(); err == nil {
						t.Fatal("expected error")
					}
				})
			}
		})
	})
}

func TestEmbeddingRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			valid := EmbeddingRequest{Model: "models/gemini-embedding-2", Content: Content{Parts: []Part{{Text: "hello"}}}}
			for _, tc := range []embeddingRequestErrorCase{
				{"missing parts", func(r *EmbeddingRequest) { r.Content.Parts = nil }},
				{"invalid config", func(r *EmbeddingRequest) { r.EmbedContentConfig.OutputDimensionality = -1 }},
			} {
				t.Run(tc.name, func(t *testing.T) {
					r := valid
					tc.modify(&r)
					if err := r.Validate(); err == nil {
						t.Fatal("expected error")
					}
				})
			}
		})
	})
}

func TestBatchEmbeddingRequest(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, tc := range []embeddingBatchErrorCase{
				{"invalid request", BatchEmbeddingRequest{Requests: []EmbeddingRequest{{}}}},
				{"mixed models", BatchEmbeddingRequest{Requests: []EmbeddingRequest{
					{Model: "models/gemini-embedding-2", Content: Content{Parts: []Part{{Text: "hello"}}}},
					{Model: "models/other-model", Content: Content{Parts: []Part{{Text: "world"}}}},
				}}},
			} {
				t.Run(tc.name, func(t *testing.T) {
					if err := tc.in.Validate(); err == nil {
						t.Fatal("expected error")
					}
				})
			}
		})
	})
}

type embeddingRequestErrorCase struct {
	name   string
	modify func(*EmbeddingRequest)
}

type embeddingBatchErrorCase struct {
	name string
	in   BatchEmbeddingRequest
}

// embeddingConfigCase exercises native configuration validation.
type embeddingConfigCase struct {
	name string
	in   EmbedContentConfig
}
