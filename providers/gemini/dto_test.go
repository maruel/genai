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
	"github.com/maruel/genai/base"
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
					name: "string",
					in:   `{"type":"string"}`,
					want: Schema{Type: TypeString},
				},
				{
					name: "integer",
					in:   `{"type":"integer"}`,
					want: Schema{Type: TypeInteger},
				},
				{
					name: "number",
					in:   `{"type":"number"}`,
					want: Schema{Type: TypeNumber},
				},
				{
					name: "boolean",
					in:   `{"type":"boolean"}`,
					want: Schema{Type: TypeBoolean},
				},
				{
					name: "array of strings",
					in:   `{"type":"array","items":{"type":"string"}}`,
					want: Schema{Type: TypeArray, Items: &Schema{Type: TypeString}},
				},
				{
					name: "object with properties and required",
					in:   `{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"integer"}},"required":["a"]}`,
					want: Schema{
						Type: TypeObject,
						Properties: map[string]Schema{
							"a": {Type: TypeString},
							"b": {Type: TypeInteger},
						},
						Required: []string{"a"},
					},
				},
				{
					name: "empty object",
					in:   `{"type":"object"}`,
					want: Schema{Type: TypeObject},
				},
				{
					name: "nested object",
					in:   `{"type":"object","properties":{"o":{"type":"object","properties":{"p":{"type":"string"}},"required":["p"]}},"required":["o"]}`,
					want: Schema{
						Type: TypeObject,
						Properties: map[string]Schema{
							"o": {
								Type: TypeObject,
								Properties: map[string]Schema{
									"p": {Type: TypeString},
								},
								Required: []string{"p"},
							},
						},
						Required: []string{"o"},
					},
				},
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
					name: "nullable via type array",
					in:   `{"type":["string","null"]}`,
					want: Schema{Type: TypeString, Nullable: true},
				},
				{
					name: "nullable via type array null first",
					in:   `{"type":["null","integer"]}`,
					want: Schema{Type: TypeInteger, Nullable: true},
				},
				{
					name: "nullable via nullable field",
					in:   `{"type":"string","nullable":true}`,
					want: Schema{Type: TypeString, Nullable: true},
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
					name: "string enum",
					in:   `{"type":"string","enum":["A","B","C"]}`,
					want: Schema{Type: TypeString, Enum: []string{"A", "B", "C"}},
				},
				{
					name: "integer enum as numbers",
					in:   `{"type":"integer","enum":[101,201,301]}`,
					want: Schema{Type: TypeInteger, Enum: []string{"101", "201", "301"}},
				},
				{
					name: "integer enum large value",
					in:   `{"type":"integer","enum":[9007199254740993]}`,
					want: Schema{Type: TypeInteger, Enum: []string{"9007199254740993"}},
				},
				{
					name: "format date-time",
					in:   `{"type":"string","format":"date-time"}`,
					want: Schema{Type: TypeString, Format: FormatDateTime},
				},
				{
					name: "format int32",
					in:   `{"type":"integer","format":"int32"}`,
					want: Schema{Type: TypeInteger, Format: FormatInt32},
				},
				{
					name: "format int64",
					in:   `{"type":"integer","format":"int64"}`,
					want: Schema{Type: TypeInteger, Format: FormatInt64},
				},
				{
					name: "format float",
					in:   `{"type":"number","format":"float"}`,
					want: Schema{Type: TypeNumber, Format: FormatFloat},
				},
				{
					name: "format double",
					in:   `{"type":"number","format":"double"}`,
					want: Schema{Type: TypeNumber, Format: FormatDouble},
				},
				{
					name: "description and title",
					in:   `{"type":"string","description":"A description","title":"A title"}`,
					want: Schema{Type: TypeString, Description: "A description", Title: "A title"},
				},
				{
					name: "default int value",
					in:   `{"type":"integer","default":123}`,
					want: Schema{Type: TypeInteger, Default: json.RawMessage(`123`)},
				},
				{
					name: "default bool value",
					in:   `{"type":"boolean","default":true}`,
					want: Schema{Type: TypeBoolean, Default: json.RawMessage(`true`)},
				},
				{
					name: "example float value",
					in:   `{"type":"number","format":"double","example":67.89}`,
					want: Schema{Type: TypeNumber, Format: FormatDouble, Example: json.RawMessage(`67.89`)},
				},
				{
					name: "minLength and maxLength",
					in:   `{"type":"string","minLength":5,"maxLength":10}`,
					want: Schema{Type: TypeString, MinLength: 5, MaxLength: 10},
				},
				{
					name: "minItems and maxItems",
					in:   `{"type":"array","items":{"type":"string"},"minItems":2,"maxItems":5}`,
					want: Schema{Type: TypeArray, Items: &Schema{Type: TypeString}, MinItems: 2, MaxItems: 5},
				},
				{
					name: "minimum and maximum for integer",
					in:   `{"type":"integer","minimum":1,"maximum":100}`,
					want: Schema{Type: TypeInteger, Minimum: 1, Maximum: 100},
				},
				{
					name: "minimum and maximum for number",
					in:   `{"type":"number","minimum":0.5,"maximum":99.9}`,
					want: Schema{Type: TypeNumber, Minimum: 0.5, Maximum: 99.9},
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
					name: "unsupported type",
					in:   `{"type":"unknown"}`,
					want: `unsupported JSON Schema type: "unknown"`,
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

func TestContentThoughtSignature(t *testing.T) {
	t.Run("round trip", func(t *testing.T) {
		want := []byte("thought signature")
		in := Content{
			Parts: []Part{{Text: "hello", ThoughtSignature: want}},
		}
		m := genai.Message{}
		if err := in.To(&m); err != nil {
			t.Fatal(err)
		}
		if len(m.Replies) != 1 {
			t.Fatalf("got %d replies, want 1", len(m.Replies))
		}
		got, ok := m.Replies[0].Opaque["signature"].([]byte)
		if !ok {
			t.Fatalf("got Opaque %#v, want signature", m.Replies[0].Opaque)
		}
		if diff := cmp.Diff(want, got); diff != "" {
			t.Errorf("signature mismatch (-want +got):\n%s", diff)
		}

		out := Content{}
		if err := out.From(&m); err != nil {
			t.Fatal(err)
		}
		if diff := cmp.Diff(in.Parts, out.Parts); diff != "" {
			t.Errorf("parts mismatch (-want +got):\n%s", diff)
		}
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

func TestImageParametersDurationS(t *testing.T) {
	got, err := json.Marshal(ImageParameters{Duration: base.DurationS(8)})
	if err != nil {
		t.Fatal(err)
	}
	const want = `{"personGeneration":"","durationSeconds":8}`
	if string(got) != want {
		t.Errorf("MarshalJSON() = %s, want %s", got, want)
	}
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
			t.Run("tool_signature", func(t *testing.T) {
				var got Part
				if err := got.FromReply(&genai.Reply{ToolCall: genai.ToolCall{ID: "call_A", Name: "status", Arguments: `{"task":3}`, Opaque: map[string]any{"signature": []byte("signed")}}}); err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff([]byte("signed"), got.ThoughtSignature); diff != "" {
					t.Fatal(diff)
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			for _, tc := range []struct {
				name   string
				opaque map[string]any
			}{
				{"wrong_type", map[string]any{"signature": "signature"}},
				{"empty", map[string]any{"signature": []byte{}}},
				{"nil", map[string]any{"signature": []byte(nil)}},
				{"unknown", map[string]any{"unknown": []byte("signature")}},
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
