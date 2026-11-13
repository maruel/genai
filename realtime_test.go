// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for portable realtime options and input validation.

package genai_test

import (
	"bytes"
	"testing"

	"github.com/maruel/genai"
)

func TestRealtimeTurnDetection(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, v := range []genai.RealtimeTurnDetection{"", genai.RealtimeTurnManual, genai.RealtimeTurnServer} {
				if err := v.Validate(); err != nil {
					t.Fatal(err)
				}
			}
		})
		t.Run("error", func(t *testing.T) {
			if err := genai.RealtimeTurnDetection("unknown").Validate(); err == nil {
				t.Fatal("unknown turn detection accepted")
			}
		})
	})
}

func TestRealtimeOptionAudio(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, v := range []genai.RealtimeOptionAudio{{}, {InputFormat: genai.AudioFormat{SampleRate: 48000}}} {
				if err := v.Validate(); err != nil {
					t.Fatal(err)
				}
			}
		})
		t.Run("error", func(t *testing.T) {
			v := genai.RealtimeOptionAudio{InputFormat: genai.AudioFormat{SampleRate: -1}}
			if err := v.Validate(); err == nil {
				t.Fatal("negative input sample rate accepted")
			}
		})
	})
}

func TestRealtimeOptionTools(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		tool := genai.ToolDef{Name: "lookup", Description: "Lookup", InputSchemaOverride: genai.JSONSchema(`{"type":"object"}`)}
		opt := func(tools ...genai.ToolDef) *genai.RealtimeOptionTools {
			return &genai.RealtimeOptionTools{Tools: tools}
		}
		t.Run("valid", func(t *testing.T) {
			for _, v := range []*genai.RealtimeOptionTools{opt(), opt(tool)} {
				if err := v.Validate(); err != nil {
					t.Fatal(err)
				}
			}
		})
		t.Run("error", func(t *testing.T) {
			for _, v := range []*genai.RealtimeOptionTools{
				opt(genai.ToolDef{Name: "invalid.name"}),
				opt(tool, tool),
				opt(genai.ToolDef{Name: "missing_schema"}),
				opt(genai.ToolDef{Name: "array_schema", InputSchemaOverride: genai.JSONSchema(`[]`)}),
			} {
				if err := v.Validate(); err == nil {
					t.Fatal("invalid tools accepted")
				}
			}
		})
	})
}

func TestRealtimeContent(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for name, c := range map[string]genai.RealtimeContent{
				"empty":      {},
				"audio":      {Audio: []byte{0, 0}},
				"text":       {Text: "hello"},
				"final text": {Text: "hello", Final: true},
				"tool calls": {ToolCalls: []genai.ToolCall{{ID: "call_1", Name: "clock", Arguments: "{}"}}},
			} {
				t.Run(name, func(t *testing.T) {
					if err := c.Validate(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			for name, c := range map[string]genai.RealtimeContent{
				"audio and text":        {Audio: []byte{0, 0}, Text: "hello"},
				"audio and tools":       {Audio: []byte{0, 0}, ToolCalls: []genai.ToolCall{{ID: "call_1", Name: "clock", Arguments: "{}"}}},
				"text and tools":        {Text: "hello", ToolCalls: []genai.ToolCall{{ID: "call_1", Name: "clock", Arguments: "{}"}}},
				"final without text":    {Final: true},
				"final with audio":      {Audio: []byte{0, 0}, Final: true},
				"final with tool calls": {ToolCalls: []genai.ToolCall{{ID: "call_1", Name: "clock", Arguments: "{}"}}, Final: true},
			} {
				t.Run(name, func(t *testing.T) {
					if err := c.Validate(); err == nil {
						t.Fatal("expected invalid content")
					}
				})
			}
		})
	})
}

func TestRealtimeInput(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		img := func(filename string) genai.Doc {
			return genai.Doc{Src: bytes.NewReader([]byte{0xff, 0xd8}), Filename: filename}
		}
		t.Run("valid", func(t *testing.T) {
			for name, in := range map[string]genai.RealtimeInput{
				"audio":                 genai.RealtimeInputContent{Audio: []byte{0, 0}},
				"audio generate":        genai.RealtimeInputContent{Audio: []byte{0, 0}, Generate: true},
				"commit audio":          genai.RealtimeInputContent{Audio: []byte{0, 0}, Commit: true},
				"commit generate":       genai.RealtimeInputContent{Audio: []byte{0, 0}, Commit: true, Generate: true},
				"empty commit":          genai.RealtimeInputContent{Commit: true},
				"empty commit generate": genai.RealtimeInputContent{Commit: true, Generate: true},
				"context":               genai.RealtimeInputContent{Text: "Current room: kitchen."},
				"text":                  genai.RealtimeInputContent{Text: "What time is it?", Generate: true},
				"generate":              genai.RealtimeInputContent{Generate: true},
				"doc":                   genai.RealtimeInputContent{Doc: img("image.jpg"), Generate: true},
				"doc text":              genai.RealtimeInputContent{Doc: img("image.jpg"), Text: "What fruit is it?", Generate: true},
				"doc url":               genai.RealtimeInputContent{Doc: genai.Doc{URL: "https://example.com/report.pdf"}, Generate: true},
				"doc pdf":               genai.RealtimeInputContent{Doc: genai.Doc{Src: bytes.NewReader([]byte{0x25, 0x50}), Filename: "report.pdf"}, Generate: true},
				"tool results":          genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "call_1", Name: "clock", Result: `{"time":"12:00"}`}}, Generate: true},
				"cancel":                genai.RealtimeCancel{},
				"cancel response":       genai.RealtimeCancel{ResponseID: "response_1"},
				"truncate":              genai.RealtimeTruncate{ItemID: "item_1", ContentIndex: 1, AudioEndMS: 1200},
				"update":                genai.RealtimeUpdate{Instructions: new("be terse")},
			} {
				t.Run(name, func(t *testing.T) {
					if err := in.Validate(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			for name, in := range map[string]genai.RealtimeInput{
				"empty content":            genai.RealtimeInputContent{},
				"empty audio":              genai.RealtimeInputContent{Audio: []byte{}},
				"empty audio generate":     genai.RealtimeInputContent{Audio: []byte{}, Generate: true},
				"partial sample":           genai.RealtimeInputContent{Audio: []byte{1}},
				"mixed audio text":         genai.RealtimeInputContent{Audio: []byte{0, 0}, Text: "x"},
				"mixed audio doc":          genai.RealtimeInputContent{Audio: []byte{0, 0}, Doc: img("image.jpg")},
				"mixed doc tools":          genai.RealtimeInputContent{Doc: img("image.jpg"), ToolResults: []genai.ToolCallResult{{ID: "call_1", Name: "clock", Result: "ok"}}},
				"doc without source":       genai.RealtimeInputContent{Doc: genai.Doc{Filename: "image.jpg"}, Generate: true},
				"doc with both sources":    genai.RealtimeInputContent{Doc: genai.Doc{Filename: "image.jpg", Src: bytes.NewReader([]byte{0xff}), URL: "https://example.com/image.jpg"}, Generate: true},
				"doc without filename":     genai.RealtimeInputContent{Doc: genai.Doc{Src: bytes.NewReader([]byte{0xff, 0xd8})}},
				"empty tool results":       genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{}},
				"missing tool name":        genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "call_1", Result: "ok"}}},
				"missing tool id":          genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{Name: "clock", Result: "ok"}}},
				"missing tool result":      genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "call_1", Name: "clock"}}},
				"duplicate result":         genai.RealtimeInputContent{ToolResults: []genai.ToolCallResult{{ID: "call_1", Name: "clock", Result: "ok"}, {ID: "call_1", Name: "clock", Result: "ok"}}},
				"negative content index":   genai.RealtimeTruncate{ItemID: "item_1", ContentIndex: -1},
				"negative played position": genai.RealtimeTruncate{ItemID: "item_1", AudioEndMS: -1},
				"missing item id":          genai.RealtimeTruncate{},
				"empty update":             genai.RealtimeUpdate{},
			} {
				t.Run(name, func(t *testing.T) {
					if err := in.Validate(); err == nil {
						t.Fatal("expected invalid input")
					}
				})
			}
		})
	})
}

func TestRealtimeUpdate(t *testing.T) {
	str := func(v string) *string { return &v }
	tools := func(v []genai.ToolDef) *[]genai.ToolDef { return &v }
	choice := func(v genai.ToolCallRequest) *genai.ToolCallRequest { return &v }
	tokens := func(v int64) *int64 { return &v }
	mods := func(v ...genai.Modality) *genai.Modalities { m := genai.Modalities(v); return &m }
	turn := func(v genai.RealtimeTurnDetection) *genai.RealtimeTurnDetection { return &v }
	tool := genai.ToolDef{Name: "lookup", Description: "Lookup", InputSchemaOverride: genai.JSONSchema(`{"type":"object"}`)}
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for name, u := range map[string]genai.RealtimeUpdate{
				"instructions":    {Instructions: str("be terse")},
				"clear tools":     {Tools: tools(nil)},
				"tools":           {Tools: tools([]genai.ToolDef{tool})},
				"tool choice":     {ToolChoice: choice(genai.ToolCallNone)},
				"max tokens":      {MaxOutputTokens: tokens(128)},
				"output modality": {OutputModalities: mods(genai.ModalityText)},
				"all fields":      {Instructions: str(""), Tools: tools([]genai.ToolDef{}), ToolChoice: choice(genai.ToolCallAny), MaxOutputTokens: tokens(1), OutputModalities: mods(genai.ModalityAudio)},
				"both modalities": {OutputModalities: mods(genai.ModalityAudio, genai.ModalityText)},
				"turn detection":  {TurnDetection: turn(genai.RealtimeTurnManual)},
				"voice":           {Voice: str("marin")},
			} {
				t.Run(name, func(t *testing.T) {
					if err := u.Validate(); err != nil {
						t.Fatal(err)
					}
				})
			}
		})
		t.Run("error", func(t *testing.T) {
			for name, u := range map[string]genai.RealtimeUpdate{
				"empty":              {},
				"bad tool choice":    {ToolChoice: choice(genai.ToolCallRequest(99))},
				"zero tokens":        {MaxOutputTokens: tokens(0)},
				"empty modalities":   {OutputModalities: mods()},
				"duplicate modality": {OutputModalities: mods(genai.ModalityText, genai.ModalityText)},
				"image modality":     {OutputModalities: mods(genai.ModalityImage)},
				"empty turn":         {TurnDetection: turn("")},
			} {
				t.Run(name, func(t *testing.T) {
					if err := u.Validate(); err == nil {
						t.Fatal("invalid update accepted")
					}
				})
			}
		})
	})
}
