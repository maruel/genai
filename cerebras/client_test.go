// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras_test

import (
	"os"
	"strconv"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Chat_json(t *testing.T) {
	c := getClient(t, "llama-3.1-8b")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Is a circle round? Reply as JSON."),
	}
	var got struct {
		Round bool `json:"round"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		DecodeAs:    got,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 173 || resp.OutputTokens != 6 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	want := genai.Message{Role: genai.Assistant, Contents: []genai.Content{{Text: `{"round": true}`}}}
	if diff := cmp.Diff(&want, &resp.Message); diff != "" {
		t.Fatalf("(+want), (-got):\n%s", diff)
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Round {
		t.Fatal("expected round")
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	c := getClient(t, "llama-3.1-8b")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
	}
	var got struct {
		Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		Tools: []genai.ToolDef{
			{
				Name:        "best_country",
				Description: "A tool to determine the best country",
				InputsAs:    &got,
			},
		},
	}
	responses := chatStream(t, c, msgs, &opts)
	want := genai.Messages{
		genai.Message{
			Role: genai.Assistant,
			ToolCalls: []genai.ToolCall{
				{ID: "1", Name: "best_country", Arguments: `{"country": "Canada"}`},
				{ID: "2", Name: "best_country", Arguments: `{"country": "USA"}`},
			},
		},
	}
	assertResponses(t, want, responses)
	if err := responses[0].ToolCalls[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Country != "Canada" {
		t.Fatal(got.Country)
	}
}

func getClient(t *testing.T, m string) *cerebras.Client {
	if os.Getenv("CEREBRAS_API_KEY") == "" {
		t.Skip("CEREBRAS_API_KEY not set")
	}
	c, err := cerebras.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport)
	return c
}

func chatStream(t *testing.T, c *cerebras.Client, msgs genai.Messages, opts *genai.ChatOptions) genai.Messages {
	ctx := t.Context()
	chunks := make(chan genai.MessageFragment)
	end := make(chan genai.Messages, 1)
	go func() {
		var pendingMsgs genai.Messages
		defer func() {
			end <- pendingMsgs
			close(end)
		}()
		for {
			select {
			case <-ctx.Done():
				return
			case pkt, ok := <-chunks:
				if !ok {
					return
				}
				var err2 error
				if pendingMsgs, err2 = pkt.Accumulate(pendingMsgs); err2 != nil {
					t.Error(err2)
					return
				}
			}
		}
	}()
	err := c.ChatStream(ctx, msgs, opts, chunks)
	close(chunks)
	responses := <-end
	t.Logf("Raw responses: %#v", responses)
	if err != nil {
		t.Fatal(err)
	}
	return responses
}

func assertResponses(t *testing.T, want, got genai.Messages) {
	if len(got) != len(want) {
		t.Errorf("Expected %d responses, got %d", len(want), len(got))
	}
	for i := range got {
		for j := range got[i].ToolCalls {
			if got[i].ToolCalls[j].ID != "" {
				got[i].ToolCalls[j].ID = strconv.Itoa(i + j + 1)
			}
		}
	}
	for i := range want {
		if diff := cmp.Diff(&want[i], &got[i]); diff != "" {
			t.Errorf("(+want), (-got):\n%s", diff)
		}
	}
	if t.Failed() {
		t.FailNow()
	}
}

func init() {
	internal.BeLenient = false
}
