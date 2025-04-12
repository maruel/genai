// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cerebras_test

import (
	"fmt"
	"os"
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
	ctx := t.Context()
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
	chunks := make(chan genai.MessageFragment)
	end := make(chan genai.Message, 10)
	go func() {
		var pendingMsgs genai.Messages
		defer func() {
			for _, m := range pendingMsgs {
				end <- m
			}
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
				var err error
				if pendingMsgs, err = pkt.Accumulate(pendingMsgs); err != nil {
					end <- genai.NewTextMessage(genai.Assistant, fmt.Sprintf("Error: %v", err))
					return
				}
			}
		}
	}()
	err := c.ChatStream(ctx, msgs, &opts, chunks)
	close(chunks)
	var responses genai.Messages
	for m := range end {
		responses = append(responses, m)
	}
	t.Logf("Raw responses: %#v", responses)
	if err != nil {
		t.Fatal(err)
	}
	if len(responses) != 1 {
		t.Fatal("Unexpected responses")
	}
	resp := responses[0]
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "best_country" || resp.ToolCalls[0].ID == "" {
		t.Fatal("Unexpected response")
	}
	if err := resp.ToolCalls[0].Decode(&got); err != nil {
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

func init() {
	internal.BeLenient = false
}
