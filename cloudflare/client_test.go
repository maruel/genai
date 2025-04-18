// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

import (
	"log"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestClient_Chat_jSON(t *testing.T) {
	c := getClient(t, "@hf/nousresearch/hermes-2-pro-mistral-7b")
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
		DecodeAs:    &got,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 0 || resp.OutputTokens != 0 {
		t.Logf("Did cloudflare finally start filling the usage fields?")
	}
	want := genai.Message{Role: genai.Assistant, Contents: []genai.Content{{Text: `{"round":true}`}}}
	if diff := cmp.Diff(&want, &resp.Message); diff != "" {
		t.Fatalf("(+want), (-got):\n%s", diff)
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Round {
		t.Fatal("unexpected")
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	c := getClient(t, "@hf/nousresearch/hermes-2-pro-mistral-7b")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
	}
	var got struct {
		Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   200,
		Tools: []genai.ToolDef{
			{
				Name:        "best_country",
				Description: "A tool to determine the best country",
				InputsAs:    &got,
			},
		},
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	log.Printf("Raw response: %#v", resp)
	// Warning: when the model is undecided, it call both.
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "best_country" {
		log.Fatal("Unexpected response")
	}
	if err := resp.ToolCalls[0].Decode(&got); err != nil {
		log.Fatal(err)
	}
	if got.Country != "Canada" {
		t.Fatal(got.Country)
	}
}

func TestClient_ChatStream(t *testing.T) {
	c := getClient(t, "@cf/meta/llama-3.2-3b-instruct")
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	responses := internaltest.ChatStream(t, c, msgs, &opts)
	if len(responses) != 1 {
		log.Fatal("Unexpected response")
	}
	resp := responses[0]
	// Normalize some of the variance. Obviously many models will still fail this test.
	if got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"); got != "hello" {
		t.Fatal(got)
	}
}

func getClient(t *testing.T, m string) *cloudflare.Client {
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if accountID == "" {
		t.Skip("CLOUDFLARE_ACCOUNT_ID not set")
	}
	if os.Getenv("CLOUDFLARE_API_KEY") == "" {
		t.Skip("CLOUDFLARE_API_KEY not set")
	}
	c, err := cloudflare.New("", "", m)
	if err != nil {
		t.Fatal(err)
	}
	fnMatch := func(r *http.Request, i cassette.Request) bool {
		r = r.Clone(r.Context())
		r.URL.Path = strings.Replace(r.URL.Path, accountID, "ACCOUNT_ID", 1)
		return defaultMatcher(r, i)
	}
	fnSave := func(i *cassette.Interaction) error {
		i.Request.URL = strings.Replace(i.Request.URL, accountID, "ACCOUNT_ID", 1)
		return nil
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport, recorder.WithHook(fnSave, recorder.AfterCaptureHook), recorder.WithMatcher(fnMatch))
	return c
}

var defaultMatcher = cassette.NewDefaultMatcher()

func init() {
	internal.BeLenient = false
}
