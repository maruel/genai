// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini_test

import (
	"bytes"
	_ "embed"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func TestClient_Chat_vision_and_JSON(t *testing.T) {
	c := getClient(t, model)
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "Is it a banana? Reply as JSON."},
				{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
			},
		},
	}
	var got struct {
		Banana bool `json:"banana"`
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
	if resp.InputTokens != 268 || resp.OutputTokens != 9 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if err := resp.Contents[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if !got.Banana {
		t.Fatal("unexpected")
	}
}

func TestClient_Chat_pDF(t *testing.T) {
	c := getClient(t, model)
	f, err := os.Open("testdata/hidden_word.pdf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Reply with only the word."},
				{Document: f},
			},
		},
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 1301 || resp.OutputTokens != 2 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if got := strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)); got != "orange" {
		t.Fatal(got)
	}
}

func TestClient_Chat_audio(t *testing.T) {
	c := getClient(t, model)
	f, err := os.Open("testdata/mystery_word.opus")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word said? Reply with only the word."},
				{Document: f},
			},
		},
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 12 || resp.OutputTokens != 2 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	if got := strings.TrimRight(strings.ToLower(resp.Contents[0].Text), "."); got != "orange" {
		t.Fatal(got)
	}
}

func TestClient_Chat_tool_use(t *testing.T) {
	c := getClient(t, model)
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   200,
	}
	internaltest.ChatToolUseCountry(t, c, &opts)
}

func TestClient_Chat_tool_use_video(t *testing.T) {
	c := getClient(t, model)
	f, err := os.Open("testdata/animation.mp4")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	msgs := genai.Messages{
		{
			Role: genai.User,
			Contents: []genai.Content{
				{Text: "What is the word? Call the tool hidden_word to tell me what word you saw."},
				{Document: f},
			},
		},
	}
	var got struct {
		Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
		Tools: []genai.ToolDef{
			{
				Name:        "hidden_word",
				Description: "A tool to state what word was seen in the video.",
				InputsAs:    &got,
			},
		},
	}
	resp, err := c.Chat(t.Context(), msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("Raw response: %#v", resp)
	if resp.InputTokens != 1079 || resp.OutputTokens != 5 {
		t.Logf("Unexpected tokens usage: %v", resp.Usage)
	}
	// Warning: there's a bug where it returns two identical tool calls. To verify.
	if len(resp.ToolCalls) == 0 || resp.ToolCalls[0].Name != "hidden_word" {
		t.Fatal("Unexpected response")
	}

	if err := resp.ToolCalls[0].Decode(&got); err != nil {
		t.Fatal(err)
	}
	if saw := strings.ToLower(got.Word); saw != "banana" {
		t.Fatal(saw)
	}
}

func TestClient_ChatStream(t *testing.T) {
	c := getClient(t, model)
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
		t.Fatal("Unexpected responses")
	}
	resp := responses[0]
	if len(resp.Contents) != 1 {
		t.Fatal("Unexpected response")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	if got := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"); got != "hello" {
		t.Fatal(got)
	}
}

func getClient(t *testing.T, m string) *gemini.Client {
	if os.Getenv("GEMINI_API_KEY") == "" {
		t.Skip("GEMINI_API_KEY not set")
	}
	t.Parallel()
	c, err := gemini.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	fnMatch := func(r *http.Request, i cassette.Request) bool {
		r = r.Clone(r.Context())
		r.URL.RawQuery = ""
		return defaultMatcher(r, i)
	}
	fnSave := func(i *cassette.Interaction) error {
		j := strings.Index(i.Request.URL, "?")
		i.Request.URL = i.Request.URL[:j]
		return nil
	}
	c.Client.Client.Transport = internaltest.Record(t, c.Client.Client.Transport, recorder.WithHook(fnSave, recorder.AfterCaptureHook), recorder.WithMatcher(fnMatch))
	return c
}

var defaultMatcher = cassette.NewDefaultMatcher()

func init() {
	internal.BeLenient = false
}
