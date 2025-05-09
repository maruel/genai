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

func TestClient_Chat_allModels(t *testing.T) {
	internaltest.TestChatAllModels(
		t,
		func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		func(m genai.Model) bool {
			id := m.GetID()
			// Only test a few models because there are too many.
			return id == "@cf/qwen/qwen2.5-coder-32b-instruct" || id == "@cf/meta/llama-4-scout-17b-16e-instruct"
		})
}

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
	internaltest.TestChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "@hf/nousresearch/hermes-2-pro-mistral-7b") })
}

func TestClient_ChatStream(t *testing.T) {
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	responses := internaltest.ChatStream(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "@cf/meta/llama-3.2-3b-instruct") }, msgs, &opts)
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
	testRecorder.Signal(t)
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if accountID == "" {
		t.Skip("CLOUDFLARE_ACCOUNT_ID not set")
	}
	if os.Getenv("CLOUDFLARE_API_KEY") == "" {
		t.Skip("CLOUDFLARE_API_KEY not set")
	}
	t.Parallel()
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
	c.Client.Client.Transport = testRecorder.Record(t, c.Client.Client.Transport, recorder.WithHook(fnSave, recorder.AfterCaptureHook), recorder.WithMatcher(fnMatch))
	return c
}

var defaultMatcher = cassette.NewDefaultMatcher()

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
