// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

import (
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

var testCases = &internaltest.TestCases{
	GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
	Default: internaltest.Settings{
		Model: "@hf/nousresearch/hermes-2-pro-mistral-7b",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
			id := m.GetID()
			// Only test a few models because there are too many.
			return id == "@cf/qwen/qwen2.5-coder-32b-instruct" || id == "@cf/meta/llama-4-scout-17b-16e-instruct"
		})
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, "@cf/meta/llama-3.2-3b-instruct", true)
}

func TestClient_Chat_jSON(t *testing.T) {
	testCases.TestChatJSON(t, "", false)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, "", false)
}

func TestClient_Chat_tool_use(t *testing.T) {
	testCases.TestChatToolUseCountry(t, "", false)
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
