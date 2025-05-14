// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cohere_test

import (
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

var testCases = &internaltest.TestCases{
	GetClient:    func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
	DefaultModel: "command-r-08-2024",
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
			id := m.GetID()
			return strings.HasPrefix(id, "command-")
		})
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, "command-r7b-12-2024", true)
}

func TestClient_Chat_jSON(t *testing.T) {
	t.Skip("Cohere's model seem to struggle at unstructured JSON. To be investigated.")
	testCases.TestChatJSON(t, "", true)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, "", true)
}

func TestClient_Chat_tool_use(t *testing.T) {
	testCases.TestChatToolUseCountry(t, "", true)
}

func getClient(t *testing.T, m string) *cohere.Client {
	testRecorder.Signal(t)
	if os.Getenv("COHERE_API_KEY") == "" {
		t.Skip("COHERE_API_KEY not set")
	}
	t.Parallel()
	c, err := cohere.New("", m)
	if err != nil {
		t.Fatal(err)
	}
	c.Client.Client.Transport = testRecorder.Record(t, c.Client.Client.Transport)
	return c
}

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
