// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Chat_allModels(t *testing.T) {
	internaltest.TestChatAllModels(
		t,
		func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		func(m genai.Model) bool {
			model := m.(*huggingface.Model)
			if model.PipelineTag != "text-generation" {
				return false
			}
			id := model.ID
			return id == "meta-llama/Llama-3.3-70B-Instruct"
		})
}

func TestClient_ChatStream(t *testing.T) {
	// TODO: Figure out why smaller models fail.
	internaltest.TestChatStream(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "meta-llama/Llama-3.3-70B-Instruct") }, false)
}

func TestClient_Chat_jSON(t *testing.T) {
	t.Skip(`{"error":"Input validation error: grammar is not supported","error_type":"validation"}`)
	internaltest.TestChatJSON(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "meta-llama/Llama-3.3-70B-Instruct") }, true)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	internaltest.TestChatJSONSchema(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "meta-llama/Llama-3.3-70B-Instruct") }, true)
}

func TestClient_Chat_tool_use(t *testing.T) {
	// TODO: Figure out why smaller models fail.
	internaltest.TestChatToolUseCountry(t, func(t *testing.T) genai.ChatProvider { return getClient(t, "meta-llama/Llama-3.3-70B-Instruct") }, true)
}

func getClient(t *testing.T, m string) *huggingface.Client {
	testRecorder.Signal(t)
	if os.Getenv("HUGGINGFACE_API_KEY") == "" {
		// Fallback to loading from the python client's cache.
		h, err := os.UserHomeDir()
		if err != nil {
			t.Fatal("can't find home directory")
		}
		if _, err := os.Stat(filepath.Join(h, ".cache", "huggingface", "token")); err != nil {
			t.Skip("HUGGINGFACE_API_KEY not set and can't find ~/.cache/huggingface/token")
		}
	}
	t.Parallel()
	c, err := huggingface.New("", m)
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
