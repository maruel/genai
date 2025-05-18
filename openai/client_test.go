// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openai_test

import (
	_ "embed"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/openai"
)

var testCases = &internaltest.TestCases{
	Default: internaltest.Settings{
		GetClient: func(t *testing.T, m string) genai.ChatProvider { return getClient(t, m) },
		// https://platform.openai.com/docs/models/gpt-4.1-nano
		Model: "gpt-4.1-nano",
	},
}

func TestClient_Chat_allModels(t *testing.T) {
	testCases.TestChatAllModels(
		t,
		func(m genai.Model) bool {
			id := m.GetID()
			// There's no way to know what model has which capability.
			if id == "babbage-002" ||
				strings.HasPrefix(id, "dall-") ||
				strings.HasPrefix(id, "davinci-") ||
				strings.HasPrefix(id, "gpt-3.5-") ||
				strings.HasPrefix(id, "o1-pro") ||
				strings.HasPrefix(id, "omni-moderation-") ||
				strings.HasPrefix(id, "text-embedding-") ||
				strings.HasPrefix(id, "tts-") ||
				strings.HasPrefix(id, "whisper-") ||
				strings.Contains(id, "-audio-") ||
				strings.Contains(id, "-image-") ||
				strings.Contains(id, "-realtime-") ||
				strings.HasSuffix(id, "-transcribe") ||
				strings.HasSuffix(id, "-tts") {
				return false
			}
			return true
		})
}

func TestClient_Chat_thinking(t *testing.T) {
	// https://platform.openai.com/docs/guides/reasoning
	testCases.TestChatThinking(t,
		&internaltest.Settings{
			Model: "o4-mini",
			Options: func(opts *genai.ChatOptions) genai.Validatable {
				return &openai.ChatOptions{
					ChatOptions: *opts,
					// This will lead to spurious HTTP 500 but it is 25% of the cost.
					ServiceTier:     openai.ServiceTierFlex,
					ReasoningEffort: openai.ReasoningEffortLow,
				}
			},
		})
}

func TestClient_ChatStream(t *testing.T) {
	testCases.TestChatStream(t, nil)
}

func TestClient_Chat_jSON(t *testing.T) {
	testCases.TestChatJSON(t, nil)
}

func TestClient_Chat_jSON_schema(t *testing.T) {
	testCases.TestChatJSONSchema(t, nil)
}

func TestClient_Chat_audio_mp3_inline(t *testing.T) {
	testCases.TestChatAudioMP3Inline(t, &internaltest.Settings{Model: "gpt-4o-audio-preview"})
}

func TestClient_Chat_vision_jPG_inline(t *testing.T) {
	testCases.TestChatVisionJPGInline(t, nil)
}

func TestClient_Chat_vision_pDF_inline(t *testing.T) {
	// TODO: Implement URL support.
	testCases.TestChatVisionPDFInline(t, nil)
}

func TestClient_Chat_tool_use_reply(t *testing.T) {
	testCases.TestChatToolUseReply(t, nil)
}

func TestClient_Chat_tool_use_position_bias(t *testing.T) {
	testCases.TestChatToolUsePositionBias(t, nil, false)
}

func getClient(t *testing.T, m string) *openai.Client {
	testRecorder.Signal(t)
	t.Parallel()
	apiKey := ""
	if os.Getenv("OPENAI_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := openai.New(apiKey, m)
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

func TestUnsupportedContinuableError(t *testing.T) {
	// Create a request with an unsupported feature (TopK)
	req := &openai.ChatRequest{}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Hello"),
	}
	opts := &genai.ChatOptions{
		TopK: 50, // OpenAI doesn't support TopK
	}

	// Initialize the request
	err := req.Init(msgs, opts, "gpt-4")

	// Check that it returns an UnsupportedContinuableError
	uce, ok := err.(*genai.UnsupportedContinuableError)
	if !ok {
		t.Fatalf("Expected UnsupportedContinuableError, got %T: %v", err, err)
	}

	// Check that the unsupported field is reported
	if len(uce.Unsupported) != 1 || uce.Unsupported[0] != "TopK" {
		t.Errorf("Expected Unsupported=[TopK], got %v", uce.Unsupported)
	}
}
