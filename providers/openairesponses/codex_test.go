// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests direct Codex authentication through the existing Responses provider.
package openairesponses_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openairesponses"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func testCodexAuth(t *testing.T, records *internaltest.Records) {
	if os.Getenv("RECORD") == "" {
		t.Setenv("CODEX_HOME", codexTestCredentials(t))
	}
	getClient := func(t *testing.T, opts ...genai.ProviderOption) *openairesponses.Client {
		opts = append(opts, &openairesponses.ProviderOption{Auth: openairesponses.AuthCodex, AppName: "gomode-test"}, genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
			recorded := records.Record(t, h, recorder.WithHook(sanitizeCodexRecording, recorder.AfterCaptureHook))
			return codexTestTransport(func(r *http.Request) (*http.Response, error) {
				if r.Header.Get("Originator") != "gomode-test" || r.Header.Get("User-Agent") != "gomode-test" {
					t.Error("missing application identity")
				}
				return recorded.RoundTrip(r)
			})
		}))
		c, err := openairesponses.New(t.Context(), opts...)
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		return c
	}
	t.Run("Ping", func(t *testing.T) {
		c := getClient(t)
		if err := c.Ping(t.Context()); err != nil {
			t.Fatal(err)
		}
		if c.ModelID() != "" {
			t.Fatal("model-less probe selected a model")
		}
	})
	t.Run("GenStream", func(t *testing.T) {
		c := getClient(t, genai.ProviderOptionModel("gpt-6-luna"))
		chunks, finish := c.GenStream(t.Context(), genai.Messages{genai.NewTextMessage("Reply exactly: luna-stream-ok")}, &genai.GenOptionText{SystemPrompt: "Answer briefly."}, genai.EffortLow)
		var text strings.Builder
		for p := range chunks {
			text.WriteString(p.Text)
		}
		res, err := finish()
		if err != nil {
			t.Fatal(err)
		}
		if text.String() != "luna-stream-ok" || res.String() != text.String() || res.Usage.InputTokens == 0 || res.Usage.OutputTokens == 0 {
			t.Fatalf("text %q result %+v", text.String(), res)
		}
	})
	t.Run("GenSync", func(t *testing.T) {
		c := getClient(t, genai.ProviderOptionModel("gpt-6-luna"))
		msgs := make(genai.Messages, 1, 5)
		msgs[0] = genai.NewTextMessage("Call report_marker with marker exactly luna-tools-ok. Do not answer directly.")
		tools := &genai.GenOptionTools{Force: genai.ToolCallRequired, Tools: []genai.ToolDef{{Name: "report_marker", Description: "Report the marker requested by the user.", Callback: func(context.Context, *markerInput) (string, error) { return "marker-confirmed", nil }}}}
		opts := &genai.GenOptionText{SystemPrompt: "Follow the user's instructions. After a tool result, reply exactly with its contents."}
		res, err := c.GenSync(t.Context(), msgs, opts, tools, genai.EffortLow)
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Replies) != 1 || res.Replies[0].ToolCall.Name != "report_marker" || res.Usage.FinishReason != genai.FinishedToolCalls {
			t.Fatalf("tool result %+v", res)
		}
		var args markerInput
		if err := json.Unmarshal([]byte(res.Replies[0].ToolCall.Arguments), &args); err != nil || args.Marker != "luna-tools-ok" {
			t.Fatalf("arguments %+v error %v", args, err)
		}
		msgs = append(msgs, res.Message)
		call := res.Replies[0].ToolCall
		msgs = append(msgs, genai.Message{ToolCallResults: []genai.ToolCallResult{{ID: call.ID, Name: call.Name, Result: "marker-confirmed"}}})
		res, err = c.GenSync(t.Context(), msgs, opts, genai.EffortLow)
		if err != nil {
			t.Fatal(err)
		}
		if res.String() != "marker-confirmed" {
			t.Fatalf("continuation %q", res.String())
		}
		msgs = append(msgs, res.Message, genai.NewTextMessage("What was the marker originally passed to the tool? Reply with just the marker."))
		res, err = c.GenSync(t.Context(), msgs, &genai.GenOptionText{SystemPrompt: "Answer briefly."}, genai.EffortLow)
		if err != nil {
			t.Fatal(err)
		}
		if res.String() != "luna-tools-ok" {
			t.Fatalf("history recall %q", res.String())
		}
	})
	t.Run("GenSyncRaw", func(t *testing.T) {
		c := getClient(t, genai.ProviderOptionModel("gpt-6-luna"))
		var in, out openairesponses.Response
		if err := in.Init(genai.Messages{genai.NewTextMessage("Reply exactly: luna-raw-ok")}, "gpt-6-luna"); err != nil {
			t.Fatal(err)
		}
		if err := c.GenSyncRaw(t.Context(), &in, &out); err != nil {
			t.Fatal(err)
		}
		res, err := out.ToResult()
		if err != nil {
			t.Fatal(err)
		}
		if res.String() != "luna-raw-ok" || out.Status != "completed" {
			t.Fatalf("raw response %+v", out)
		}
	})
	t.Run("New", func(t *testing.T) {
		for _, opt := range []genai.ProviderOption{genai.ProviderOptionAPIKey("secret"), genai.ProviderOptionRemote("https://example.com"), genai.ProviderOptionModalities{genai.ModalityAudio}, genai.ModelGood} {
			c, err := openairesponses.New(t.Context(), &openairesponses.ProviderOption{Auth: openairesponses.AuthCodex}, opt)
			if c != nil || err == nil {
				t.Fatalf("accepted %T", opt)
			}
		}
		for _, opt := range []*openairesponses.ProviderOption{{Auth: "unknown"}, {Auth: openairesponses.AuthCodex, AppName: "bad\r\nheader"}} {
			if c, err := openairesponses.New(t.Context(), opt); c != nil || err == nil {
				t.Fatal("accepted invalid provider options")
			}
		}
		c, err := openairesponses.New(t.Context(), &openairesponses.ProviderOption{Auth: openairesponses.AuthCodex}, genai.ModelCheap)
		if err != nil {
			t.Fatal(err)
		}
		internaltest.CleanupCloser(t, c)
		if c.ModelID() != "gpt-6-luna" || c.Name() != "openairesponses" || c.Capabilities() != (genai.ProviderCapabilities{}) {
			t.Fatal("incorrect subscription metadata")
		}
		for _, opt := range []genai.GenOption{&genai.GenOptionText{MaxTokens: 1}, &openairesponses.GenOptionText{PreviousResponseID: "resp"}} {
			if _, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("hello")}, opt); err == nil {
				t.Fatalf("accepted unsupported subscription generation option %T", opt)
			}
		}
		if s, err := c.ConnectRealtime(t.Context()); s != nil || err == nil {
			t.Fatal("accepted subscription realtime")
		}
		if s, err := c.WebSocket(t.Context()); s != nil || err == nil {
			t.Fatal("accepted subscription websocket")
		}
		if j, err := c.GenAsync(t.Context(), nil); j != "" || err == nil {
			t.Fatal("accepted subscription background job")
		}
		if r, err := c.Embed(t.Context(), nil); r != nil || err == nil {
			t.Fatal("accepted subscription embeddings")
		}
	})
}

type markerInput struct {
	Marker string `json:"marker"`
}
type codexTestTransport func(*http.Request) (*http.Response, error)

func (f codexTestTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type codexTestModels struct {
	Models []codexTestModel `json:"models"`
}
type codexTestModel struct {
	Slug          string `json:"slug"`
	DisplayName   string `json:"display_name"`
	ContextWindow int64  `json:"context_window"`
}

func codexTestCredentials(t *testing.T) string {
	dir := t.TempDir()
	token := "header." + base64.RawURLEncoding.EncodeToString([]byte(`{"exp":4102444800,"https://api.openai.com/auth":{"chatgpt_user_id":"test-user"}}`)) + ".signature"
	raw, err := json.Marshal(map[string]any{"auth_mode": "chatgpt", "tokens": map[string]string{"access_token": token, "account_id": "test-account"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "auth.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func sanitizeCodexRecording(i *cassette.Interaction) error {
	if strings.Contains(i.Request.URL, "/models?") {
		var out codexTestModels
		if err := json.Unmarshal([]byte(i.Response.Body), &out); err != nil {
			return err
		}
		b, err := json.Marshal(out)
		if err != nil {
			return err
		}
		i.Response.Body = string(b)
	} else if strings.Contains(i.Request.URL, "/responses") {
		lines := strings.Split(i.Response.Body, "\n")
		for n, line := range lines {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			var event map[string]any
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
				return err
			}
			if response, ok := event["response"].(map[string]any); ok {
				delete(response, "safety_identifier")
				delete(response, "prompt_cache_key")
			}
			b, err := json.Marshal(event)
			if err != nil {
				return err
			}
			lines[n] = "data: " + string(b)
		}
		i.Response.Body = strings.Join(lines, "\n")
	}
	return nil
}
