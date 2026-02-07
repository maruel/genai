// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
)

func getClientInner(t *testing.T, fn func(http.RoundTripper) http.RoundTripper, opts ...genai.ProviderOption) (genai.Provider, error) {
	hasAPIKey := os.Getenv("ANTHROPIC_API_KEY") != ""
	for _, opt := range opts {
		if _, ok := opt.(genai.ProviderOptionAPIKey); ok {
			hasAPIKey = true
			break
		}
	}
	if !hasAPIKey {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	return anthropic.New(t.Context(), opts...)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.RecordWithName(t, t.Name()+"/Warmup", h)
	})
	if err2 != nil {
		t.Fatal(err2)
	}
	cachedModels, err2 := cl.ListModels(t.Context())
	if err2 != nil {
		t.Fatal(err2)
	}
	getClient := func(t *testing.T, m string) genai.Provider {
		t.Parallel()
		opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
		if m != "" {
			opts = append(opts, genai.ProviderOptionModel(m))
		}
		ci, err := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		}, opts...)
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, ""))
	})

	t.Run("GenAsync-Text", func(t *testing.T) {
		internaltest.TestCapabilitiesGenAsync(t, getClient(t, string(genai.ModelCheap)))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		genaiModels, err := getClient(t, "").ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var models []scoreboard.Model
		for _, m := range genaiModels {
			id := m.GetID()
			models = append(models, scoreboard.Model{Model: id})
			if strings.HasPrefix(id, "claude-sonnet") || strings.HasPrefix(id, "claude-opus") {
				models = append(models, scoreboard.Model{Model: id, Reason: true})
			}
		}

		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			popts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				popts = append(popts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("ANTHROPIC_API_KEY") == "" {
				popts = append(popts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if fn != nil {
				popts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, popts...)
			}
			c, err := anthropic.New(t.Context(), popts...)
			if err != nil {
				t.Fatal(err)
			}
			if model.Reason {
				return &internaltest.InjectOptions{
					Provider: c,
					Opts:     []genai.GenOptions{&anthropic.GenOptionsText{ThinkingBudget: 1024}},
				}
			}
			return &internaltest.InjectOptions{
				Provider: c,
				Opts:     []genai.GenOptions{&anthropic.GenOptionsText{ThinkingBudget: 0}},
			}
		}

		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	// This is a tricky test since batch operations can take up to 24h to complete.
	t.Run("Batch", func(t *testing.T) {
		ctx := t.Context()
		// Using an extremely old cheap model that nobody uses helps a lot on reducing the latency, I got it to work
		// within a few minutes.
		c := getClient(t, "claude-3-haiku-20240307").(*anthropic.Client)
		msgs := genai.Messages{genai.NewTextMessage("Tell a joke in 10 words")}
		job, err := c.GenAsync(ctx, msgs)
		if err != nil {
			t.Fatal(err)
		}
		// TODO: Detect when recording and sleep only in this case.
		isRecording := os.Getenv("RECORD") == "1"
		for {
			res, err := c.PokeResult(ctx, job)
			if err != nil {
				t.Fatal(err)
			}
			if res.Usage.FinishReason == genai.Pending {
				if isRecording {
					t.Logf("Waiting...")
					time.Sleep(time.Second)
				}
				continue
			}
			if res.Usage.InputTokens == 0 || res.Usage.OutputTokens == 0 {
				t.Error("expected usage")
			}
			if res.Usage.FinishReason != genai.FinishedStop {
				t.Errorf("finish reason: %s", res.Usage.FinishReason)
			}
			if s := res.String(); len(s) < 15 {
				t.Errorf("not enough text: %q", s)
			}
			break
		}
	})

	t.Run("MCP", func(t *testing.T) {
		// TODO: Re-record cassettes from a machine where Anthropic's MCP
		// connector can reach the Smithery server.
		t.Skip("MCP cassettes need re-recording with Smithery service token")
		// Anthropic requires an HTTP header to enable MCP use. See
		// https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector
		wrapper := func(h http.RoundTripper) http.RoundTripper {
			return &roundtrippers.Header{
				Transport: h,
				Header:    http.Header{"anthropic-beta": []string{"mcp-client-2025-04-04"}},
			}
		}
		// Get a Smithery service token from the API key.
		// See https://smithery.ai/docs/use/connect#service-tokens
		token, err := getSmitheryServiceToken(os.Getenv("SMITHERY_API_KEY"), "@simonfraserduncan/echo-mcp")
		if err != nil {
			t.Fatal(err)
		}
		prompt := "Use the echo tool to echo 'hello world'."
		msgs := genai.Messages{genai.NewTextMessage(prompt)}
		mcp := anthropic.MCPServer{
			Name:               "echo",
			Type:               "url",
			URL:                "https://server.smithery.ai/@simonfraserduncan/echo-mcp",
			AuthorizationToken: token,
			ToolConfiguration:  anthropic.ToolConfiguration{Enabled: true},
		}
		t.Run("GenSyncRaw", func(t *testing.T) {
			cc, err := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return wrapper(testRecorder.Record(t, h))
			}, genai.ModelGood)
			if err != nil {
				log.Fatal(err)
			}
			c := cc.(*anthropic.Client)
			// Use raw calls to use the MCP client. It is not yet generalized in genai.
			in := anthropic.ChatRequest{MCPServers: []anthropic.MCPServer{mcp}}
			if err = in.Init(msgs, c.ModelID()); err != nil {
				t.Fatal(err)
			}
			out := anthropic.ChatResponse{}
			if err = c.GenSyncRaw(t.Context(), &in, &out); err != nil {
				t.Fatal(err)
			}
			res, err := out.ToResult()
			if err != nil {
				t.Fatal(err)
			}
			if got := res.String(); !strings.Contains(got, "echo") {
				t.Fatal(got)
			}
		})
		t.Run("GenStreamRaw", func(t *testing.T) {
			cc, err := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return wrapper(testRecorder.Record(t, h))
			}, genai.ModelGood)
			if err != nil {
				log.Fatal(err)
			}
			c := cc.(*anthropic.Client)
			// Use raw calls to use the MCP client. It is not yet generalized in genai.
			in := anthropic.ChatRequest{MCPServers: []anthropic.MCPServer{mcp}}
			if err = in.Init(msgs, c.ModelID()); err != nil {
				t.Fatal(err)
			}
			chunks, finish := c.GenStreamRaw(t.Context(), &in)
			fragments, finish2 := anthropic.ProcessStream(chunks)
			res := genai.Result{}
			for f := range fragments {
				if f.IsZero() {
					continue
				}
				if err = f.Validate(); err != nil {
					t.Fatal(err)
				}
				if err = res.Accumulate(f); err != nil {
					t.Fatal(err)
				}
			}
			if err = finish(); err != nil {
				t.Fatal(err)
			}
			if res.Usage, res.Logprobs, err = finish2(); err != nil {
				t.Fatal(err)
			}
			if got := res.String(); !strings.Contains(got, "echo") {
				t.Fatal(got)
			}
		})
	})

	t.Run("CountTokens", func(t *testing.T) {
		c := getClient(t, string(genai.ModelCheap)).(*anthropic.Client)
		msgs := genai.Messages{genai.NewTextMessage("Hello, world!")}
		resp, err := c.CountTokens(t.Context(), msgs)
		if err != nil {
			t.Fatal(err)
		}
		if resp.InputTokens <= 0 {
			t.Fatalf("expected positive input tokens, got %d", resp.InputTokens)
		}
	})

	t.Run("GetModel", func(t *testing.T) {
		c := getClient(t, "").(*anthropic.Client)
		m, err := c.GetModel(t.Context(), "claude-sonnet-4-20250514")
		if err != nil {
			t.Fatal(err)
		}
		if m.ID != "claude-sonnet-4-20250514" {
			t.Errorf("ID = %q, want %q", m.ID, "claude-sonnet-4-20250514")
		}
		if m.DisplayName == "" {
			t.Error("expected DisplayName to be set")
		}
		if m.CreatedAt.IsZero() {
			t.Error("expected CreatedAt to be set")
		}
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			return getClientInner(st, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			}, opts...)
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, string(genai.ModelCheap))
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("claude-3-haiku-20240307"),
				},
				ErrGenSync:   "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
				ErrGenStream: "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
				ErrListModel: "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 404\nnot_found_error: model: bad model",
				ErrGenStream: "http 404\nnot_found_error: model: bad model",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			return getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			}, opts...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func TestStructuredOutput(t *testing.T) {
	type myStruct struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	msgs := genai.Messages{genai.NewTextMessage("test")}
	tests := []struct {
		name       string
		opts       []genai.GenOptions
		wantFormat bool
		wantType   string
		wantSchema string // "object" or "full"
	}{
		{
			name: "DecodeAs",
			opts: []genai.GenOptions{&genai.GenOptionsText{
				DecodeAs: &myStruct{},
			}},
			wantFormat: true,
			wantType:   "json_schema",
			wantSchema: "full",
		},
		{
			name: "ReplyAsJSON",
			opts: []genai.GenOptions{&genai.GenOptionsText{
				ReplyAsJSON: true,
			}},
			wantFormat: true,
			wantType:   "json_schema",
			wantSchema: "object",
		},
		{
			name:       "Neither",
			opts:       []genai.GenOptions{&genai.GenOptionsText{}},
			wantFormat: false,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var req anthropic.ChatRequest
			if err := req.Init(msgs, "claude-sonnet-4-20250514", tc.opts...); err != nil {
				t.Fatal(err)
			}
			if !tc.wantFormat {
				if req.OutputConfig.Format.Type != "" {
					t.Fatal("expected OutputConfig.Format to be zero")
				}
				return
			}
			if req.OutputConfig.Format.Type == "" {
				t.Fatal("expected OutputConfig.Format to be set")
			}
			if got := req.OutputConfig.Format.Type; got != tc.wantType {
				t.Errorf("Type = %q, want %q", got, tc.wantType)
			}
			if req.OutputConfig.Format.Schema == nil {
				t.Fatal("expected Schema to be set")
			}
			switch tc.wantSchema {
			case "object":
				if got := req.OutputConfig.Format.Schema.Type; got != "object" {
					t.Errorf("Schema.Type = %q, want \"object\"", got)
				}
			case "full":
				if got := req.OutputConfig.Format.Schema.Type; got != "object" {
					t.Errorf("Schema.Type = %q, want \"object\"", got)
				}
				if req.OutputConfig.Format.Schema.Properties == nil {
					t.Error("expected Schema.Properties to be set for DecodeAs")
				}
			}
		})
	}
}

func TestEffort(t *testing.T) {
	msgs := genai.Messages{genai.NewTextMessage("test")}
	t.Run("valid", func(t *testing.T) {
		for _, e := range []anthropic.Effort{"", anthropic.EffortLow, anthropic.EffortMedium, anthropic.EffortHigh, anthropic.EffortMax} {
			t.Run(string(e), func(t *testing.T) {
				var req anthropic.ChatRequest
				if err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{Effort: e}); err != nil {
					t.Fatal(err)
				}
				if req.OutputConfig.Effort != e {
					t.Errorf("Effort = %q, want %q", req.OutputConfig.Effort, e)
				}
			})
		}
	})
	t.Run("invalid", func(t *testing.T) {
		var req anthropic.ChatRequest
		err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{Effort: "bogus"})
		if err == nil {
			t.Fatal("expected error for invalid effort")
		}
		if !strings.Contains(err.Error(), "invalid Effort") {
			t.Errorf("unexpected error: %v", err)
		}
	})
}

func TestThinking(t *testing.T) {
	msgs := genai.Messages{genai.NewTextMessage("test")}
	t.Run("valid", func(t *testing.T) {
		t.Run("adaptive", func(t *testing.T) {
			var req anthropic.ChatRequest
			if err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{Thinking: anthropic.ThinkingAdaptive}); err != nil {
				t.Fatal(err)
			}
			if req.Thinking.Type != "adaptive" {
				t.Errorf("Thinking.Type = %q, want %q", req.Thinking.Type, "adaptive")
			}
			if req.Thinking.BudgetTokens != 0 {
				t.Errorf("Thinking.BudgetTokens = %d, want 0", req.Thinking.BudgetTokens)
			}
		})
		t.Run("enabled", func(t *testing.T) {
			var req anthropic.ChatRequest
			if err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{Thinking: anthropic.ThinkingEnabled, ThinkingBudget: 2048}); err != nil {
				t.Fatal(err)
			}
			if req.Thinking.Type != "enabled" {
				t.Errorf("Thinking.Type = %q, want %q", req.Thinking.Type, "enabled")
			}
			if req.Thinking.BudgetTokens != 2048 {
				t.Errorf("Thinking.BudgetTokens = %d, want 2048", req.Thinking.BudgetTokens)
			}
		})
		t.Run("disabled", func(t *testing.T) {
			var req anthropic.ChatRequest
			if err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{Thinking: anthropic.ThinkingDisabled}); err != nil {
				t.Fatal(err)
			}
			if req.Thinking.Type != "disabled" {
				t.Errorf("Thinking.Type = %q, want %q", req.Thinking.Type, "disabled")
			}
		})
		t.Run("auto_detect_enabled", func(t *testing.T) {
			var req anthropic.ChatRequest
			if err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{ThinkingBudget: 2048}); err != nil {
				t.Fatal(err)
			}
			if req.Thinking.Type != "enabled" {
				t.Errorf("Thinking.Type = %q, want %q", req.Thinking.Type, "enabled")
			}
			if req.Thinking.BudgetTokens != 2048 {
				t.Errorf("Thinking.BudgetTokens = %d, want 2048", req.Thinking.BudgetTokens)
			}
		})
		t.Run("auto_detect_disabled", func(t *testing.T) {
			var req anthropic.ChatRequest
			if err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{}); err != nil {
				t.Fatal(err)
			}
			if req.Thinking.Type != "disabled" {
				t.Errorf("Thinking.Type = %q, want %q", req.Thinking.Type, "disabled")
			}
		})
	})
	t.Run("invalid", func(t *testing.T) {
		t.Run("bogus", func(t *testing.T) {
			var req anthropic.ChatRequest
			err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{Thinking: "bogus"})
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), "invalid ThinkingType") {
				t.Errorf("unexpected error: %v", err)
			}
		})
		t.Run("adaptive_with_budget", func(t *testing.T) {
			var req anthropic.ChatRequest
			err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{Thinking: anthropic.ThinkingAdaptive, ThinkingBudget: 2048})
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), "ThinkingBudget must not be set") {
				t.Errorf("unexpected error: %v", err)
			}
		})
		t.Run("enabled_without_budget", func(t *testing.T) {
			var req anthropic.ChatRequest
			err := req.Init(msgs, "claude-sonnet-4-20250514", &anthropic.GenOptionsText{Thinking: anthropic.ThinkingEnabled})
			if err == nil {
				t.Fatal("expected error")
			}
			if !strings.Contains(err.Error(), "ThinkingBudget must be set") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	})
}

// getSmitheryServiceToken exchanges a Smithery API key for a scoped service
// token. See https://smithery.ai/docs/use/connect#service-tokens
func getSmitheryServiceToken(apiKey, namespace string) (string, error) {
	type tokenPolicy struct {
		Namespaces []string `json:"namespaces"`
		Resources  []string `json:"resources"`
		Operations []string `json:"operations"`
		TTL        string   `json:"ttl"`
	}
	body, err := json.Marshal(struct {
		Policy []tokenPolicy `json:"policy"`
	}{
		Policy: []tokenPolicy{{
			Namespaces: []string{namespace},
			Resources:  []string{"connections"},
			Operations: []string{"read", "execute"},
			TTL:        "1h",
		}},
	})
	if err != nil {
		return "", err
	}
	req, err := http.NewRequest("POST", "https://registry.smithery.ai/tokens", bytes.NewReader(body))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+apiKey)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusCreated {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("smithery token API error (status %d): %s", resp.StatusCode, b)
	}
	var result struct {
		Token string `json:"token"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return "", err
	}
	return result.Token, nil
}

func init() {
	internal.BeLenient = false
}
