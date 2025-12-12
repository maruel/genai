// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	_ "embed"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
)

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return anthropic.New(t.Context(), &opts, fn)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, genai.ProviderOptions{Model: genai.ModelNone}, func(h http.RoundTripper) http.RoundTripper {
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
		opts := genai.ProviderOptions{Model: m, PreloadedModels: cachedModels}
		ci, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Scoreboard", func(t *testing.T) {
		genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var models []smoketest.Model
		for _, m := range genaiModels {
			id := m.GetID()
			models = append(models, smoketest.Model{Model: id})
			if strings.HasPrefix(id, "claude-sonnet") || strings.HasPrefix(id, "claude-opus") {
				models = append(models, smoketest.Model{Model: id, Reason: true})
			}
		}

		getClientRT := func(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := genai.ProviderOptions{Model: model.Model, PreloadedModels: cachedModels}
			if os.Getenv("ANTHROPIC_API_KEY") == "" {
				opts.APIKey = "<insert_api_key_here>"
			}
			c, err := anthropic.New(t.Context(), &opts, fn)
			if err != nil {
				t.Fatal(err)
			}
			if model.Reason {
				return &internaltest.InjectOptions{
					Provider: c,
					Opts:     []genai.Options{&anthropic.OptionsText{ThinkingBudget: 1024}},
				}
			}
			return &internaltest.InjectOptions{
				Provider: c,
				Opts:     []genai.Options{&anthropic.OptionsText{ThinkingBudget: 0}},
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
		// Anthropic requires an HTTP header to enable MCP use. See
		// https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector
		wrapper := func(h http.RoundTripper) http.RoundTripper {
			return &roundtrippers.Header{
				Transport: h,
				Header:    http.Header{"anthropic-beta": []string{"mcp-client-2025-04-04"}},
			}
		}
		prompt := "Remember that my name is Bob, my mother is Jane and my father is John. If you saved it, reply with 'Done'."
		msgs := genai.Messages{genai.NewTextMessage(prompt)}
		mcp := anthropic.MCPServer{
			Name:              "memory",
			Type:              "url",
			URL:               "https://mcp.maruel.ca",
			ToolConfiguration: anthropic.ToolConfiguration{Enabled: true},
		}
		t.Run("GenSyncRaw", func(t *testing.T) {
			cc, err := getClientInner(t, genai.ProviderOptions{}, func(h http.RoundTripper) http.RoundTripper {
				return wrapper(testRecorder.Record(t, h))
			})
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
			if got := res.String(); !strings.Contains(got, "memory") {
				t.Fatal(got)
			}
		})
		t.Run("GenStreamRaw", func(t *testing.T) {
			cc, err := getClientInner(t, genai.ProviderOptions{}, func(h http.RoundTripper) http.RoundTripper {
				return wrapper(testRecorder.Record(t, h))
			})
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
			if got := res.String(); !strings.Contains(got, "memory") {
				t.Fatal(got)
			}
		})
	})

	t.Run("Preferred", func(t *testing.T) {
		data := []struct {
			name string
			want string
		}{
			{genai.ModelCheap, "claude-haiku-4-5-20251001"},
			{genai.ModelGood, "claude-sonnet-4-5-20250929"},
			{genai.ModelSOTA, "claude-opus-4-1-20250805"},
		}
		for _, line := range data {
			t.Run(line.name, func(t *testing.T) {
				if got := getClient(t, line.name).ModelID(); got != line.want {
					t.Fatalf("got model %q, want %q", got, line.want)
				}
			})
		}
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: genai.ProviderOptions{
					APIKey: "bad apiKey",
					Model:  "claude-3-haiku-20240307",
				},
				ErrGenSync:   "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
				ErrGenStream: "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
				ErrListModel: "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 404\nnot_found_error: model: bad model",
				ErrGenStream: "http 404\nnot_found_error: model: bad model",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			opts.OutputModalities = genai.Modalities{genai.ModalityText}
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClient_Provider_errors(t, f, data)
	})
}

func init() {
	internal.BeLenient = false
}
