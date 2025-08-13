// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"context"
	_ "embed"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := anthropic.New(&genai.OptionsProvider{APIKey: apiKey, Model: model.Model}, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
		return &injectOption{Client: c, opts: anthropic.OptionsText{ThinkingBudget: 1024}}
	}
	return &injectOption{Client: c, opts: anthropic.OptionsText{ThinkingBudget: 0}}
}

type injectOption struct {
	*anthropic.Client
	opts anthropic.OptionsText
}

func (i *injectOption) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		n.OptionsText = *opts.(*genai.OptionsText)
	}
	opts = &n
	return i.Client.GenSync(ctx, msgs, opts)
}

func (i *injectOption) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		n.OptionsText = *opts.(*genai.OptionsText)
	}
	opts = &n
	return i.Client.GenStream(ctx, msgs, replies, opts)
}

func TestClient_Scoreboard(t *testing.T) {
	genaiModels, err := getClient(t, base.NoModel).ListModels(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var models []scoreboardtest.Model
	for _, m := range genaiModels {
		id := m.GetID()
		models = append(models, scoreboardtest.Model{Model: id})
		if strings.HasPrefix(id, "claude-sonnet") {
			models = append(models, scoreboardtest.Model{Model: id, Thinking: true})
		}
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

// This is a tricky test since batch operations can take up to 24h to complete.
func TestClient_Batch(t *testing.T) {
	ctx := t.Context()
	// Using an extremely old cheap model that nobody uses helps a lot on reducing the latency, I got it to work
	// within a few minutes.
	c := getClient(t, "claude-3-haiku-20240307")
	msgs := genai.Messages{genai.NewTextMessage("Tell a joke in 10 words")}
	job, err := c.GenAsync(ctx, msgs, nil)
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
		if res.FinishReason == genai.Pending {
			if isRecording {
				t.Logf("Waiting...")
				time.Sleep(time.Second)
			}
			continue
		}
		if res.InputTokens == 0 || res.OutputTokens == 0 {
			t.Error("expected usage")
		}
		if res.FinishReason != genai.FinishedStop {
			t.Errorf("finish reason: %s", res.FinishReason)
		}
		if s := res.AsText(); len(s) < 15 {
			t.Errorf("not enough text: %q", s)
		}
		break
	}
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "claude-3-5-haiku-20241022"},
		{base.PreferredGood, "claude-sonnet-4-20250514"},
		{base.PreferredSOTA, "claude-opus-4-20250514"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			if got := getClient(t, line.name).ModelID(); got != line.want {
				t.Fatalf("got model %q, want %q", got, line.want)
			}
		})
	}
}

func TestClient_Provider_errors(t *testing.T) {
	data := []internaltest.ProviderError{
		{
			Name:         "bad apiKey",
			APIKey:       "bad apiKey",
			Model:        "claude-3-haiku-20240307",
			ErrGenSync:   "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
			ErrGenStream: "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
			ErrListModel: "http 401\nauthentication_error: invalid x-api-key\nget a new API key at https://console.anthropic.com/settings/keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404\nnot_found_error: model: bad model",
			ErrGenStream: "http 404\nnot_found_error: model: bad model",
			ErrListModel: "",
		},
	}
	f := func(t *testing.T, apiKey, model string) (genai.Provider, error) {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *anthropic.Client {
	t.Parallel()
	c, err := getClientInner(t, "", m)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, apiKey, m string) (*anthropic.Client, error) {
	if apiKey == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	return anthropic.New(&genai.OptionsProvider{APIKey: apiKey, Model: m}, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
