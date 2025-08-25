// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	_ "embed"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	opts := genai.ProviderOptions{
		APIKey:          apiKey,
		Model:           model.Model,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := anthropic.New(t.Context(), &opts, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
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

func TestClient_Scoreboard(t *testing.T) {
	genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
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
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, "claude-3-5-haiku-20241022"},
		{genai.ModelGood, "claude-sonnet-4-20250514"},
		{genai.ModelSOTA, "claude-opus-4-1-20250805"},
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
		return getClientInner(t, opts)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *anthropic.Client {
	t.Parallel()
	opts := genai.ProviderOptions{
		Model:           m,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := getClientInner(t, opts)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, opts genai.ProviderOptions) (*anthropic.Client, error) {
	if opts.APIKey == "" && os.Getenv("ANTHROPIC_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return anthropic.New(t.Context(), &opts, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.Record(t, h)
	})
}

func loadCachedModelsList(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r myrecorder.Recorder
		var err2 error
		ctx := t.Context()
		opts := genai.ProviderOptions{Model: genai.ModelNone}
		if os.Getenv("ANTHROPIC_API_KEY") == "" {
			opts.APIKey = "<insert_api_key_here>"
		}
		c, err := anthropic.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
			r, err2 = testRecorder.Records.Record("WarmupCache", h)
			return r
		})
		if err != nil {
			t.Fatal(err)
		}
		if err2 != nil {
			t.Fatal(err2)
		}
		if cachedModels, err = c.ListModels(ctx); err != nil {
			t.Fatal(err)
		}
		if err = r.Stop(); err != nil {
			t.Fatal(err)
		}
	})
	return cachedModels
}

var doOnce sync.Once

var cachedModels []genai.Model

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
