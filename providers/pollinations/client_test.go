// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package pollinations_test

import (
	"context"
	_ "embed"
	"errors"
	"net/http"
	"os"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/pollinations"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
	"github.com/maruel/httpjson"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	c, err := pollinations.New(t.Context(), &genai.ProviderOptions{Model: model.Model}, fn)
	if err != nil {
		t.Fatal(err)
	}
	cachedModels = warmupCache(t)
	isImage := false
	for i := range cachedModels {
		if cachedModels[i].GetID() == model.Model {
			_, isImage = cachedModels[i].(pollinations.ImageModel)
			break
		}
	}
	c2 := &hideHTTP500{Client: c}
	if isImage {
		return &imageModelClient{parent: c2}
	}
	if model.Thinking {
		return &adapters.ProviderThinking{Provider: c2, ThinkingTokenStart: "<think>", ThinkingTokenEnd: "\n</think>\n"}
	}
	return c2
}

type hideHTTP500 struct {
	*pollinations.Client
}

func (h *hideHTTP500) Unwrap() genai.Provider {
	return h.Client
}

func (h *hideHTTP500) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenSync(ctx, msgs, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; pollinations.ai throws HTTP 500 on unsupported file formats.
			return resp, errors.New("pollinations.ai is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

func (h *hideHTTP500) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenStream(ctx, msgs, chunks, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; pollinations.ai throws HTTP 500 on unsupported file formats.
			return resp, errors.New("pollinations.ai is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

func (h *hideHTTP500) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenDoc(ctx, msg, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; pollinations.ai throws HTTP 500 on unsupported file formats.
			return resp, errors.New("pollinations.ai is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

type parent interface {
	genai.ProviderGenDoc
	scoreboard.ProviderScore
}

type imageModelClient struct {
	parent
}

func (i *imageModelClient) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	if v, ok := opts.(*genai.OptionsImage); ok {
		// Ask for a smaller size.
		n := *v
		n.Width = 256
		n.Height = 256
		opts = &n
	}
	return i.parent.GenDoc(ctx, msg, opts)
}

func TestClient_Scoreboard(t *testing.T) {
	models := warmupCache(t)
	var sbModels []scoreboardtest.Model
	for _, m := range models {
		id := m.GetID()
		sbModels = append(sbModels, scoreboardtest.Model{Model: id, Thinking: id == "deepseek-reasoning"})
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, sbModels, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, "llamascout"},
		{genai.ModelGood, "openai-large"},
		{genai.ModelSOTA, "deepseek-reasoning"},
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
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "model \"bad model\" not supported by pollinations",
			ErrGenStream: "model \"bad model\" not supported by pollinations",
			ErrGenDoc:    "model \"bad model\" not supported by pollinations",
		},
	}
	f := func(t *testing.T, apiKey, model string) (genai.Provider, error) {
		return getClientInner(t, &genai.ProviderOptions{APIKey: "genai-unittests", Model: model, OutputModalities: genai.Modalities{genai.ModalityText}})
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *pollinations.Client {
	t.Parallel()
	c, err := getClientInner(t, &genai.ProviderOptions{APIKey: "genai-unittests", Model: m})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, opts *genai.ProviderOptions) (*pollinations.Client, error) {
	c, err := pollinations.New(t.Context(), opts, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
	if err != nil {
		return nil, err
	}
	warmupCache(t)
	return c, nil
}

func warmupCache(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r internal.Recorder
		var err2 error
		ctx := t.Context()
		c, err := pollinations.New(ctx, &genai.ProviderOptions{APIKey: "genai-unittests", Model: genai.ModelNone}, func(h http.RoundTripper) http.RoundTripper {
			r, err2 = testRecorder.Records.Record("WarmupCache", h)
			return r
		})
		if err != nil {
			t.Fatal(err)
		}
		if err2 != nil {
			t.Fatal(err2)
		}
		if cachedModels, err = pollinations.Cache.Warmup(ctx, c); err != nil {
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
