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
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/pollinations"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
	"github.com/maruel/httpjson"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func gc(t testing.TB, name, m string) (genai.Provider, http.RoundTripper) {
	var rt http.RoundTripper
	fn := func(h http.RoundTripper) http.RoundTripper {
		if name == "" {
			rt = h
			return h

		}
		r, err2 := testRecorder.Records.Record(name, h)
		if err2 != nil {
			t.Fatal(err2)
		}
		t.Cleanup(func() {
			if err3 := r.Stop(); err3 != nil {
				t.Error(err3)
			}
		})
		rt = r
		return r
	}
	c, err := pollinations.New("", m, fn)
	if err != nil {
		t.Fatal(err)
	}
	cachedModels = warmupCache(t)
	isImage := false
	for i := range cachedModels {
		if cachedModels[i].GetID() == c.Model {
			_, isImage = cachedModels[i].(pollinations.ImageModel)
			break
		}
	}
	c2 := &hideHTTP500{Client: c}
	if isImage {
		return &imageModelClient{parent: c2}, rt
	}
	if m == "deepseek-reasoning" {
		return &adapters.ProviderGenThinking{ProviderGen: c2, TagName: "think"}, rt
	}
	return c2, rt
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
			return resp, errors.New("together.ai is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

func (h *hideHTTP500) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	resp, err := h.Client.GenStream(ctx, msgs, chunks, opts)
	if err != nil {
		var herr *httpjson.Error
		if errors.As(err, &herr) && herr.StatusCode == 500 {
			// Hide the failure; pollinations.ai throws HTTP 500 on unsupported file formats.
			return resp, errors.New("together.ai is having a bad day")
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
			return resp, errors.New("together.ai is having a bad day")
		}
		return resp, err
	}
	return resp, err
}

type parent interface {
	genai.ProviderGenDoc
	genai.ProviderModel
	genai.ProviderScoreboard
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
	usage := genai.Usage{}
	models := warmupCache(t)
	for _, m := range models {
		id := m.GetID()
		t.Run(id, func(t *testing.T) {
			// Run one model at a time otherwise we can't collect the total usage.
			usage.Add(scoreboardtest.RunOneModel(t, func(t testing.TB, sn string) (genai.Provider, http.RoundTripper) {
				return gc(t, sn, id)
			}))
		})
	}
	t.Logf("Usage: %#v", usage)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "llamascout"},
		{base.PreferredGood, "openai-large"},
		{base.PreferredSOTA, "deepseek-reasoning"},
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
	t.Skip("TODO")
	data := []internaltest.ProviderError{
		{
			Name:         "bad apiKey",
			APIKey:       "bad apiKey",
			Model:        "llama3-8b-8192",
			ErrGenSync:   "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
			ErrGenStream: "http 401: error invalid_api_key (invalid_request_error): Invalid API Key. You can get a new API key at https://console.groq.com/keys",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404: error model_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
			ErrGenStream: "http 404: error model_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.Provider {
		return getClientInner(t, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *pollinations.Client {
	t.Parallel()
	return getClientInner(t, m)
}

func getClientInner(t *testing.T, m string) *pollinations.Client {
	c, err := pollinations.New("genai-unittests", m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
	if err != nil {
		t.Fatal(err)
	}
	warmupCache(t)
	return c
}

func warmupCache(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r *recorder.Recorder
		var err2 error
		c, err := pollinations.New("genai-unittests", "", func(h http.RoundTripper) http.RoundTripper {
			r, err2 = testRecorder.Records.Record("WarmupCache", h)
			return r
		})
		if err != nil {
			t.Fatal(err)
		}
		if err2 != nil {
			t.Fatal(err2)
		}
		if cachedModels, err = pollinations.Cache.Warmup(c); err != nil {
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
