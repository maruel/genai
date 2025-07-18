// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package bfl

import (
	"context"
	_ "embed"
	"errors"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
)

func TestClient_Scoreboard(t *testing.T) {
	internaltest.TestScoreboard(t, func(t *testing.T, m string) genai.ProviderGen {
		return &injectOption{Client: getClient(t, m), t: t, opts: genai.OptionsImage{Width: 256, Height: 256}}
	}, nil)
}

type injectOption struct {
	*Client
	t    *testing.T
	opts genai.OptionsImage
}

func (i *injectOption) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		return genai.Result{}, errors.New("implement me")
	}
	opts = &n
	p := adapters.ProviderGenDocToGen{ProviderGenDoc: i.Client}
	return p.GenSync(ctx, msgs, opts)
}

func (i *injectOption) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		return genai.Result{}, errors.New("implement me")
	}
	opts = &n
	p := adapters.ProviderGenDocToGen{ProviderGenDoc: i.Client}
	return p.GenStream(ctx, msgs, replies, opts)
}

func (i *injectOption) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		return genai.Result{}, errors.New("implement me")
	}
	opts = &n
	return i.Client.GenDoc(ctx, msg, opts)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "flux-dev"},
		{base.PreferredGood, "flux-pro-1.1"},
		{base.PreferredSOTA, "flux-pro-1.1-ultra"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			if got := getClient(t, line.name).ModelID(); got != line.want {
				t.Fatalf("got model %q, want %q", got, line.want)
			}
		})
	}
}

func TestClient_ProviderGen_errors(t *testing.T) {
	data := []internaltest.ProviderGenError{
		{
			Name:         "bad apiKey",
			ApiKey:       "bad apiKey",
			Model:        "flux-dev",
			ErrGenSync:   "http 403: error Not authenticated - Invalid Authentication",
			ErrGenStream: "http 403: error Not authenticated - Invalid Authentication",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 404: error Not Found",
			ErrGenStream: "http 404: error Not Found",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.ProviderGen {
		return &adapters.ProviderGenDocToGen{ProviderGenDoc: getClientInner(t, apiKey, model)}
	}
	internaltest.TestClient_ProviderGen_errors(t, f, data)
}

func getClient(t *testing.T, m string) *Client {
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *Client {
	if apiKey == "" && os.Getenv("BFL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
	if err != nil {
		t.Fatal(err)
	}
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
	// Speed up the test.
	waitForPoll = time.Millisecond
}
