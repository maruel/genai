// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package bfl

import (
	"context"
	_ "embed"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("BFL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := New(apiKey, model.Model, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
		t.Fatal("unexpected")
	}
	return &imageModelClient{c}
}

func TestClient_Scoreboard(t *testing.T) {
	// bfl does not have a public API to list models.
	sb := getClient(t, "").Scoreboard()
	var models []scoreboardtest.Model
	for _, sc := range sb.Scenarios {
		if sc.GenDoc != nil {
			for _, model := range sc.Models {
				models = append(models, scoreboardtest.Model{Model: model})
			}
		}
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

type imageModelClient struct {
	*Client
}

func (i *imageModelClient) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	if v, ok := opts.(*genai.OptionsImage); ok {
		// Ask for a smaller size.
		n := *v
		n.Width = 256
		n.Height = 256
		opts = &n
	}
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

func TestClient_Provider_errors(t *testing.T) {
	data := []internaltest.ProviderError{
		{
			Name:      "bad apiKey",
			APIKey:    "bad apiKey",
			Model:     "flux-dev",
			ErrGenDoc: "http 403: error Not authenticated - Invalid Authentication",
		},
		{
			Name:      "bad model",
			Model:     "bad model",
			ErrGenDoc: "http 404: error Not Found",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.Provider {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
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
