// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package bfl_test

import (
	"context"
	_ "embed"
	"net/http"
	"os"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/bfl"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, opts genai.ProviderOptions, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if opts.APIKey == "" && os.Getenv("BFL_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	return bfl.New(t.Context(), &opts, fn)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})

	getClient := func(t *testing.T, m string) genai.Provider {
		t.Parallel()
		ci, err := getClientInner(t, genai.ProviderOptions{Model: m}, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Scoreboard", func(t *testing.T) {
		// bfl does not have a public API to list models.
		sb := getClient(t, genai.ModelNone).Scoreboard()
		var models []smoketest.Model
		for _, sc := range sb.Scenarios {
			for _, model := range sc.Models {
				models = append(models, smoketest.Model{Model: model})
			}
		}
		getClientRT := func(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := genai.ProviderOptions{Model: model.Model}
			if os.Getenv("BFL_API_KEY") == "" {
				opts.APIKey = "<insert_api_key_here>"
			}
			c, err := bfl.New(t.Context(), &opts, fn)
			if err != nil {
				t.Fatal(err)
			}
			if model.Reason {
				t.Fatal("unexpected")
			}
			return &imageModelClient{c}
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		data := []struct {
			name string
			want string
		}{
			{genai.ModelCheap, "flux-dev"},
			{genai.ModelGood, "flux-pro-1.1"},
			{genai.ModelSOTA, "flux-pro-1.1-ultra"},
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
					Model:  "flux-dev",
				},
				ErrGenSync:   "http 403\nNot authenticated - Invalid Authentication",
				ErrGenStream: "http 403\nNot authenticated - Invalid Authentication",
			},
			{
				Name: "bad model",
				Opts: genai.ProviderOptions{
					Model: "bad model",
				},
				ErrGenSync:   "http 404\nNot Found",
				ErrGenStream: "http 404\nNot Found",
			},
		}
		f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
			opts.OutputModalities = genai.Modalities{genai.ModalityImage}
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClient_Provider_errors(t, f, data)
	})
}

type imageModelClient struct {
	*bfl.Client
}

func (i *imageModelClient) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	for i := range opts {
		if v, ok := opts[i].(*genai.OptionsImage); ok {
			// Ask for a smaller size.
			n := *v
			n.Width = 256
			n.Height = 256
			n.PollInterval = time.Millisecond
			opts[i] = &n
		}
	}
	return i.Client.GenSync(ctx, msgs, opts...)
}

func init() {
	internal.BeLenient = false
}
