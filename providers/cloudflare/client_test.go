// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

import (
	"net/http"
	"os"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/internal/myrecorder"
	"github.com/maruel/genai/providers/cloudflare"
	"github.com/maruel/genai/smoke/smoketest"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func getClientRT(t testing.TB, model smoketest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("CLOUDFLARE_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	accountID := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
	if accountID == "" {
		accountID = "ACCOUNT_ID"
	}
	opts := genai.ProviderOptions{
		APIKey:          apiKey,
		AccountID:       accountID,
		Model:           model.Model,
		PreloadedModels: loadCachedModelsList(t),
	}
	c, err := cloudflare.New(t.Context(), &opts, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
		t.Fatal("implement me")
	}
	return c
}

func TestClient_Scoreboard(t *testing.T) {
	// Cloudflare hosts a ton of useless models, so just get the ones already in the scoreboard.
	sb := getClient(t, genai.ModelNone).Scoreboard()
	var models []smoketest.Model
	for _, sc := range sb.Scenarios {
		for _, model := range sc.Models {
			models = append(models, smoketest.Model{Model: model})
		}
	}
	smoketest.Run(t, getClientRT, models, testRecorder.Records)
}

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{genai.ModelCheap, "@cf/meta/llama-3.2-1b-instruct"},
		{genai.ModelGood, "@cf/meta/llama-3.3-70b-instruct-fp8-fast"},
		{genai.ModelSOTA, "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b"},
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
				Model:  "@hf/nousresearch/hermes-2-pro-mistral-7b",
			},
			ErrGenSync:   "http 401\nAuthentication error\nget a new API key at https://dash.cloudflare.com/profile/api-tokens",
			ErrGenStream: "http 401\nAuthentication error\nget a new API key at https://dash.cloudflare.com/profile/api-tokens",
			ErrListModel: "http 400\nUnable to authenticate request",
		},
		{
			Name: "bad model",
			Opts: genai.ProviderOptions{
				Model: "bad model",
			},
			ErrGenSync:   "http 400\nNo route for that URI",
			ErrGenStream: "http 400\nNo route for that URI",
		},
	}
	f := func(t *testing.T, opts genai.ProviderOptions) (genai.Provider, error) {
		opts.OutputModalities = genai.Modalities{genai.ModalityText}
		return getClientInner(t, opts)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *cloudflare.Client {
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

func getClientInner(t *testing.T, opts genai.ProviderOptions) (*cloudflare.Client, error) {
	if opts.APIKey == "" && os.Getenv("CLOUDFLARE_API_KEY") == "" {
		opts.APIKey = "<insert_api_key_here>"
	}
	if opts.AccountID == "" && os.Getenv("CLOUDFLARE_ACCOUNT_ID") == "" {
		opts.AccountID = "ACCOUNT_ID"
	}
	wrapper := func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.Record(t, h, recorder.WithHook(trimRecordingInternal, recorder.AfterCaptureHook), recorder.WithMatcher(matchCassetteInternal))
	}
	return cloudflare.New(t.Context(), &opts, wrapper)
}

// trimRecording trims API key and noise from the recording.
func trimRecordingInternal(i *cassette.Interaction) error {
	// Zap the account ID from the URL path before saving.
	i.Request.URL = reAccount.ReplaceAllString(i.Request.URL, "/accounts/ACCOUNT_ID/")
	return nil
}

func matchCassetteInternal(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	// When matching, ignore the account ID from the URL path.
	r.URL.Path = reAccount.ReplaceAllString(r.URL.Path, "/accounts/ACCOUNT_ID/")
	return myrecorder.DefaultMatcher(r, i)
}

func loadCachedModelsList(t testing.TB) []genai.Model {
	doOnce.Do(func() {
		var r myrecorder.Recorder
		var err2 error
		ctx := t.Context()
		opts := genai.ProviderOptions{Model: genai.ModelNone}
		if os.Getenv("CLOUDFLARE_API_KEY") == "" {
			opts.APIKey = "<insert_api_key_here>"
		}
		if os.Getenv("CLOUDFLARE_ACCOUNT_ID") == "" {
			opts.AccountID = "ACCOUNT_ID"
		}
		c, err := cloudflare.New(ctx, &opts, func(h http.RoundTripper) http.RoundTripper {
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
