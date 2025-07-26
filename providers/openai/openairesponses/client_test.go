// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openairesponses_test

import (
	"context"
	"errors"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openai/openairesponses"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("OPENAI_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := openairesponses.New(apiKey, model.Model, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
		return &injectOption{
			Client: c,
			opts: openairesponses.OptionsText{
				// This will lead to spurious HTTP 500 but it is 25% of the cost.
				ServiceTier:     openairesponses.ServiceTierFlex,
				ReasoningEffort: openairesponses.ReasoningEffortLow,
			},
		}
	}
	if model.Model == "gpt-image-1" {
		return &imageClient{Client: c}
	}
	return c
}

func TestClient_Scoreboard(t *testing.T) {
	genaiModels, err := getClient(t, "").ListModels(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	var models []scoreboardtest.Model
	for _, m := range genaiModels {
		id := m.GetID()
		models = append(models, scoreboardtest.Model{Model: id, Thinking: strings.HasPrefix(id, "o") && !strings.Contains(id, "moderation")})
	}
	scoreboardtest.AssertScoreboard(t, getClientRT, models, testRecorder.Records)
}

type injectOption struct {
	*openairesponses.Client
	opts openairesponses.OptionsText
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

// imageClient only exposes GenDoc to save on costs.
type imageClient struct {
	*openairesponses.Client
}

func (i *imageClient) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("disabled to save on costs")
}

func (i *imageClient) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("disabled to save on costs")
}

func (i *imageClient) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	// TODO: Specify quality "low"
	// TODO: Test "jpeg" and "webp".
	return i.Client.GenDoc(ctx, msg, opts)
}

/*
// This is a tricky test since batch operations can take up to 24h to complete.
func TestClient_Batch(t *testing.T) {
	ctx := t.Context()
	c := getClient(t, "gpt-3.5-turbo")
	// Using an extremely old cheap model that nobody uses helps a lot on reducing the latency, I got it to work
	// within a few minutes.
	msgs := genai.Messages{genai.NewTextMessage(genai.User, "Tell a joke in 10 words")}
	job, err := c.GenAsync(ctx, msgs, nil)
	if err != nil {
		t.Fatal(err)
	}
	// TODO: Detect when recording and sleep only in this case.
	is_recording := os.Getenv("RECORD") == "1"
	for {
		res, err := c.PokeResult(ctx, job)
		if err != nil {
			t.Fatal(err)
		}
		if res.FinishReason == genai.Pending {
			if is_recording {
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
*/

func TestClient_Preferred(t *testing.T) {
	data := []struct {
		name string
		want string
	}{
		{base.PreferredCheap, "gpt-4.1-nano"},
		{base.PreferredGood, "gpt-4.1-mini"},
		{base.PreferredSOTA, "o3-pro"},
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
			Model:        "gpt-4.1-nano",
			ErrGenSync:   "http 401: openai responses error: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
			ErrGenStream: "http 401: openai responses error: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
			ErrGenDoc:    "http 401: openai responses error: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
			ErrListModel: "http 401: openai responses error: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys. (type: invalid_request_error, code: invalid_api_key)",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 400: openai responses error: The requested model 'bad model' does not exist. (type: invalid_request_error, code: model_not_found)",
			ErrGenStream: "http 400: openai responses error: The requested model 'bad model' does not exist. (type: invalid_request_error, code: model_not_found)",
			ErrGenDoc:    "http 400: openai responses error: Invalid value: 'bad model'. Supported values are: 'gpt-image-1', 'gpt-image-0721-mini-alpha', 'dall-e-2', and 'dall-e-3'. (type: invalid_request_error, code: invalid_value)",
		},
	}
	f := func(t *testing.T, apiKey, model string) genai.Provider {
		return getClientInner(t, apiKey, model)
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *openairesponses.Client {
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *openairesponses.Client {
	if apiKey == "" && os.Getenv("OPENAI_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := openairesponses.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
}
