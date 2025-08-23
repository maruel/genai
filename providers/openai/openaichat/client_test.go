// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package openaichat_test

import (
	"context"
	_ "embed"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openai/openaichat"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func getClientRT(t testing.TB, model scoreboardtest.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
	apiKey := ""
	if os.Getenv("OPENAI_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := openaichat.New(t.Context(), &genai.ProviderOptions{APIKey: apiKey, Model: model.Model}, fn)
	if err != nil {
		t.Fatal(err)
	}
	if model.Thinking {
		return &injectThinking{
			injectOption: injectOption{
				Client: c,
				opts: openaichat.OptionsText{
					// This will lead to spurious HTTP 500 but it is 25% of the cost.
					ServiceTier:     openaichat.ServiceTierFlex,
					ReasoningEffort: openaichat.ReasoningEffortLow,
				},
			},
		}
	}
	if model.Model == "gpt-image-1" {
		return &imageClient{Client: c}
	}
	return c
}

func TestClient_Scoreboard(t *testing.T) {
	genaiModels, err := getClient(t, genai.ModelNone).ListModels(t.Context())
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
	*openaichat.Client
	opts openaichat.OptionsText
}

func (i *injectOption) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		n.OptionsText = *opts.(*genai.OptionsText)
	}
	opts = &n
	return i.Client.GenSync(ctx, msgs, opts)
}

func (i *injectOption) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	n := i.opts
	if opts != nil {
		n.OptionsText = *opts.(*genai.OptionsText)
	}
	opts = &n
	return i.Client.GenStream(ctx, msgs, replies, opts)
}

func (i *injectOption) Unwrap() genai.Provider {
	return i.Client
}

// OpenAI returns the count of reasoning tokens but never return them. Duh. This messes up the scoreboard so
// inject fake thinking whitespace.
type injectThinking struct {
	injectOption
}

func (i *injectThinking) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	res, err := i.injectOption.GenSync(ctx, msgs, opts)
	if res.Usage.ReasoningTokens > 0 {
		res.Replies = append(res.Replies, genai.Reply{Thinking: "\n"})
	}
	return res, err
}

func (i *injectThinking) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	res, err := i.injectOption.GenStream(ctx, msgs, replies, opts)
	return res, err
}

// imageClient only exposes GenDoc to save on costs.
type imageClient struct {
	*openaichat.Client
}

func (i *imageClient) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("disabled to save on costs")
}

func (i *imageClient) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	return genai.Result{}, errors.New("disabled to save on costs")
}

func (i *imageClient) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	// TODO: Specify quality "low"
	// TODO: Test "jpeg" and "webp".
	return i.Client.GenDoc(ctx, msg, opts)
}

func (i *imageClient) Unwrap() genai.Provider {
	return i.Client
}

// This is a tricky test since batch operations can take up to 24h to complete.
func TestClient_Batch(t *testing.T) {
	ctx := t.Context()
	c := getClient(t, "gpt-3.5-turbo")
	// Using an extremely old cheap model that nobody uses helps a lot on reducing the latency, I got it to work
	// within a few minutes.
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
		modality genai.Modality
		name     string
		want     string
	}{
		{genai.ModalityText, genai.ModelCheap, "gpt-4.1-nano"},
		{genai.ModalityText, genai.ModelGood, "gpt-5-mini"},
		{genai.ModalityText, genai.ModelSOTA, "o3-pro"},
		{genai.ModalityImage, genai.ModelCheap, "dall-e-3"},
		{genai.ModalityImage, genai.ModelGood, "gpt-image-1"},
		{genai.ModalityImage, genai.ModelSOTA, "gpt-image-1"},
	}
	for _, line := range data {
		t.Run(line.name, func(t *testing.T) {
			t.Run(fmt.Sprintf("%s-%s", line.modality, line.name), func(t *testing.T) {
				opts := genai.ProviderOptions{Model: line.name, OutputModalities: genai.Modalities{line.modality}}
				c, err := getClientInner(t, &opts)
				if err != nil {
					t.Fatal(err)
				}
				if got := c.ModelID(); got != line.want {
					t.Fatalf("got model %q, want %q", got, line.want)
				}
			})
		})
	}
}

func TestClient_Provider_errors(t *testing.T) {
	data := []internaltest.ProviderError{
		{
			Name:         "bad apiKey",
			APIKey:       "bad apiKey",
			Model:        "gpt-4.1-nano",
			ErrGenSync:   "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			ErrGenStream: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			ErrGenDoc:    "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			ErrListModel: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
		},
		{
			Name:         "bad model",
			Model:        "bad model",
			ErrGenSync:   "http 400\ninvalid_request_error: invalid model ID",
			ErrGenStream: "http 400\ninvalid_request_error: invalid model ID",
			ErrGenDoc:    "http 400\ninvalid_request_error/invalid_value for \"model\": Invalid value: 'bad model'. Supported values are: 'gpt-image-1', 'gpt-image-1-io', 'gpt-image-0721-mini-alpha', 'dall-e-2', and 'dall-e-3'.",
		},
	}
	f := func(t *testing.T, apiKey, model string) (genai.Provider, error) {
		return getClientInner(t, &genai.ProviderOptions{APIKey: apiKey, Model: model, OutputModalities: genai.Modalities{genai.ModalityText}})
	}
	internaltest.TestClient_Provider_errors(t, f, data)
}

func getClient(t *testing.T, m string) *openaichat.Client {
	t.Parallel()
	c, err := getClientInner(t, &genai.ProviderOptions{Model: m})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func getClientInner(t *testing.T, opts *genai.ProviderOptions) (*openaichat.Client, error) {
	o := *opts
	if o.APIKey == "" && os.Getenv("OPENAI_API_KEY") == "" {
		o.APIKey = "<insert_api_key_here>"
	}
	return openaichat.New(t.Context(), &o, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
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
