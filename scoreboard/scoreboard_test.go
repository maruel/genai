// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package scoreboard_test

import (
	"context"
	"flag"
	"fmt"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/providers/cerebras"
	"github.com/maruel/genai/providers/deepseek"
	"github.com/maruel/genai/providers/groq"
	"github.com/maruel/genai/scoreboard/scoreboardtest"
)

func TestCreateScenario(t *testing.T) {
	totalUsage := genai.Usage{}
	for _, provider := range []string{"deepseek", "groq"} {
		t.Run(provider, func(t *testing.T) {
			providerUsage := genai.Usage{}
			cc := getClient(t, provider, t.Name()+"/ListModels", "")
			models, err2 := cc.(genai.ProviderModel).ListModels(t.Context())
			if err2 != nil {
				t.Fatal(err2)
			}
			for _, m := range models {
				id := m.GetID()
				t.Run(id, func(t *testing.T) {
					providerUsage.Add(scoreboardtest.RunOneModel(t, func(t testing.TB, sn string) genai.Provider {
						return getClient(t, provider, sn, id)
					}))
				})
			}
			t.Logf("Usage: %#v", providerUsage)
			totalUsage.Add(providerUsage)
		})
	}
	t.Logf("Usage: %#v", totalUsage)
}

func getClient(t testing.TB, provider, name, m string) genai.Provider {
	fn := func(h http.RoundTripper) http.RoundTripper {
		if name == "" {
			return h
		}
		r, err2 := recorder.Record(name, h)
		if err2 != nil {
			t.Fatal(err2)
		}
		t.Cleanup(func() {
			if err3 := r.Stop(); err3 != nil {
				t.Error(err3)
			}
		})
		return r
	}
	switch provider {
	case "cerebras":
		apiKey := ""
		if os.Getenv("CEREBRAS_API_KEY") == "" {
			apiKey = "<insert_api_key_here>"
		}
		c, err := cerebras.New(apiKey, m, fn)
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(m, "qwen") {
			return &adapters.ProviderGenThinking{
				ProviderGen: &adapters.ProviderGenAppend{
					ProviderGen: c,
					Append:      genai.NewTextMessage(genai.User, "/think"),
				},
				TagName: "think",
			}
		}
		return c
	case "deepseek":
		apiKey := ""
		if os.Getenv("DEEPSEEK_API_KEY") == "" {
			apiKey = "<insert_api_key_here>"
		}
		c, err := deepseek.New(apiKey, m, fn)
		if err != nil {
			t.Fatal(err)
		}
		return c
	case "groq":
		apiKey := ""
		if os.Getenv("GROQ_API_KEY") == "" {
			apiKey = "<insert_api_key_here>"
		}
		c, err := groq.New(apiKey, m, fn)
		if err != nil {
			t.Fatal(err)
		}
		if strings.HasPrefix(m, "qwen/") {
			return &handleGroqReasoning{
				Client: &adapters.ProviderGenAppend{
					ProviderGen: c,
					Append:      genai.NewTextMessage(genai.User, "/think"),
				},
				t: t,
			}
		}
		if m == "deepseek-r1-distill-llama-70b" {
			return &handleGroqReasoning{Client: c, t: t}
		}
		return c
	default:
		t.Fatal("implement me")
		return nil
	}
}

// handleGroqReasoning automatically enables the reasoning parsing feature of Groq.
type handleGroqReasoning struct {
	Client genai.ProviderGen
	t      testing.TB
}

func (h *handleGroqReasoning) Name() string {
	return h.Client.Name()
}

func (h *handleGroqReasoning) ModelID() string {
	return h.Client.ModelID()
}

func (h *handleGroqReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	if opts != nil {
		if o := opts.(*genai.OptionsText); len(o.Tools) != 0 || o.DecodeAs != nil || o.ReplyAsJSON {
			opts = &groq.OptionsText{ReasoningFormat: groq.ReasoningFormatParsed, OptionsText: *o}
			return h.Client.GenSync(ctx, msgs, opts)
		}
	}
	c := adapters.ProviderGenThinking{ProviderGen: h.Client, TagName: "think"}
	return c.GenSync(ctx, msgs, opts)
}

func (h *handleGroqReasoning) GenStream(ctx context.Context, msgs genai.Messages, replies chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	if opts != nil {
		if o := opts.(*genai.OptionsText); len(o.Tools) != 0 || o.DecodeAs != nil || o.ReplyAsJSON {
			opts = &groq.OptionsText{ReasoningFormat: groq.ReasoningFormatParsed, OptionsText: *o}
			return h.Client.GenStream(ctx, msgs, replies, opts)
		}
	}
	c := adapters.ProviderGenThinking{ProviderGen: h.Client, TagName: "think"}
	return c.GenStream(ctx, msgs, replies, opts)
}

func (h *handleGroqReasoning) Unwrap() genai.Provider {
	return h.Client
}

var recorder *internal.Records

func TestMain(m *testing.M) {
	var err error
	recorder, err = internal.NewRecords("testdata")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	flag.Parse()
	filtered := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "test.run" {
			filtered = true
		}
	})
	code := m.Run()
	if err = recorder.Close(); err != nil {
		if !filtered {
			fmt.Fprintf(os.Stderr, "%v\n", err)
			code = max(code, 1)
		}
	}
	os.Exit(code)
}

func init() {
	internal.BeLenient = false
}
