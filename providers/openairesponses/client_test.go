// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the OpenAI Responses provider client.

package openairesponses_test

import (
	"errors"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/openaibase"
	"github.com/maruel/genai/providers/openairesponses"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, fn func(http.RoundTripper) http.RoundTripper, opts ...genai.ProviderOption) (genai.Provider, error) {
	hasAPIKey := false
	for _, opt := range opts {
		if _, ok := opt.(genai.ProviderOptionAPIKey); ok {
			hasAPIKey = true
			break
		}
	}
	if !hasAPIKey {
		key := internaltest.GetEnv("OPENAI_API_KEY")
		if key == "" {
			key = "<insert_api_key_here>"
		}
		opts = append(opts, genai.ProviderOptionAPIKey(key))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	c, err := openairesponses.New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	return c, err
}

func TestClient(t *testing.T) {
	t.Run("New", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("embedding modality", func(t *testing.T) {
				for _, id := range []string{"text-embedding-3-small", "gpt-6-luna", "unknown-model"} {
					t.Run(id, func(t *testing.T) {
						c, err := openairesponses.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel(id), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingNoTransport{t: t} }))
						if err != nil {
							t.Fatal(err)
						}
						internaltest.CleanupCloser(t, c)
						want := genai.Modalities{genai.ModalityText}
						if id == "text-embedding-3-small" {
							want = genai.Modalities{genai.ModalityEmbedding}
						}
						sb := c.Scoreboard()
						if !slices.Equal(c.OutputModalities(), want) || (sb.Embedding(id) != nil) != (id == "text-embedding-3-small") {
							t.Fatalf("modalities %v, qualification %+v", c.OutputModalities(), sb.Embedding(id))
						}
					})
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			t.Run("embedding preference", func(t *testing.T) {
				c, err := openairesponses.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ModelCheap, genai.ProviderOptionModalities{genai.ModalityEmbedding}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper { return embeddingNoTransport{t: t} }))
				if err == nil || c != nil {
					t.Fatalf("client %v, error %v", c, err)
				}
			})
		})
	})

	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	t.Run("CodexAuth", func(t *testing.T) { testCodexAuth(t, testRecorder) })
	cl, err2 := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
		return testRecorder.RecordWithName(t, t.Name()+"/Warmup", h)
	})
	if err2 != nil {
		t.Fatal(err2)
	}
	cachedModels, err2 := cl.ListModels(t.Context())
	if err2 != nil {
		t.Fatal(err2)
	}
	getClient := func(t *testing.T, m string) genai.Provider {
		t.Parallel()
		opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
		if m != "" {
			opts = append(opts, genai.ProviderOptionModel(m))
		}
		ci, err := getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		}, opts...)
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, ""))
	})

	t.Run("ConnectRealtime", func(t *testing.T) {
		c := getClient(t, "gpt-realtime-2.1")
		for _, tc := range []struct {
			name string
			opts []genai.RealtimeOption
		}{
			{name: "nil", opts: []genai.RealtimeOption{nil}},
			{name: "image", opts: []genai.RealtimeOption{genai.Modalities{genai.ModalityImage}}},
			{name: "endpoint", opts: []genai.RealtimeOption{&openairesponses.RealtimeOption{Endpoint: "https://example.com"}}},
			{name: "duplicate", opts: []genai.RealtimeOption{&openairesponses.RealtimeOption{}, &openairesponses.RealtimeOption{}}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				s, err := c.ConnectRealtime(t.Context(), tc.opts...)
				if s != nil || err == nil {
					t.Fatalf("session %v, error %v", s, err)
				}
			})
		}
	})

	t.Run("Scoreboard", func(t *testing.T) {
		c := getClient(t, "")
		genaiModels, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		scenarios := c.Scoreboard().Models()
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			id := m.GetID()
			if strings.HasPrefix(id, "gpt-5") {
				continue
			}
			found := false
			for _, sc := range scenarios {
				if sc.Model == id {
					models = append(models, scoreboard.Model{Model: id, Reason: sc.Reason})
					found = true
				}
			}
			if !found {
				reason := (strings.HasPrefix(id, "gpt-6") || strings.HasPrefix(id, "o")) && !strings.Contains(id, "moderation")
				models = append(models, scoreboard.Model{Model: id, Reason: reason})
			}
		}
		if !slices.Contains(models, scoreboard.Model{Model: "text-embedding-3-small"}) {
			models = append(models, scoreboard.Model{Model: "text-embedding-3-small"})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			key := internaltest.GetEnv("OPENAI_API_KEY")
			if key == "" {
				key = "<insert_api_key_here>"
			}
			opts = append(opts, genai.ProviderOptionAPIKey(key))
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
			}
			c, err := openairesponses.New(t.Context(), opts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			if strings.Contains(model.Model, "realtime") {
				rec := &internaltest.RealtimeRecorder{
					Provider: c,
					T:        t,
					Target:   "wss://api.openai.com/v1/realtime",
					Dir:      filepath.Join("testdata", filepath.FromSlash(t.Name())),
					Updating: internaltest.UpdatingRecordings(),
					Mutation: func(opts []genai.RealtimeOption, endpoint string) []genai.RealtimeOption {
						out := slices.Clone(opts)
						for i, opt := range out {
							if o, ok := opt.(*openairesponses.RealtimeOption); ok {
								clone := *o
								clone.Endpoint = endpoint
								out[i] = &clone
								return out
							}
						}
						return append(out, &openairesponses.RealtimeOption{Endpoint: endpoint})
					},
				}
				effort := genai.EffortOff
				if model.Reason {
					effort = genai.EffortHigh
				}
				return &internaltest.InjectRealtimeOptions{
					Provider: rec,
					Opts:     []genai.RealtimeOption{effort},
				}
			}
			// This will lead to spurious HTTP 500 but it is 25% of the cost.
			tier := openairesponses.ServiceTierFlex
			res := genai.EffortLow
			if model.Reason {
				var p genai.Provider = &internaltest.InjectOptions{
					Provider: c,
					Opts: []genai.GenOption{
						&openairesponses.GenOptionText{
							ServiceTier: tier,
						}, res,
					},
				}
				if tier == openairesponses.ServiceTierFlex {
					p = &internaltest.RetryOnRateLimit{
						Provider: p,
						FallbackOpts: []genai.GenOption{
							&openairesponses.GenOptionText{ServiceTier: openairesponses.ServiceTierDefault},
						},
					}
				}
				return p
			}
			if slices.Equal(c.OutputModalities(), []genai.Modality{genai.ModalityText}) {
				// See https://platform.openai.com/docs/guides/flex-processing
				if id := c.ModelID(); id == "o3" || id == "o4-mini" || strings.HasPrefix(id, "gpt-6") {
					var p genai.Provider = &internaltest.InjectOptions{
						Provider: c,
						Opts:     []genai.GenOption{&openairesponses.GenOptionText{ServiceTier: tier}},
					}
					if tier == openairesponses.ServiceTierFlex {
						p = &internaltest.RetryOnRateLimit{
							Provider: p,
							FallbackOpts: []genai.GenOption{
								&openairesponses.GenOptionText{ServiceTier: openairesponses.ServiceTierDefault},
							},
						}
					}
					return p
				}
			}
			return c
		}
		smoketest.Run(t, smoke.HTTPSession(testRecorder.Records, getClientRT), models, &smoketest.RunOptions{RealtimeVoices: openaibase.RealtimeVoices(), Qualify: []scoreboard.Model{{Model: "text-embedding-3-small"}}})

		// The Responses API travels over WebSocket too. One model proves the wire; the
		// rows of the other models stay with the HTTP run because a row records its
		// wire. The socket always carries reasoning summaries, so its row is a
		// thinking row.
		t.Run("WebSocket", func(t *testing.T) {
			getClientWS := func(t testing.TB, model scoreboard.Model, name string) genai.Provider {
				key := internaltest.GetEnv("OPENAI_API_KEY")
				if key == "" {
					key = "<insert_api_key_here>"
				}
				opts := []genai.ProviderOption{
					genai.ProviderOptionAPIKey(key),
					genai.ProviderOptionPreloadedModels(cachedModels),
				}
				if model.Model != "" {
					opts = append(opts, genai.ProviderOptionModel(model.Model))
				}
				if name == "" {
					// The harness asks a model what it can do before it records; the
					// answer does not need a socket.
					c, err := openairesponses.New(t.Context(), opts...)
					if err != nil {
						t.Fatal(err)
					}
					internaltest.CleanupCloser(t, c)
					return c
				}
				proxy := internaltest.NewWebSocketProxy(t, name, "https://api.openai.com/v1")
				c, err := openairesponses.New(t.Context(), append(opts, genai.ProviderOptionRemote(proxy.URL()))...)
				if err != nil {
					t.Fatal(err)
				}
				internaltest.CleanupCloser(t, c)
				ws, err := c.WebSocket(t.Context())
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := ws.Close(); err != nil {
						t.Error(err)
					}
				})
				return ws
			}
			m := scoreboard.Model{Model: "gpt-6-luna", Reason: true}
			smoketest.Run(t, smoke.WebSocket(getClientWS), []scoreboard.Model{m}, &smoketest.RunOptions{Qualify: []scoreboard.Model{m}})
		})
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			opts := []genai.ProviderOption{
				genai.ProviderOptionModalities{modality},
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model != "" {
				opts = append(opts, genai.ProviderOptionModel(model))
			}
			return getClientInner(st, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			}, opts...)
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, string(genai.ModelCheap))
		})
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("gpt-6-luna"),
				},
				ErrGenSync:   "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrGenStream: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrListModel: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			},
			{
				Name: "bad apiKey image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("bad apiKey"),
					genai.ProviderOptionModel("gpt-image-1"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
				ErrGenStream: "http 401\ninvalid_request_error/invalid_api_key: Incorrect API key provided: bad apiKey. You can find your API key at https://platform.openai.com/account/api-keys.",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\ninvalid_request_error/model_not_found for \"model\": The requested model 'bad model' does not exist.",
				ErrGenStream: "http 400\ninvalid_request_error/model_not_found for \"model\": The requested model 'bad model' does not exist.",
			},
			{
				Name: "bad model image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 400\nimage_generation_user_error/invalid_value for \"model\": The model 'bad model' does not exist.",
				ErrGenStream: "http 400\nimage_generation_user_error/invalid_value for \"model\": The model 'bad model' does not exist.",
			},
			/* TODO:
			{
				Name: "audio not supported",
				Opts: []genai.ProviderOption{
					genai.ModelGood,
					genai.ProviderOptionModalities{genai.ModalityAudio},
				},
				ErrGenSync:   "OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio",
				ErrGenStream: "OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio",
			},
			*/
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			return getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			}, opts...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func TestPreviousResponseID(t *testing.T) {
	t.Run("empty_messages_without_prev_id", func(t *testing.T) {
		var req openairesponses.Response
		err := req.Init(nil, "gpt-6-luna")
		if err == nil {
			t.Fatal("expected error for empty messages without PreviousResponseID")
		}
		if !strings.Contains(err.Error(), "no messages provided") {
			t.Errorf("unexpected error: %v", err)
		}
	})
}

func TestWebSearchActions(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name string
			in   openairesponses.WebSearchAction
			want []genai.CitationSource
		}{
			{
				name: "find in page",
				in:   openairesponses.WebSearchAction{Type: openairesponses.WebSearchActionFindInPage, URL: "https://go.dev/dl/", Pattern: "go1.26"},
				want: []genai.CitationSource{{Type: genai.CitationWeb, URL: "https://go.dev/dl/"}},
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				m := openairesponses.Message{
					Type:   openairesponses.MessageWebSearchCall,
					Status: "completed",
					Action: tt.in,
				}
				res, err := (&openairesponses.Response{Output: []openairesponses.Message{m}}).ToResult()
				if err != nil {
					t.Fatal(err)
				}
				if err := res.Validate(); err != nil {
					t.Fatal(err)
				}
				if len(res.Replies) != 1 {
					t.Fatalf("replies = %d, want 1", len(res.Replies))
				}
				assertCitationSources(t, res.Replies[0].Citation.Sources, tt.want)

				fragments, finish := openairesponses.ProcessStream(slices.Values([]openairesponses.ResponseStreamChunkResponse{{
					Type: openairesponses.ResponseOutputItemDone,
					Item: m,
				}}))
				got := slices.Collect(fragments)
				_, _, err = finish()
				if err != nil {
					t.Fatal(err)
				}
				if len(got) != 1 {
					t.Fatalf("fragments = %d, want 1", len(got))
				}
				if err := got[0].Validate(); err != nil {
					t.Fatal(err)
				}
				assertCitationSources(t, got[0].Citation.Sources, tt.want)
			})
		}
	})

	t.Run("error", func(t *testing.T) {
		m := openairesponses.Message{
			Type:   openairesponses.MessageWebSearchCall,
			Status: "completed",
			Action: openairesponses.WebSearchAction{Type: "visit"},
		}
		if _, err := (&openairesponses.Response{Output: []openairesponses.Message{m}}).ToResult(); err == nil || err.Error() != `implement action type "visit"` {
			t.Fatalf("ToResult() error = %v, want implement action type", err)
		}
		fragments, finish := openairesponses.ProcessStream(slices.Values([]openairesponses.ResponseStreamChunkResponse{{
			Type: openairesponses.ResponseOutputItemDone,
			Item: m,
		}}))
		for range fragments {
		}
		if _, _, err := finish(); err == nil || err.Error() != `implement action type "visit"` {
			t.Fatalf("ProcessStream() error = %v, want implement action type", err)
		}
	})
}

func assertCitationSources(t *testing.T, got, want []genai.CitationSource) {
	if len(got) != len(want) {
		t.Fatalf("sources = %#v, want %#v", got, want)
	}
	for i := range got {
		if got[i].Type != want[i].Type || got[i].URL != want[i].URL || got[i].Snippet != want[i].Snippet {
			t.Errorf("source %d = %#v, want %#v", i, got[i], want[i])
		}
	}
}

func init() {
	internal.BeLenient = false
}

type embeddingNoTransport struct{ t *testing.T }

func (e embeddingNoTransport) RoundTrip(*http.Request) (*http.Response, error) {
	e.t.Fatal("unexpected HTTP request")
	return nil, errors.New("unexpected HTTP request")
}
