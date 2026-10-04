// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Cloudflare Workers AI provider client.

package cloudflare_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cloudflare"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
)

func getClientInner(t *testing.T, fn func(http.RoundTripper) http.RoundTripper, opts ...genai.ProviderOption) (genai.Provider, error) {
	hasAPIKey := false
	hasAccountID := false
	for _, opt := range opts {
		switch opt.(type) {
		case genai.ProviderOptionAPIKey:
			hasAPIKey = true
		case cloudflare.AccountID:
			hasAccountID = true
		}
	}
	if !hasAPIKey {
		key := internaltest.GetEnv("CLOUDFLARE_API_KEY")
		if key == "" {
			key = "<insert_api_key_here>"
		}
		opts = append(opts, genai.ProviderOptionAPIKey(key))
	}
	if !hasAccountID {
		id := internaltest.GetEnv("CLOUDFLARE_ACCOUNT_ID")
		if id == "" {
			id = "ACCOUNT_ID"
		}
		opts = append(opts, cloudflare.AccountID(id))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	c, err := cloudflare.New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	return c, err
}

func TestClient(t *testing.T) {
	t.Run("SystemOne", func(t *testing.T) {
		t.Run("CLEF", testClientDecisionGeneration)
		t.Run("errors", testClientDecisionErrors)
		t.Run("cancellation", testClientDecisionCancellation)
	})
	t.Run("GenSync", testClientDecisionChat)
	t.Run("SystemOneRaw", testClientDecisionRaw)
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
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

	t.Run("Scoreboard", func(t *testing.T) {
		genaiModels, err := getClient(t, "").ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		models := make([]scoreboard.Model, 0, len(genaiModels))
		for _, m := range genaiModels {
			id := m.GetID()
			models = append(models, scoreboard.Model{Model: id, Reason: strings.Contains(id, "deepseek-r1")})
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			key := internaltest.GetEnv("CLOUDFLARE_API_KEY")
			if key == "" {
				key = "<insert_api_key_here>"
			}
			opts = append(opts, genai.ProviderOptionAPIKey(key))
			id := internaltest.GetEnv("CLOUDFLARE_ACCOUNT_ID")
			if id == "" {
				id = "ACCOUNT_ID"
			}
			opts = append(opts, cloudflare.AccountID(id))
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
			}
			c, err := cloudflare.New(t.Context(), opts...)
			if c != nil {
				internaltest.CleanupCloser(t, c)
			}
			if err != nil {
				t.Fatal(err)
			}
			if model.Reason {
				// Check if it has predefined thinking tokens.
				for _, sc := range c.Scoreboard().Scenarios {
					if sc.Reason && slices.Contains(sc.Models, model.Model) {
						if sc.ReasoningTokenEnd != "" {
							return &adapters.ProviderReasoning{
								Provider:            c,
								ReasoningTokenStart: sc.ReasoningTokenStart,
								ReasoningTokenEnd:   sc.ReasoningTokenEnd,
							}
						}
						break
					}
				}
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records, nil)
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
					genai.ProviderOptionModel("@hf/nousresearch/hermes-2-pro-mistral-7b"),
				},
				ErrGenSync:   "http 401\nAuthentication error\nget a new API key at https://dash.cloudflare.com/profile/api-tokens",
				ErrGenStream: "http 401\nAuthentication error\nget a new API key at https://dash.cloudflare.com/profile/api-tokens",
				ErrListModel: "http 400\nUnable to authenticate request",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\nNo route for that URI",
				ErrGenStream: "http 400\nNo route for that URI",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			return getClientInner(t, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			}, opts...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func init() {
	internal.BeLenient = false
}

type clefTransport struct {
	target    *url.URL
	transport http.RoundTripper
}

func (c clefTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	req := r.Clone(r.Context())
	req.URL.Scheme, req.URL.Host = c.target.Scheme, c.target.Host
	return c.transport.RoundTrip(req)
}

func clefClient(t *testing.T, model string, h http.HandlerFunc) *cloudflare.Client {
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}
	opts := []genai.ProviderOption{cloudflare.AccountID("account"), genai.ProviderOptionAPIKey("fake-key"), genai.ProviderOptionModel(model), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
		return clefTransport{target: u, transport: srv.Client().Transport}
	})}
	c, err := cloudflare.New(t.Context(), opts...)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Error(err)
		}
	})
	return c
}

const clefResponse = `{"success":true,"errors":[],"messages":[],"result":{"model":"clef","answers":{"billing":{"type":"noul","noul":0.9},"route":{"type":"choice","choice":"billing","confidence":0.8,"probabilities":{"billing":0.9,"support":0.1}},"urgency":{"type":"score","score":0.75,"confidence":0.5,"legend":{"0":"low","1":{"level":"high"}},"probabilities":{"0":0.25,"1":0.75}}},"usage":{"input_tokens":42,"output_tokens":0}}}`

func clefQuestions() genai.Questions {
	return genai.Questions{
		"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this about billing?")},
		"route":   {Type: genai.QuestionChoice, Instructions: genai.Object{"question": "Which team?"}, Choice: map[string]genai.DecisionContent{"billing": nil, "support": nil}},
		"urgency": {Type: genai.QuestionScore, Instructions: genai.Array{"How urgent?"}, Score: []genai.DecisionContent{genai.Text("low"), genai.Object{"level": "high"}}},
	}
}

func testClientDecisionGeneration(t *testing.T) {
	t.Run("reasoning usage", func(t *testing.T) {
		c := clefClient(t, "@cf/cloudflare/clef", func(w http.ResponseWriter, _ *http.Request) {
			body := strings.Replace(clefResponse, `"output_tokens":0`, `"output_tokens":3,"reasoning_tokens":2`, 1)
			if _, err := w.Write([]byte(body)); err != nil {
				t.Error(err)
			}
		})
		res, err := c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("x"), Questions: clefQuestions()})
		if err != nil {
			t.Fatal(err)
		}
		if res.Usage.ReasoningTokens != 2 || res.Usage.InputTokens != 42 || res.Usage.OutputTokens != 3 {
			t.Fatalf("unexpected usage: %+v", res.Usage)
		}
	})
	for _, tc := range []struct{ name, model, selector, path string }{
		{"clef", "@cf/cloudflare/clef", "clef", "@cf/cloudflare/clef"},
		{"clef-flash", "@cf/cloudflare/clef-flash", "clef-flash", "@cf/cloudflare/clef-flash"},
		{"future provider", "@hf/future/decision-v2", "decision-v2", "@hf/future/decision-v2"},
		{"escaped future model", "@cf/cloudflare/future?revision=2#%", "future?revision=2#%", "@cf/cloudflare/future%3Frevision=2%23%25"},
		{"dot route component", "@hf/future/../decision-v2", "decision-v2", "@hf/future/%2E%2E/decision-v2"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Run("valid", func(t *testing.T) {
				q := genai.Questions{
					"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this about billing?")},
					"route":   {Type: genai.QuestionChoice, Instructions: genai.Object{"question": "Which team?"}, Choice: map[string]genai.DecisionContent{"billing": nil, "support": nil}},
					"urgency": {Type: genai.QuestionScore, Instructions: genai.Array{"How urgent?"}, Score: []genai.DecisionContent{genai.Text("low"), genai.Object{"level": "high"}}},
				}
				calls := 0
				c := clefClient(t, tc.model, func(w http.ResponseWriter, r *http.Request) {
					calls++
					if r.Method != http.MethodPost || r.RequestURI != "/client/v4/accounts/account/ai/run/"+tc.path || r.URL.RawQuery != "" || r.Header.Get("Authorization") != "Bearer fake-key" {
						t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
					}
					var req struct {
						Model     string                     `json:"model"`
						State     map[string]int             `json:"state"`
						Questions map[string]json.RawMessage `json:"questions"`
						Images    []string                   `json:"images"`
						Stream    bool                       `json:"stream"`
					}
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						t.Error(err)
					}
					if req.Model != tc.selector || req.State["ticket"] != 42 || len(req.Questions) != 3 || len(req.Images) != 1 || req.Stream {
						t.Errorf("unexpected request %+v", req)
					}
					for id, want := range map[string]string{
						"billing": `{"type":"noul","instructions":"Is this about billing?"}`,
						"route":   `{"type":"choice","instructions":{"question":"Which team?"},"criteria":{"billing":null,"support":null}}`,
						"urgency": `{"type":"score","instructions":["How urgent?"],"criteria":["low",{"level":"high"}]}`,
					} {
						if string(req.Questions[id]) != want {
							t.Errorf("question %s: got %s, want %s", id, req.Questions[id], want)
						}
					}
					w.Header().Set("Content-Type", "application/json")
					if _, err := w.Write([]byte(clefResponse)); err != nil {
						t.Error(err)
					}
				})
				req := genai.SystemOneRequest{State: genai.Object{"ticket": 42}, Questions: q, Docs: []genai.Doc{{Filename: "image.png", Src: bytes.NewReader(clefPNG(t))}}}
				res, err := c.SystemOne(t.Context(), &req)
				if err != nil {
					t.Fatal(err)
				}
				if calls != 1 {
					t.Errorf("calls=%d", calls)
				}

				if res.Answers["billing"].Noul != 0.9 || res.Answers["route"].Choice != "billing" || res.Answers["urgency"].Score != 0.75 || !reflect.DeepEqual(res.Answers["urgency"].Legend["1"], genai.Object{"level": "high"}) {
					t.Errorf("unexpected answers %+v", res.Answers)
				}
				if res.Usage.InputTokens != 42 || res.Usage.OutputTokens != 0 {
					t.Errorf("unexpected usage %+v", res.Usage)
				}
			})
		})
	}
}

func testClientDecisionCancellation(t *testing.T) {
	t.Run("cancellation", func(t *testing.T) {
		c := clefClient(t, "@cf/cloudflare/clef", func(http.ResponseWriter, *http.Request) { t.Error("unexpected HTTP request") })
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		_, err := c.SystemOne(ctx, &genai.SystemOneRequest{State: genai.Text("x"), Questions: clefQuestions()})
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("got %v, want cancellation", err)
		}
	})
}

func testClientDecisionRaw(t *testing.T) {
	t.Run("future selector escaped independently", func(t *testing.T) {
		for _, tc := range []struct{ model, selector, path string }{
			{"future?revision=2#%/branch", "future?revision=2#%/branch", "@cf/cloudflare/future%3Frevision=2%23%25%2Fbranch"},
			{"..", "..", "@cf/cloudflare/%2E%2E"},
			{"@hf/future/decision-v2", "decision-v2", "@hf/future/decision-v2"},
		} {
			t.Run(tc.selector, func(t *testing.T) {
				c := clefClient(t, "@hf/other/configured-model", func(w http.ResponseWriter, r *http.Request) {
					if r.RequestURI != "/client/v4/accounts/account/ai/run/"+tc.path || r.URL.RawQuery != "" {
						t.Errorf("unsafe or configured route: %s", r.RequestURI)
					}
					var body struct {
						Model string `json:"model"`
					}
					if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
						t.Error(err)
					}
					if body.Model != tc.selector {
						t.Errorf("body selector: %q", body.Model)
					}
					if _, err := w.Write([]byte(clefResponse)); err != nil {
						t.Error(err)
					}
				})
				var out cloudflare.SystemOneResponse
				if err := c.SystemOneRaw(t.Context(), &cloudflare.SystemOneRequest{Model: " " + tc.model + " ", State: genai.Text("state"), Questions: clefQuestions()}, &out); err != nil {
					t.Fatal(err)
				}
			})
		}
	})
	t.Run("rounded probabilities", func(t *testing.T) {
		c := clefClient(t, "@cf/chat", func(w http.ResponseWriter, _ *http.Request) {
			if _, err := w.Write([]byte(`{"success":true,"errors":[],"messages":[],"result":{"model":"clef-flash","answers":{"rating":{"type":"score","score":2.05,"confidence":0.5,"legend":{"0":"none","1":"low","2":"medium","3":"high"},"probabilities":{"0":0,"1":0.14,"2":0.68,"3":0.17}}},"usage":{"input_tokens":10,"output_tokens":0}}}`)); err != nil {
				t.Error(err)
			}
		})
		q := genai.Questions{"rating": {Type: genai.QuestionScore, Instructions: genai.Text("rate"), Score: []genai.DecisionContent{genai.Text("none"), genai.Text("low"), genai.Text("medium"), genai.Text("high")}}}
		var out cloudflare.SystemOneResponse
		if err := c.SystemOneRaw(t.Context(), &cloudflare.SystemOneRequest{Model: "clef-flash", State: genai.Text("state"), Questions: q}, &out); err != nil {
			t.Fatal(err)
		}
		res := genai.SystemOneResponse{}
		if err := out.To(&res); err != nil {
			t.Fatal(err)
		}
		if res.Answers["rating"].Probabilities["3"] != 0.17 {
			t.Fatalf("rounded probabilities lost: %+v", res.Answers)
		}
	})
	t.Run("SystemOneRaw", func(t *testing.T) {
		c := clefClient(t, "@cf/chat", func(w http.ResponseWriter, r *http.Request) {
			if !strings.HasSuffix(r.URL.Path, "/@cf/cloudflare/clef-flash") {
				t.Errorf("route %s", r.URL.Path)
			}
			var req struct {
				Model  string              `json:"model"`
				State  []any               `json:"state"`
				Images []map[string]string `json:"images"`
			}
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Error(err)
			}
			if req.Model != "clef-flash" || len(req.State) != 2 || len(req.Images) != 1 || req.Images[0]["content_type"] != "image/png" {
				t.Errorf("unexpected request %+v", req)
			}
			if _, err := w.Write([]byte(clefResponse)); err != nil {
				t.Error(err)
			}
		})
		var out cloudflare.SystemOneResponse
		if err := c.SystemOneRaw(t.Context(), &cloudflare.SystemOneRequest{Model: " clef-flash\n", State: genai.Array{"one", genai.Object{"ticket": 42}}, Questions: clefQuestions(), Images: []cloudflare.DecisionImage{{ContentType: "image/png", Base64: base64.StdEncoding.EncodeToString(clefPNG(t))}}}, &out); err != nil {
			t.Fatal(err)
		}
		if out.Result.Answers["billing"].Noul != 0.9 {
			t.Errorf("unexpected response %+v", out)
		}
	})
}

func testClientDecisionChat(t *testing.T) {
	for _, model := range []string{"@cf/chat", "@cf/cloudflare/clef", "@cf/cloudflare/clef-flash"} {
		t.Run(model, func(t *testing.T) {
			c := clefClient(t, model, func(w http.ResponseWriter, r *http.Request) {
				if !strings.HasSuffix(r.URL.Path, "/"+model) {
					t.Errorf("route %s", r.URL.Path)
				}
				var req map[string]json.RawMessage
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Error(err)
				}
				if req["messages"] == nil || req["questions"] != nil {
					t.Errorf("not a chat request %v", req)
				}
				if _, err := w.Write([]byte(`{"success":true,"errors":[],"messages":[],"result":{"response":"Hello","usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}}`)); err != nil {
					t.Error(err)
				}
			})
			res, err := c.GenSync(t.Context(), genai.Messages{genai.NewTextMessage("hello")})
			if err != nil {
				t.Fatal(err)
			}
			if res.String() != "Hello" {
				t.Errorf("got %q", res.String())
			}
		})
	}
}

func testClientDecisionErrors(t *testing.T) {
	t.Run("response errors", func(t *testing.T) {
		for _, tc := range []struct {
			name, body string
			status     int
		}{
			{"malformed", `{`, 200},
			{"failure no errors", `{"success":false,"errors":[],"result":null}`, 200},
			{"missing result", `{"success":true}`, 200},
			{"failure envelope", `{"success":false,"errors":[{"code":1,"message":"failed"}],"result":null}`, 200},
			{"failure with result", strings.Replace(strings.Replace(clefResponse, `"success":true`, `"success":false`, 1), `"errors":[]`, `"errors":[{"code":1,"message":"failed"}]`, 1), 200},
			{"missing noul", strings.Replace(clefResponse, `,"noul":0.9`, "", 1), 200},
			{"null noul", strings.Replace(clefResponse, `"noul":0.9`, `"noul":null`, 1), 200},
			{"out of range", strings.Replace(clefResponse, `"noul":0.9`, `"noul":2`, 1), 200},
			{"unknown type", strings.Replace(clefResponse, `"type":"noul"`, `"type":"other"`, 1), 200},
			{"mismatched type", strings.Replace(clefResponse, `"billing":{"type":"noul","noul":0.9}`, `"billing":{"type":"choice","choice":"x","probabilities":{"x":1},"confidence":1}`, 1), 200},
			{"missing question", strings.Replace(clefResponse, `"billing":{"type":"noul","noul":0.9},`, "", 1), 200},
			{"unknown choice", strings.Replace(clefResponse, `"choice":"billing"`, `"choice":"x"`, 1), 200},
			{"missing probabilities", strings.Replace(clefResponse, `"support":0.1`, `"wrong":0.1`, 1), 200},
			{"out of range probability", strings.Replace(clefResponse, `"support":0.1`, `"support":1.2`, 1), 200},
			{"null probability", strings.Replace(strings.Replace(clefResponse, `"support":0.1`, `"support":null`, 1), `"billing":0.9`, `"billing":1`, 1), 200},
			{"populated errors", strings.Replace(clefResponse, `"errors":[]`, `"errors":[{"code":1,"message":"failed"}]`, 1), 200},
			{"bad score", strings.Replace(clefResponse, `"score":0.75`, `"score":2`, 1), 200},
			{"bad legend", strings.Replace(clefResponse, `"0":"low"`, `"2":"low"`, 1), 200},
			{"missing usage", strings.Replace(clefResponse, `"input_tokens":42,`, "", 1), 200},
			{"negative reasoning", strings.Replace(clefResponse, `"input_tokens":42,`, `"input_tokens":42,"reasoning_tokens":-1,`, 1), 200},
			{"HTTP error", `{"success":false,"errors":[{"code":1000,"message":"bad request"}],"messages":[]}`, 400},
		} {
			t.Run(tc.name, func(t *testing.T) {
				c := clefClient(t, "@cf/cloudflare/clef", func(w http.ResponseWriter, _ *http.Request) {
					w.WriteHeader(tc.status)
					if _, err := w.Write([]byte(tc.body)); err != nil {
						t.Error(err)
					}
				})
				_, err := c.SystemOne(t.Context(), &genai.SystemOneRequest{State: genai.Text("x"), Questions: clefQuestions()})
				if err == nil {
					t.Fatal("expected error")
				}
				if tc.status == 400 || strings.HasPrefix(tc.name, "failure") || tc.name == "populated errors" {
					api, ok := errors.AsType[*cloudflare.ErrorResponse](err)
					if !ok {
						t.Fatalf("expected API error: %v", err)
					}
					if tc.name == "failure no errors" {
						return
					}
					wantCode, wantText := 1, "failed"
					if tc.status == 400 {
						wantCode, wantText = 1000, "bad request"
					}
					if len(api.Errors) != 1 || api.Errors[0].Code != wantCode || api.Errors[0].Message != wantText || !strings.Contains(err.Error(), wantText) {
						t.Errorf("lost API details: %+v", api)
					}
				}
			})
		}
	})
}

func TestNew(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, model := range []string{"@hf/future/decision-v2", "future-v2", "@cf/chat", "cheap"} {
			t.Run(model, func(t *testing.T) {
				c := clefClient(t, model, func(http.ResponseWriter, *http.Request) { t.Error("unexpected model discovery request") })
				if c.ModelID() != model {
					t.Errorf("model changed: %q", c.ModelID())
				}
			})
		}
	})
}
