// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the Groq provider client.

package groq_test

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"io"
	"iter"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/groq"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
)

func getClientInner(t *testing.T, opts []genai.ProviderOption, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	if !slices.ContainsFunc(opts, func(o genai.ProviderOption) bool { _, ok := o.(genai.ProviderOptionAPIKey); return ok }) {
		key := internaltest.GetEnv("GROQ_API_KEY")
		if key == "" {
			key = "<insert_api_key_here>"
		}
		opts = append(opts, genai.ProviderOptionAPIKey(key))
	}
	if fn != nil {
		opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
	}
	c, err := groq.New(t.Context(), opts...)
	if c != nil {
		internaltest.CleanupCloser(t, c)
	}
	return c, err
}

func TestNew(t *testing.T) {
	t.Run("SOTA", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			cl, err := groq.New(t.Context(), genai.ProviderOptionAPIKey("test-key"), genai.ModelSOTA,
				genai.ProviderOptionPreloadedModels{&groq.Model{ID: "openai/gpt-oss-120b"}},
			)
			if cl != nil {
				internaltest.CleanupCloser(t, cl)
			}
			if err == nil || !strings.Contains(err.Error(), "failed to find a model automatically") {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	})
}

func TestGenOption(t *testing.T) {
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			t.Run("ServiceTier", func(t *testing.T) {
				for _, v := range []groq.ServiceTier{"", "auto", "flex", "on_demand", "performance"} {
					t.Run(string(v), func(t *testing.T) {
						o := groq.GenOption{ServiceTier: v}
						if err := o.Validate(); err != nil {
							t.Fatal(err)
						}
					})
				}
			})
		})
		t.Run("error", func(t *testing.T) {
			for _, tc := range []struct {
				name string
				opt  groq.GenOption
				want string
			}{
				{name: "ServiceTier", opt: groq.GenOption{ServiceTier: "invalid"}, want: "invalid service tier"},
			} {
				t.Run(tc.name, func(t *testing.T) {
					if err := tc.opt.Validate(); err == nil || !strings.Contains(err.Error(), tc.want) {
						t.Fatalf("unexpected error: %v", err)
					}
				})
			}
		})
	})
}

func TestClient(t *testing.T) {
	t.Run("TranscriptionRaw", testClientTranscriptionRaw)
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, nil, func(h http.RoundTripper) http.RoundTripper {
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
		ci, err := getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		return ci
	}

	t.Run("Capabilities", func(t *testing.T) {
		internaltest.TestCapabilities(t, getClient(t, ""))
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
			found := false
			for _, sc := range scenarios {
				if sc.Model == id {
					models = append(models, scoreboard.Model{Model: id, Reason: sc.Reason})
					found = true
				}
			}
			if !found {
				models = append(models, scoreboard.Model{Model: id})
			}
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{genai.ProviderOptionPreloadedModels(cachedModels)}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			key := internaltest.GetEnv("GROQ_API_KEY")
			if key == "" {
				key = "<insert_api_key_here>"
			}
			opts = append(opts, genai.ProviderOptionAPIKey(key))
			if fn != nil {
				opts = append(opts, genai.ProviderOptionTransportWrapper(fn))
			}
			cl, err := groq.New(t.Context(), opts...)
			if cl != nil {
				internaltest.CleanupCloser(t, cl)
			}
			if err != nil {
				t.Fatal(err)
			}
			if strings.HasPrefix(model.Model, "qwen/") {
				var o genai.EffortLevel
				if model.Reason {
					o = genai.EffortMedium
				}
				return &handleGroqReasoning{Provider: cl, opt: o}
			}
			return cl
		}
		smoketest.Run(t, smoke.HTTP(testRecorder.Records, getClientRT), models, nil)
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
			return getClientInner(st, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			})
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
					genai.ProviderOptionModel("llama-3.1-8b-instant"),
				},
				ErrGenSync:   "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
				ErrGenStream: "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
				ErrListModel: "http 401\ninvalid_api_key (invalid_request_error): Invalid API Key\nget a new API key at https://console.groq.com/keys",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 404\nmodel_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
				ErrGenStream: "http 404\nmodel_not_found (invalid_request_error): The model `bad model` does not exist or you do not have access to it.",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			return getClientInner(t, opts, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})
}

func testClientTranscriptionUpload(t *testing.T) {
	t.Parallel()
	t.Run("valid", func(t *testing.T) {
		t.Parallel()

		t.Run("URL", func(t *testing.T) {
			t.Parallel()
			c := transcriptionClient(t, "whisper-large-v3", func(w http.ResponseWriter, r *http.Request) {
				r.Body = http.MaxBytesReader(w, r.Body, 4096)
				if err := r.ParseMultipartForm(1024); err != nil { // #nosec G120 -- MaxBytesReader bounds this test request to 4 KiB.
					t.Error(err)
					return
				}
				defer func() { _ = r.MultipartForm.RemoveAll() }()
				if r.FormValue("url") != "https://audio.example/speech.wav" || r.FormValue("model") != "whisper-large-v3" || len(r.MultipartForm.File) != 0 {
					t.Errorf("URL form: %v", r.MultipartForm)
				}
				_, _ = io.WriteString(w, `{"text":"I love bananas."}`)
			})
			var out groq.TranscriptionResponse
			err := c.TranscriptionRaw(t.Context(), &groq.TranscriptionRequest{Model: c.ModelID(), URL: "https://audio.example/speech.wav"}, &out)
			text := ""
			if out.Text != nil {
				text = *out.Text
			}
			if err != nil || text != "I love bananas." {
				t.Fatalf("transcription %q, %v", text, err)
			}
		})
		t.Run("early response replays after upload", testTranscriptionEarlyRetry)
	})
	t.Run("error", func(t *testing.T) {
		t.Parallel()
		t.Run("cancel waits for upload teardown", testTranscriptionCancelUpload)
		t.Run("provider", func(t *testing.T) {
			t.Parallel()
			c := transcriptionClient(t, "whisper-large-v3", func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, `{"error":{"message":"quota reached","type":"rate_limit_error","code":"rate_limit"}}`)
			})
			var out groq.TranscriptionResponse
			err := c.TranscriptionRaw(t.Context(), &groq.TranscriptionRequest{Model: c.ModelID(), Filename: "a.wav", File: strings.NewReader("audio")}, &out)
			if e, ok := errors.AsType[*groq.ErrorResponse](err); !ok || e.ErrorVal.Code != "rate_limit" {
				t.Fatalf("provider error: %v", err)
			}
		})
		t.Run("input", func(t *testing.T) {
			t.Parallel()
			c := transcriptionClient(t, "whisper-large-v3", func(http.ResponseWriter, *http.Request) { t.Error("invalid audio sent") })
			for name, doc := range map[string]genai.Doc{
				"missing source": {},
				"oversize":       {Filename: "a.wav", Src: bytes.NewReader(make([]byte, 25*1024*1024+1))},
			} {
				t.Run(name, func(t *testing.T) {
					var out groq.TranscriptionResponse
					if err := c.TranscriptionRaw(t.Context(), &groq.TranscriptionRequest{Model: c.ModelID(), File: doc.Src, Filename: doc.Filename, URL: doc.URL}, &out); err == nil {
						t.Fatal("invalid input accepted")
					}
				})
			}
		})
	})
}

func transcriptionClient(t *testing.T, model string, handler http.HandlerFunc) *groq.Client {
	s := httptest.NewServer(handler)
	t.Cleanup(s.Close)
	c, err := groq.New(t.Context(), genai.ProviderOptionAPIKey("test-key"), genai.ProviderOptionModel(model), genai.ProviderOptionTransportWrapper(func(transport http.RoundTripper) http.RoundTripper {
		retry := &roundtrippers.Retry{Transport: roundtrippers.Unwrap(transport), TimeAfter: func(time.Duration) <-chan time.Time {
			ch := make(chan time.Time, 1)
			ch <- time.Time{}
			return ch
		}}
		return transcriptionTransport{target: s.URL, transport: retry}
	}))
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

func testTranscriptionEarlyRetry(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		src := &blockedAudioSource{reader: strings.NewReader("encoded speech"), started: make(chan struct{}), release: make(chan struct{}), readDone: make(chan struct{})}
		uploadDone := make(chan struct{})
		calls := 0
		c, err := groq.New(t.Context(), genai.ProviderOptionAPIKey("test-key"), genai.ProviderOptionModel("whisper-large-v3"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
			return &roundtrippers.Retry{Transport: groqRoundTripFunc(func(r *http.Request) (*http.Response, error) {
				calls++
				if calls == 1 {
					body := r.Body
					go func() {
						_, _ = io.Copy(io.Discard, body)
						_ = body.Close()
						close(uploadDone)
					}()
					<-src.started
					return &http.Response{StatusCode: http.StatusServiceUnavailable, Header: http.Header{}, Body: http.NoBody, Request: r}, nil
				}
				<-uploadDone
				r.Body = http.MaxBytesReader(nil, r.Body, 4096)
				if err := r.ParseMultipartForm(1024); err != nil { // #nosec G120 -- MaxBytesReader bounds this test request to 4 KiB.
					return nil, err
				}
				defer func() { _ = r.MultipartForm.RemoveAll() }()
				f, _, err := r.FormFile("file")
				if err != nil {
					return nil, err
				}
				b, err := io.ReadAll(f)
				if err := errors.Join(err, f.Close(), r.Body.Close()); err != nil {
					return nil, err
				}
				if string(b) != "encoded speech" {
					t.Errorf("retry uploaded %q", b)
				}
				return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(`{"text":"I love bananas."}`)), Request: r}, nil
			}), TimeAfter: func(time.Duration) <-chan time.Time {
				ch := make(chan time.Time, 1)
				ch <- time.Time{}
				return ch
			}}
		}))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := c.Close(); err != nil {
				t.Error(err)
			}
		})
		var wg sync.WaitGroup
		wg.Go(func() {
			var out groq.TranscriptionResponse
			err := c.TranscriptionRaw(t.Context(), &groq.TranscriptionRequest{Model: c.ModelID(), Filename: "speech.wav", File: src}, &out)
			text := ""
			if out.Text != nil {
				text = *out.Text
			}
			if err != nil || text != "I love bananas." {
				t.Errorf("transcription %q, %v", text, err)
			}
		})
		synctest.Wait()
		close(src.release)
		wg.Wait()
		if calls != 2 {
			t.Fatalf("request attempts %d", calls)
		}
	})
}

func testTranscriptionCancelUpload(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		src := &blockedAudioSource{reader: strings.NewReader("encoded speech"), started: make(chan struct{}), release: make(chan struct{}), readDone: make(chan struct{}), maxRead: 1}
		ctx, cancel := context.WithCancel(t.Context())
		t.Cleanup(cancel)
		var body io.ReadCloser
		var reopen func() (io.ReadCloser, error)
		var upload sync.WaitGroup
		c, err := groq.New(t.Context(), genai.ProviderOptionAPIKey("test-key"), genai.ProviderOptionModel("whisper-large-v3"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
			return groqRoundTripFunc(func(r *http.Request) (*http.Response, error) {
				body = r.Body
				reopen = r.GetBody
				upload.Go(func() {
					defer func() { _ = body.Close() }()
					var b [1]byte
					for {
						if _, err := body.Read(b[:]); err != nil {
							return
						}
						select {
						case <-src.readDone:
							return
						default:
						}
					}
				})
				<-src.started
				cancel()
				return nil, ctx.Err()
			})
		}))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := c.Close(); err != nil {
				t.Error(err)
			}
		})
		done := make(chan error, 1)
		go func() {
			var out groq.TranscriptionResponse
			err := c.TranscriptionRaw(ctx, &groq.TranscriptionRequest{Model: c.ModelID(), Filename: "speech.wav", File: src}, &out)
			done <- err
		}()
		synctest.Wait()
		var got error
		returned := false
		select {
		case got = <-done:
			returned = true
			t.Error("returned source ownership before upload read stopped")
		default:
		}
		close(src.release)
		upload.Wait()
		if !returned {
			got = <-done
		}
		if !errors.Is(got, context.Canceled) {
			t.Errorf("cancellation %v", got)
		}
		reads := src.reads.Load()
		if _, err := body.Read(make([]byte, 8)); err == nil {
			t.Error("closed upload body remained readable")
		}
		if src.reads.Load() != reads {
			t.Error("upload read caller-owned source after TranscriptionRaw returned")
		}
		if _, err := reopen(); !errors.Is(err, io.ErrClosedPipe) {
			t.Errorf("reopened finished source: %v", err)
		}
	})
}

type groqRoundTripFunc func(*http.Request) (*http.Response, error)

func (f groqRoundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type blockedAudioSource struct {
	reader                     *strings.Reader
	started, release, readDone chan struct{}
	maxRead                    int
	reads                      atomic.Int32
}

func (s *blockedAudioSource) Read(b []byte) (int, error) {
	first := s.reads.Add(1) == 1
	if first {
		close(s.started)
		<-s.release
	}
	if s.maxRead != 0 && len(b) > s.maxRead {
		b = b[:s.maxRead]
	}
	n, err := s.reader.Read(b)
	if first {
		close(s.readDone)
	}
	return n, err
}

func (s *blockedAudioSource) Seek(offset int64, whence int) (int64, error) {
	return s.reader.Seek(offset, whence)
}

type transcriptionTransport struct {
	target    string
	transport http.RoundTripper
}

func (t transcriptionTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	u, err := url.Parse(t.target)
	if err != nil {
		return nil, err
	}
	r = r.Clone(r.Context())
	r.URL.Scheme, r.URL.Host = u.Scheme, u.Host
	return t.transport.RoundTrip(r)
}

type handleGroqReasoning struct {
	genai.Provider
	opt genai.EffortLevel
}

func (h *handleGroqReasoning) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	opts = append(opts, h.opt)
	return h.Provider.GenSync(ctx, msgs, opts...)
}

func (h *handleGroqReasoning) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	opts = append(opts, h.opt)
	return h.Provider.GenStream(ctx, msgs, opts...)
}

func (h *handleGroqReasoning) Unwrap() genai.Provider {
	return h.Provider
}

func init() {
	internal.BeLenient = false
}

func TestChatRequest(t *testing.T) {
	t.Run("Init", func(t *testing.T) {
		for _, tc := range []struct {
			model string
			level genai.EffortLevel
			want  groq.ReasoningEffort
		}{
			{"qwen/qwen3.8-27b", genai.EffortOff, groq.ReasoningEffortNone},
			{"meta-llama/muse-spark-open", genai.EffortMax, groq.ReasoningEffortXHigh},
			{"qwen/qwen3-32b", genai.EffortHigh, groq.ReasoningEffortDefault},
		} {
			t.Run(tc.model, func(t *testing.T) {
				var r groq.ChatRequest
				if err := r.Init(genai.Messages{genai.NewTextMessage("hello")}, tc.model, tc.level, genai.IncludeReasoning); err != nil {
					t.Fatal(err)
				}
				if r.ReasoningEffort != tc.want || r.ReasoningFormat != groq.ReasoningFormatParsed {
					t.Fatalf("thinking: %q, %q", r.ReasoningEffort, r.ReasoningFormat)
				}
			})
		}
	})
}

func testClientTranscriptionRaw(t *testing.T) {
	t.Run("transport ownership", testClientTranscriptionUpload)
	t.Parallel()
	t.Run("valid", func(t *testing.T) {
		t.Run("upload", func(t *testing.T) {
			t.Parallel()
			c := transcriptionClient(t, "whisper-large-v3-turbo", func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/openai/v1/audio/transcriptions" || r.Header.Get("Authorization") != "Bearer test-key" {
					t.Errorf("request %s %s, credentials present %t", r.Method, r.URL.Path, r.Header.Get("Authorization") != "")
				}
				r.Body = http.MaxBytesReader(w, r.Body, 4096)
				if err := r.ParseMultipartForm(1024); err != nil { // #nosec G120 -- MaxBytesReader bounds this test request to 4 KiB.
					t.Error(err)
					return
				}
				defer func() { _ = r.MultipartForm.RemoveAll() }()
				if r.FormValue("model") != "whisper-large-v3-turbo" || r.FormValue("response_format") != "verbose_json" || r.FormValue("language") != "en" || r.FormValue("prompt") != "bananas" || r.FormValue("temperature") != "0.25" || !slices.Equal(r.MultipartForm.Value["timestamp_granularities[]"], []string{"word", "segment"}) {
					t.Errorf("fields %v", r.MultipartForm.Value)
				}
				file, header, err := r.FormFile("file")
				if err != nil {
					t.Error(err)
					return
				}
				body, err := io.ReadAll(file)
				if err := errors.Join(err, file.Close()); err != nil {
					t.Error(err)
				}
				if string(body) != "encoded speech" || header.Filename != "speech.wav" {
					t.Errorf("uploaded %q as %q", body, header.Filename)
				}
				_, _ = io.WriteString(w, `{"text":"I love bananas.","x_groq":{"id":"request-1"},"task":"transcribe","language":"english","duration":1.5,"words":[{"word":"bananas","start":0.5,"end":1.5}],"segments":[{"id":0,"seek":0,"start":0,"end":1.5,"text":"I love bananas.","tokens":[12,34],"temperature":0.25,"avg_logprob":-0.1,"compression_ratio":1.2,"no_speech_prob":0.01}]}`)
			})
			var out groq.TranscriptionResponse
			err := c.TranscriptionRaw(t.Context(), &groq.TranscriptionRequest{Model: "whisper-large-v3-turbo", Filename: "speech.wav", File: strings.NewReader("encoded speech"), ResponseFormat: groq.TranscriptionVerboseJSON, Language: "en", Prompt: "bananas", Temperature: new(0.25), TimestampGranularities: []groq.TranscriptionGranularity{groq.TranscriptionWord, groq.TranscriptionSegmentTime}}, &out)
			if err != nil {
				t.Fatal(err)
			}
			if out.Text == nil || *out.Text != "I love bananas." || out.XGroq.ID != "request-1" || out.Language != "english" || out.Duration != 1.5 || len(out.Words) != 1 || out.Words[0].Start != 0.5 || len(out.Segments) != 1 || out.Segments[0].AvgLogprob != -0.1 || !slices.Equal(out.Segments[0].Tokens, []int{12, 34}) {
				t.Fatalf("native transcription %+v", out)
			}
		})
	})
}
