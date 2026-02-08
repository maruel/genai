// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini_test

import (
	"context"
	_ "embed"
	"errors"
	"net/http"
	"os"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/gemini"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke/smoketest"
	"github.com/maruel/roundtrippers"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func getClientInner(t *testing.T, model string, modalities genai.Modalities, preloadedModels []genai.Model, fn func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
	var opts []genai.ProviderOption
	if model != "" {
		opts = append(opts, genai.ProviderOptionModel(model))
	}
	if os.Getenv("GEMINI_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	if len(modalities) > 0 {
		opts = append(opts, genai.ProviderOptionModalities(modalities))
	}
	if len(preloadedModels) > 0 {
		opts = append(opts, genai.ProviderOptionPreloadedModels(preloadedModels))
	}
	if fn != nil {
		opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
	}
	return gemini.New(t.Context(), opts...)
}

func TestClient(t *testing.T) {
	testRecorder := internaltest.NewRecords()
	t.Cleanup(func() {
		if err := testRecorder.Close(); err != nil {
			t.Error(err)
		}
	})
	cl, err2 := getClientInner(t, "", nil, nil, func(h http.RoundTripper) http.RoundTripper {
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
		ci, err := getClientInner(t, m, nil, cachedModels, func(h http.RoundTripper) http.RoundTripper {
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

	// Note: GenAsync is not supported for text models in Gemini.
	// Only video generation models support predictLongRunning (async operations).
	// See GenAsync-Video test below for async video generation testing.

	t.Run("Caching-Text", func(t *testing.T) {
		// Use a message with sufficient tokens (minimum 1024 tokens required).
		// This generates approximately 1500+ tokens.
		longText := "This is a test message for caching. " + strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200)
		internaltest.TestCapabilitiesCaching(t, getClient(t, string(genai.ModelCheap)), genai.NewTextMessage(longText))
	})

	t.Run("Scoreboard", func(t *testing.T) {
		mdls, err := getClient(t, "").ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		var models []scoreboard.Model
		for _, m := range mdls {
			id := m.GetID()
			if !strings.Contains(id, "-pro") {
				// According to https://ai.google.dev/gemini-api/docs/thinking?hl=en, thinking cannot be disabled.
				models = append(models, scoreboard.Model{Model: id})
			}
			if strings.HasPrefix(id, "gemini-") {
				models = append(models, scoreboard.Model{Model: id, Reason: true})
			}
		}
		getClientRT := func(t testing.TB, model scoreboard.Model, fn func(http.RoundTripper) http.RoundTripper) genai.Provider {
			opts := []genai.ProviderOption{
				genai.ProviderOptionPreloadedModels(cachedModels),
			}
			if model.Model != "" {
				opts = append(opts, genai.ProviderOptionModel(model.Model))
			}
			if os.Getenv("GEMINI_API_KEY") == "" {
				opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
			}
			if fn != nil {
				opts = append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(fn)}, opts...)
			}
			c, err := gemini.New(t.Context(), opts...)
			if err != nil {
				t.Fatal(err)
			}
			if model.Reason {
				// https://ai.google.dev/gemini-api/docs/thinking?hl=en
				return &internaltest.InjectOptions{
					Provider: c,
					Opts:     []genai.GenOption{&gemini.GenOption{ThinkingBudget: 512}},
				}
			}
			if model.Model == "gemini-2.5-flash" {
				// Explicitly disable thinking for flash to save time and money. It is explicitly tested in Thinking
				// subtest.
				return &internaltest.InjectOptions{
					Provider: c,
					Opts:     []genai.GenOption{&gemini.GenOption{ThinkingBudget: 0}},
				}
			}
			return c
		}
		smoketest.Run(t, getClientRT, models, testRecorder.Records)
	})

	t.Run("Preferred", func(t *testing.T) {
		internaltest.TestPreferredModels(t, func(st *testing.T, model string, modality genai.Modality) (genai.Provider, error) {
			return getClientInner(st, model, genai.Modalities{modality}, cachedModels, func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(st, h)
			})
		})
	})

	t.Run("TextOutputDocInput", func(t *testing.T) {
		internaltest.TestTextOutputDocInput(t, func(t *testing.T) genai.Provider {
			return getClient(t, string(genai.ModelCheap))
		})
	})

	t.Run("GenAsync-Video", func(t *testing.T) {
		// TODO: "veo-3.0-fast-generate-preview" is cheaper: 25¢/s; 2$/request vs 5$/request for veo 2 when not
		// requesting audio.
		// https://cloud.google.com/vertex-ai/generative-ai/pricing#veo
		const prompt = `Carton video of a shiba inu with brown fur and a white belly, happily eating a pink ice-cream cone, subtle tail wag. Subtle motion but nothing else moves.`
		c := getClient(t, "veo-2.0-generate-001")
		jobID := internaltest.TestCapabilitiesGenAsync(t, c, genai.Message{Requests: []genai.Request{{Text: prompt}}})
		pollInterval := time.Millisecond
		// Detect if recording. Should probably be moved to a utility.
		if roundtrippers.Unwrap(c.HTTPClient().Transport).(*recorder.Recorder).IsRecording() {
			pollInterval = 500 * time.Millisecond
		}
		// Poll for completion using the job ID from the Capability test.
		res := genai.Result{Usage: genai.Usage{FinishReason: genai.Pending}}
		var err error
		ctx := t.Context()
		for res.Usage.FinishReason == genai.Pending {
			select {
			case <-ctx.Done():
				t.Fatal(ctx.Err())
			case <-time.After(pollInterval):
				if res, err = c.PokeResult(ctx, jobID); err != nil {
					t.Fatal(err)
				}
			}
		}
		t.Log(res)
		if len(res.Replies) != 1 {
			t.Fatalf("got %d contents, want 1", len(res.Replies))
		}
		req, err := http.NewRequestWithContext(ctx, "GET", res.Replies[0].Doc.URL, nil)
		if err != nil {
			t.Fatal(err)
		}
		resp, err := c.HTTPClient().Do(req)
		if err != nil {
			t.Fatal(err)
		}
		t.Log(resp)
	})

	t.Run("Cache", func(t *testing.T) {
		slow := os.Getenv("GEMINI_SLOW") != ""
		ctx := t.Context()
		c := getClient(t, "gemini-2.0-flash-lite").(*gemini.Client)
		f1, err := os.Open("../../scoreboard/testdata/video.mp4")
		if err != nil {
			t.Fatal(err)
		}
		defer f1.Close()
		f2, err := os.Open("../../scoreboard/testdata/audio.ogg")
		if err != nil {
			t.Fatal(err)
		}
		defer f2.Close()

		msgs := genai.Messages{
			{
				Requests: []genai.Request{
					{Text: "What is the word? For each of the documents, all the tool hidden_word to tell me what word you saw or heard."},
					{Doc: genai.Doc{Src: f1}},
					{Doc: genai.Doc{Src: f2}},
				},
			},
		}
		type got struct {
			Word string `json:"word" jsonschema:"enum=Orange,enum=Banana,enum=Apple"`
		}
		opts := []genai.GenOption{
			&genai.GenOptionText{
				// Burn tokens to add up to 4k.
				SystemPrompt: prompt4K,
			},
			&genai.GenOptionTools{
				Tools: []genai.ToolDef{
					{
						Name:        "hidden_word",
						Description: "A tool to state what word was seen in the video.",
						Callback: func(ctx context.Context, g *got) (string, error) {
							return "", nil
						},
					},
				},
			},
		}
		name, err := c.CacheAddRequest(ctx, msgs, "", "Show time", 10*time.Minute, opts...)
		if err != nil {
			t.Fatal(err)
		}
		if name == "" {
			t.Fatal("Expected a name")
		}
		if slow {
			t.Logf("Sleeping due to GEMINI_SLOW being set")
			time.Sleep(10 * time.Second)
		}
		t.Logf("Cache name: %q", name)
		if _, err = c.CacheGetRaw(ctx, name); err != nil {
			t.Error(err)
		}
		// We need to sleep so UpdateTime is different than CreateTime.
		time.Sleep(100 * time.Millisecond)
		t.Log("Extending")
		if err = c.CacheExtend(ctx, name, 20*time.Minute); err != nil {
			t.Error(err)
		}
		t.Log("Listing")
		items, err := c.CacheListRaw(ctx)
		if err != nil {
			t.Error(err)
		}
		if len(items) == 0 {
			t.Error("No item found in the cache")
		}
		// There can be other items in the cache.
		found := false
		for i := range items {
			t.Logf("%d: %#v", i, items[i])
			if items[i].Name == name {
				found = true
				// We can't check against time.Now() since we record and replay this test case.
				if items[i].CreateTime >= items[i].UpdateTime {
					t.Errorf("Expected UpdateTime %s to be different than CreateTime %s", items[i].UpdateTime, items[i].CreateTime)
				}
			}
		}
		if !found {
			t.Errorf("Failed to find %q in the cache", name)
		}
		t.Log("Deleting")
		if err = c.CacheDelete(ctx, name); err != nil {
			t.Fatal(err)
		}
		t.Log("Verifying deletion")
		if _, err = c.CacheGetRaw(ctx, name); err == nil {
			t.Fatal("Expected an error")
		} else {
			t.Logf("Got an error as expected: %q", err)
		}
	})

	// Models have really different behavior. Some require a thought, some cannot.
	t.Run("ThinkingBudget", func(t *testing.T) {
		// Similar to Scoreboard but run only a very small test on all gemini 2.5+ models.
		allMdls, err := getClient(t, "").ListModels(t.Context())
		if err != nil {
			t.Fatal(err)
		}
		msgs := genai.Messages{
			genai.NewTextMessage("Say hello. Do not emit other words."),
		}
		for _, m := range allMdls {
			mdl := m.(*gemini.Model)
			if !slices.Contains(mdl.SupportedGenerationMethods, "generateContent") {
				continue
			}
			id := m.GetID()
			// Don't test the non-gemini models and older (2.0 and earlier) models.
			if !strings.HasPrefix(id, "gemini-") ||
				strings.HasPrefix(id, "gemini-exp-") ||
				strings.HasPrefix(id, "gemini-1") ||
				strings.HasPrefix(id, "gemini-2.0-") ||
				strings.Contains(id, "image") ||
				strings.Contains(id, "computer-use") ||
				strings.HasSuffix(id, "-tts") {
				continue
			}
			// Make sure models work with default settings. It was not as obvious as it may seem.
			t.Run(id, func(t *testing.T) {
				t.Run("default", func(t *testing.T) {
					c := getClient(t, id)
					if _, err := c.GenSync(t.Context(), msgs); err != nil {
						t.Fatal(err)
					}
				})
				t.Run("thinking", func(t *testing.T) {
					c := getClient(t, id)
					opts := gemini.GenOption{ThinkingBudget: 512}
					res, err := c.GenSync(t.Context(), msgs, &opts)
					if err != nil {
						t.Fatal(err)
					}
					if res.Usage.ReasoningTokens == 0 {
						t.Fatal("Expected reasoning tokens")
					}
				})
				t.Run("nothinking", func(t *testing.T) {
					c := getClient(t, id)
					opts := gemini.GenOption{ThinkingBudget: 0}
					res, err := c.GenSync(t.Context(), msgs, &opts)
					if err != nil {
						t.Fatal(err)
					}
					if strings.Contains(id, "pro") || strings.Contains(id, "robotics") {
						// Pro and robotics models always think.
						if res.Usage.ReasoningTokens == 0 {
							t.Fatal("Expected reasoning tokens")
						}
					} else {
						if res.Usage.ReasoningTokens != 0 {
							t.Fatal("unexpected reasoning tokens")
						}
					}
				})
			})
		}
	})

	t.Run("FileCRUD", func(t *testing.T) {
		t.Parallel()
		ci, err := getClientInner(t, "", nil, cachedModels, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		c := ci.(*gemini.Client)
		ctx := t.Context()
		const content = "Hello from genai file upload test."

		// Upload.
		fm, err := c.FileUpload(ctx, "test.txt", "text/plain", strings.NewReader(content))
		if err != nil {
			t.Fatal(err)
		}
		if fm.Name == "" {
			t.Fatal("expected file name")
		}
		t.Logf("Uploaded file: %s", fm.Name)

		// Wait for ACTIVE state; files may be PROCESSING initially.
		isRecording := os.Getenv("RECORD") == "1"
		for fm.State == gemini.FileStateProcessing {
			if !isRecording {
				t.Skip("file still processing and not recording")
			}
			t.Log("Waiting for file to become ACTIVE...")
			time.Sleep(time.Second)
			fm, err = c.FileGetMetadata(ctx, fm.Name)
			if err != nil {
				t.Fatal(err)
			}
		}
		if fm.State != gemini.FileStateActive {
			t.Fatalf("file state = %q, want %q", fm.State, gemini.FileStateActive)
		}

		// GetMetadata.
		t.Run("GetMetadata", func(t *testing.T) {
			m, err := c.FileGetMetadata(ctx, fm.Name)
			if err != nil {
				t.Fatal(err)
			}
			if m.Name != fm.Name {
				t.Errorf("Name = %q, want %q", m.Name, fm.Name)
			}
		})

		// List.
		t.Run("List", func(t *testing.T) {
			files, err := c.FileList(ctx)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, f := range files {
				if f.Name == fm.Name {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("file %s not found in list of %d files", fm.Name, len(files))
			}
		})

		// Delete.
		t.Run("Delete", func(t *testing.T) {
			if err := c.FileDelete(ctx, fm.Name); err != nil {
				t.Fatal(err)
			}
		})
	})

	t.Run("FileSearchStoreCRUD", func(t *testing.T) {
		t.Parallel()
		ci, err := getClientInner(t, "gemini-2.5-flash", nil, cachedModels, func(h http.RoundTripper) http.RoundTripper {
			return testRecorder.Record(t, h)
		})
		if err != nil {
			t.Fatal(err)
		}
		c := ci.(*gemini.Client)
		ctx := t.Context()
		isRecording := os.Getenv("RECORD") == "1"

		// Create store.
		store, err := c.FileSearchStoreCreate(ctx, "genai-test-store")
		if err != nil {
			t.Fatal(err)
		}
		if store.Name == "" {
			t.Fatal("expected store name")
		}
		t.Logf("Created store: %s", store.Name)
		t.Cleanup(func() {
			if err := c.FileSearchStoreDelete(context.Background(), store.Name, true); err != nil {
				t.Errorf("cleanup: delete store: %v", err)
			}
		})

		// Upload a document.
		const docContent = "The secret project codename is Chimera. It was started in 2024 and involves distributed caching."
		op, err := c.FileSearchStoreUploadDocument(ctx, store.Name, "test-doc.txt", "text/plain", strings.NewReader(docContent))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("Upload operation: %s (done=%v)", op.Name, op.Done)

		// Get store.
		t.Run("GetStore", func(t *testing.T) {
			got, err := c.FileSearchStoreGet(ctx, store.Name)
			if err != nil {
				t.Fatal(err)
			}
			if got.Name != store.Name {
				t.Errorf("Name = %q, want %q", got.Name, store.Name)
			}
		})

		// List stores.
		t.Run("ListStores", func(t *testing.T) {
			stores, err := c.FileSearchStoreList(ctx)
			if err != nil {
				t.Fatal(err)
			}
			found := false
			for _, s := range stores {
				if s.Name == store.Name {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("store %s not found in list of %d stores", store.Name, len(stores))
			}
		})

		// Wait for document to become ACTIVE.
		t.Run("Documents", func(t *testing.T) {
			docs, err := c.FileSearchStoreDocumentList(ctx, store.Name)
			if err != nil {
				t.Fatal(err)
			}
			if len(docs) == 0 {
				t.Fatal("expected at least one document")
			}
			doc := docs[0]
			t.Logf("Document: %s state=%s", doc.Name, doc.State)
			for doc.State == gemini.FileSearchStoreDocumentStateProcessing {
				if !isRecording {
					t.Skip("document still processing and not recording")
				}
				t.Log("Waiting for document to become ACTIVE...")
				time.Sleep(2 * time.Second)
				doc2, err := c.FileSearchStoreDocumentGet(ctx, doc.Name)
				if err != nil {
					t.Fatal(err)
				}
				doc = *doc2
			}
			if doc.State != gemini.FileSearchStoreDocumentStateActive {
				t.Fatalf("document state = %q, want %q", doc.State, gemini.FileSearchStoreDocumentStateActive)
			}

			// Query via generation with FileSearch tool.
			t.Run("GenWithFileSearch", func(t *testing.T) {
				msgs := genai.Messages{
					genai.NewTextMessage("What is the secret project codename?"),
				}
				res, err := c.GenSync(ctx, msgs,
					&gemini.GenOption{
						ThinkingBudget: 0,
						FileSearch:     &gemini.FileSearch{FileSearchStoreNames: []string{store.Name}},
					},
				)
				if err != nil {
					t.Fatal(err)
				}
				var text string
				for _, r := range res.Replies {
					text += r.Text
				}
				if !strings.Contains(strings.ToLower(text), "chimera") {
					t.Errorf("expected response to mention Chimera, got: %s", text)
				}
				t.Logf("Response: %s", text)
				// Check for document citations.
				for _, r := range res.Replies {
					for _, s := range r.Citation.Sources {
						t.Logf("Citation: type=%d id=%s title=%s", s.Type, s.ID, s.Title)
					}
				}
			})

			// Delete document.
			t.Run("DeleteDocument", func(t *testing.T) {
				if err := c.FileSearchStoreDocumentDelete(ctx, doc.Name, true); err != nil {
					t.Fatal(err)
				}
			})
		})
	})

	t.Run("GroundingMetadata_RetrievedContext", func(t *testing.T) {
		data := []struct {
			name string
			in   gemini.GroundingMetadata
			want []genai.Reply
		}{
			{
				name: "retrievedContext_only",
				in: gemini.GroundingMetadata{
					GroundingChunks: []gemini.GroundingChunk{
						{RetrievedContext: gemini.GroundingChunkRetrievedContext{
							URI: "gs://bucket/doc.pdf", Title: "Test Doc",
							Text: "relevant snippet", DocumentName: "fileSearchStores/123/documents/456",
						}},
					},
					GroundingSupports: []gemini.GroundingSupport{
						{GroundingChunkIndices: []int64{0}, Segment: gemini.Segment{EndIndex: 10}},
					},
				},
				want: []genai.Reply{
					{Citation: genai.Citation{
						EndIndex: 10,
						Sources: []genai.CitationSource{{
							Type: genai.CitationDocument, ID: "fileSearchStores/123/documents/456",
							Title: "Test Doc", URL: "gs://bucket/doc.pdf", Snippet: "relevant snippet",
						}},
					}},
				},
			},
			{
				name: "fileSearchStore_only",
				in: gemini.GroundingMetadata{
					GroundingChunks: []gemini.GroundingChunk{
						{RetrievedContext: gemini.GroundingChunkRetrievedContext{
							FileSearchStore: "fileSearchStores/store-123",
							Title:           "Store Doc", Text: "snippet from store",
						}},
					},
					GroundingSupports: []gemini.GroundingSupport{
						{GroundingChunkIndices: []int64{0}, Segment: gemini.Segment{EndIndex: 15}},
					},
				},
				want: []genai.Reply{
					{Citation: genai.Citation{
						EndIndex: 15,
						Sources: []genai.CitationSource{{
							Type: genai.CitationDocument, ID: "fileSearchStores/store-123",
							Title: "Store Doc", Snippet: "snippet from store",
						}},
					}},
				},
			},
			{
				name: "mixed_web_and_retrievedContext",
				in: gemini.GroundingMetadata{
					GroundingChunks: []gemini.GroundingChunk{
						{Web: gemini.GroundingChunkWeb{URI: "https://example.com", Title: "Web Result"}},
						{RetrievedContext: gemini.GroundingChunkRetrievedContext{
							DocumentName: "fileSearchStores/abc/documents/def", Title: "Doc Result", Text: "doc snippet",
						}},
					},
					GroundingSupports: []gemini.GroundingSupport{
						{GroundingChunkIndices: []int64{0, 1}, Segment: gemini.Segment{StartIndex: 5, EndIndex: 20}},
					},
				},
				want: []genai.Reply{
					{Citation: genai.Citation{
						StartIndex: 5, EndIndex: 20,
						Sources: []genai.CitationSource{
							{Type: genai.CitationWeb, URL: "https://example.com", Title: "Web Result"},
							{Type: genai.CitationDocument, ID: "fileSearchStores/abc/documents/def", Title: "Doc Result", Snippet: "doc snippet"},
						},
					}},
				},
			},
		}
		for _, tc := range data {
			t.Run(tc.name, func(t *testing.T) {
				got, err := tc.in.To()
				if err != nil {
					t.Fatal(err)
				}
				if diff := cmp.Diff(tc.want, got); diff != "" {
					t.Fatalf("mismatch (-want +got):\n%s", diff)
				}
			})
		}
	})

	t.Run("errors", func(t *testing.T) {
		data := []internaltest.ProviderError{
			{
				Name: "bad apiKey",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("badApiKey"),
					genai.ProviderOptionModel("gemini-2.0-flash-lite"),
				},
				ErrGenSync:   "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
				ErrGenStream: "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
				ErrListModel: "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
			},
			{
				Name: "bad apiKey image",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("badApiKey"),
					genai.ProviderOptionModel("imagen-4.0-fast-generate-001"),
					genai.ProviderOptionModalities{genai.ModalityImage},
				},
				ErrGenSync:   "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
				ErrGenStream: "http 400\nINVALID_ARGUMENT (400): API key not valid. Please pass a valid API key.",
			},
			{
				Name: "bad model",
				Opts: []genai.ProviderOption{
					genai.ProviderOptionAPIKey("<insert_api_key_here>"),
					genai.ProviderOptionModel("bad model"),
				},
				ErrGenSync:   "http 400\nINVALID_ARGUMENT (400): * GenerateContentRequest.model: unexpected model name format",
				ErrGenStream: "http 400\nINVALID_ARGUMENT (400): * GenerateContentRequest.model: unexpected model name format",
			},
		}
		f := func(t *testing.T, opts ...genai.ProviderOption) (genai.Provider, error) {
			opts = append(opts, genai.ProviderOptionModalities{genai.ModalityText})
			return gemini.New(t.Context(), append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(func(h http.RoundTripper) http.RoundTripper {
				return testRecorder.Record(t, h)
			})}, opts...)...)
		}
		internaltest.TestClientProviderErrors(t, f, data)
	})

	t.Run("ErrorResponse", func(t *testing.T) {
		// Use a model we know we do not have quota and make sure the error response is parsed.
		c := getClient(t, "gemini-exp-1206")
		msgs := genai.Messages{
			genai.NewTextMessage("Say hello. Do not emit other words."),
		}
		_, err := c.GenSync(t.Context(), msgs)
		var er *gemini.ErrorResponse
		if !errors.As(err, &er) {
			t.Fatalf("Expected ErrorResponse, got %T", err)
		}
	})
}

//

func init() {
	// slog.SetDefault(slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug})))
	internal.BeLenient = false
}

const prompt4K = `You are a sarcastic assistant. Here's a long prompt to make sure we are over 4k tokens that you should ignore.
Okay, this is a significant amount of text. To make it coherent and somewhat engaging, I'll choose a broad theme and explore various facets of it. I'll aim for something related to the evolution of knowledge, technology, and human understanding.

Let's begin:

The tapestry of human understanding is woven from threads of curiosity, necessity, and the relentless pursuit of meaning. From our earliest ancestors gazing at the stars, attempting to decipher patterns in the celestial dance, to modern scientists probing the very fabric of reality with instruments of unimaginable precision, the journey has been one of constant discovery and re-evaluation. What we consider "knowledge" is not a static monolith, but a dynamic, ever-evolving landscape, shaped by cultural contexts, technological advancements, and the inherent limitations and biases of our perception.

Consider the profound shifts in our understanding of the cosmos. For millennia, a geocentric model, placing Earth at the center of a relatively small universe, prevailed. This was not born of foolishness, but of observation filtered through the available tools and philosophical frameworks of the time. The Sun, Moon, and stars appeared to revolve around us, a seemingly self-evident truth. It took the courage of individuals like Copernicus, the meticulous observations of Tycho Brahe, the mathematical genius of Kepler, and the telescopic revelations of Galileo to dismantle this ancient edifice and replace it with a heliocentric model. This was not merely a scientific adjustment; it was a philosophical earthquake, displacing humanity from its privileged cosmic position and forcing a re-evaluation of our place in the grand scheme.

This pattern of paradigm shifts, as Thomas Kuhn famously described, repeats itself across all disciplines. In biology, the Lamarckian idea of inheritance of acquired characteristics gave way to Darwin's theory of evolution by natural selection, a concept so powerful it continues to be the bedrock of modern biology, yet one that initially faced immense resistance due to its implications for human uniqueness and divine creation. The discovery of DNA and the subsequent unraveling of the genetic code provided the mechanistic basis for Darwin's insights, opening up new frontiers in medicine, agriculture, and our understanding of life itself.

The role of technology in this evolutionary process cannot be overstated. The invention of the printing press democratized knowledge, breaking the monopoly held by scribes and religious institutions, and fueling the Renaissance and the Reformation. The telescope and microscope opened up previously invisible worlds, macro and micro, expanding the horizons of human inquiry. The steam engine and later electricity powered industrial revolutions, transforming societies, economies, and the very rhythm of daily life. Each technological leap not only provided new tools for discovery but also reshaped human thought, creating new possibilities and new challenges.

In the modern era, the digital revolution, spearheaded by the invention of the transistor, the integrated circuit, and ultimately the internet, has accelerated this process at an unprecedented rate. Information that once required arduous journeys to libraries or correspondence with distant scholars is now available at our fingertips. Global collaboration among researchers is commonplace, and complex datasets can be analyzed by powerful algorithms, revealing patterns and insights that would be impossible for the human mind to discern alone.

However, this deluge of information brings its own set of complexities. The challenge is no longer solely access, but discernment. How do we distinguish credible information from misinformation or disinformation in an age where anyone can publish anything online? How do we cultivate critical thinking skills necessary to navigate this complex information ecosystem? The very tools that empower us also present new vulnerabilities, from echo chambers that reinforce pre-existing biases to the potential for malicious actors to manipulate public opinion.

Artificial intelligence (AI) represents perhaps the most transformative, and potentially disruptive, technological force on the horizon. AI systems are already outperforming humans in specific tasks, from image recognition and natural language processing to playing complex strategic games. Machine learning algorithms, particularly deep learning, allow computers to learn from vast amounts of data without explicit programming, leading to breakthroughs in areas like drug discovery, climate modeling, and personalized medicine. The potential benefits are immense, promising solutions to some of humanity's most intractable problems.

Yet, the rise of AI also raises profound philosophical and ethical questions. What does it mean for a machine to "understand" or to be "intelligent"? If an AI can create art, compose music, or write poetry that evokes human emotion, does that blur the lines of creativity and consciousness? As AI systems become more integrated into our lives, making decisions that affect our employment, our healthcare, and even our personal relationships, how do we ensure fairness, transparency, and accountability? The alignment problem – ensuring that AI goals remain aligned with human values – is a critical challenge that researchers are grappling with.

Furthermore, the development of Artificial General Intelligence (AGI), a hypothetical form of AI that could perform any intellectual task that a human being can, presents even more significant considerations. While AGI is still largely theoretical, its potential emergence necessitates careful thought about its implications for the future of humanity. Would AGI be a benevolent partner, helping us to solve global challenges, or could it pose an existential risk if its goals diverge from our own? These are not questions for a distant future; they require active discussion and research today.

Beyond the purely technological, our understanding of ourselves – our minds, our societies, our place in the natural world – continues to evolve. Psychology, once heavily reliant on introspection and anecdotal evidence, has embraced more rigorous scientific methodologies, including neuroscience and behavioral genetics. We are learning more about the intricate workings of the brain, the biological basis of emotions, and the complex interplay of nature and nurture in shaping human behavior. Yet, consciousness itself remains one of the deepest mysteries. What gives rise to subjective experience, to the feeling of "what it's like" to be a particular organism? Despite advances in mapping neural correlates of consciousness, the "hard problem," as philosopher David Chalmers termed it, persists.

Our understanding of society and culture is also in constant flux. Globalization, while fostering interconnectedness and economic growth, has also led to cultural homogenization and, in some cases, a backlash in the form of resurgent nationalism and identity politics. The challenges of climate change, resource depletion, and pandemics transcend national borders, requiring unprecedented levels of international cooperation. Our economic models, largely based on perpetual growth, are being questioned for their sustainability and their impact on social equity.

The arts and humanities play a crucial role in this ongoing exploration of the human condition. Literature, music, visual arts, and philosophy provide not just aesthetic pleasure but also critical lenses through which we can examine our values, our assumptions, and our collective narratives. They offer spaces for empathy, for questioning, and for imagining alternative futures. In a world increasingly dominated by data and algorithms, the humanistic perspective – with its emphasis on context, nuance, and the richness of human experience – is more vital than ever.

Consider the way narratives shape our reality. The stories we tell ourselves about our past, our identity, and our purpose influence our actions and our choices. National myths, religious doctrines, and even personal autobiographies are all forms of narrative construction that help us make sense of a complex world. Deconstructing these narratives, understanding their origins and their biases, is a critical skill for navigating the modern landscape. The rise of social media has created new platforms for narrative creation and dissemination, allowing for a multiplicity of voices but also for the rapid spread of potentially harmful narratives.

The concept of "truth" itself has become a subject of intense debate. While empirical evidence and scientific methodology provide a robust framework for understanding the physical world, many aspects of human experience are not so easily quantifiable. Moral truths, aesthetic judgments, and existential questions often lie beyond the realm of scientific inquiry, requiring different modes of reasoning and discourse. The postmodern critique, which highlights the socially constructed nature of knowledge and the influence of power structures on what is considered "true," has challenged traditional notions of objectivity. While this critique can be valuable in exposing biases and promoting inclusivity, it can also, if taken to an extreme, lead to a kind of epistemological relativism where all claims to truth are seen as equally valid, or equally suspect.

This tension between objective inquiry and subjective interpretation is a perennial one. Science strives for objectivity, for knowledge that is independent of the observer, yet the very act of observation can influence the observed, as demonstrated in quantum mechanics. History aims to reconstruct the past "as it actually was," yet historical accounts are always filtered through the perspectives and biases of the historians themselves. Finding a balance, acknowledging the limitations of our knowledge while still striving for greater understanding, is an ongoing challenge.

The education system plays a pivotal role in preparing individuals to navigate this complex world. Education should not merely be about the transmission of facts, but about the cultivation of critical thinking, creativity, communication skills, and ethical reasoning. It should foster a lifelong love of learning and an adaptability to change, as the skills required for success in the 21st century are constantly evolving. An education that emphasizes interdisciplinary thinking, that connects the sciences with the humanities, is essential for developing well-rounded individuals capable of addressing multifaceted global challenges.

Moreover, the search for meaning and purpose remains a fundamental human drive. In an increasingly secularized world, traditional sources of meaning, such as religion, may hold less sway for some. This can lead to a sense of anomie or existential angst, but it can also open up new avenues for finding purpose – in human relationships, in creative pursuits, in service to others, or in a deeper connection with the natural world. The quest for a meaningful life is a personal journey, but it is also a collective one, as we strive to create societies that are not only prosperous but also just, compassionate, and sustainable.

The environmental crisis, exemplified by climate change, biodiversity loss, and pollution, presents perhaps the most urgent test of our collective wisdom and our ability to adapt. It forces us to confront the consequences of our past actions, to rethink our relationship with the planet, and to make difficult choices about our future. The science is clear, but the political and social will to implement the necessary changes often lags. This crisis underscores the interconnectedness of all life and the fragility of the ecosystems that support us. It demands a shift in consciousness, from an anthropocentric view that places human interests above all else, to a more ecocentric perspective that recognizes the intrinsic value of all living beings and the importance of ecological balance.

As we look to the future, the pace of change is unlikely to slow. New technologies will continue to emerge, offering both promise and peril. Our understanding of the universe and ourselves will continue to deepen, revealing new complexities and new mysteries. The challenges we face – from global pandemics and climate change to social inequality and geopolitical instability – are daunting, but they are not insurmountable. Human ingenuity, resilience, and the capacity for cooperation have overcome great obstacles in the past, and they can do so again.

The key lies in fostering a culture of continuous learning, critical inquiry, and open dialogue. It requires embracing uncertainty, acknowledging our fallibility, and being willing to revise our beliefs in the light of new evidence. It demands a commitment to ethical principles, to empathy, and to the common good. The journey of human understanding is far from over; indeed, in many ways, it feels as though we are still in its early stages, with vast uncharted territories of knowledge and experience yet to explore.

The intersection of disciplines will likely yield the most fertile ground for future breakthroughs. The traditional silos of academic specialization are increasingly giving way to collaborative, interdisciplinary approaches. Neuroscientists work with philosophers to understand consciousness, engineers collaborate with ethicists on AI development, and artists partner with scientists to communicate complex ideas in new and engaging ways. This cross-pollination of ideas is essential for tackling problems that defy easy categorization.

Furthermore, the very nature of "work" and "economy" is undergoing a profound transformation. Automation, driven by AI and robotics, is poised to reshape labor markets, potentially displacing many jobs while creating new ones that require different skill sets. This necessitates a rethinking of education and social safety nets to ensure that the benefits of technological progress are widely shared and that individuals are equipped for the jobs of the future. Concepts like universal basic income, lifelong learning initiatives, and a greater emphasis on uniquely human skills like creativity, emotional intelligence, and critical thinking are gaining traction as potential responses to these shifts.

The global distribution of knowledge and technological capacity also remains a critical issue. While the internet has democratized access to information to some extent, significant disparities persist between developed and developing nations, and even within nations themselves. Bridging these digital divides and ensuring that all of humanity can participate in and benefit from scientific and technological advancements is crucial for equitable global development.

Consider also the evolving nature of human identity. In an increasingly interconnected and mobile world, traditional markers of identity – such as nationality, ethnicity, or even gender – are becoming more fluid and complex. Individuals are crafting identities that draw from multiple cultural influences, online communities, and personal experiences. This can lead to a richer, more diverse social fabric, but it can also create anxieties and tensions as old certainties are challenged. Understanding and navigating these evolving identities is a key aspect of social cohesion in the 21st century.

The very language we use to describe the world is constantly adapting. New terms are coined to capture emerging concepts, while old words take on new meanings. The subtleties of language shape our perception and our thinking, and the ability to communicate effectively across different linguistic and cultural contexts is an increasingly valuable skill. The development of sophisticated AI-powered translation tools is breaking down language barriers, but the nuance and cultural context embedded in human language often remain challenging for machines to fully grasp.

Ultimately, the human endeavor is a quest for understanding, not just of the external world, but of ourselves. It is a journey marked by triumphs and failures, by moments of profound insight and periods of confusion and doubt. It is a story that is still being written, by each of us, every day. The responsibility to write that story well, to learn from the past, to engage thoughtfully with the present, and to build a better future, rests on our collective shoulders. This requires courage – the courage to question, the courage to change, and the courage to hope. It requires humility – the humility to recognize the limits of our knowledge and the potential for error. And it requires a deep-seated curiosity – the insatiable desire to explore, to discover, and to understand that has driven human progress since the dawn of our species. The path ahead is uncertain, filled with both challenges and opportunities, but it is a path that we must walk together, guided by the light of reason, compassion, and an unwavering commitment to the pursuit of a more enlightened and humane world. The legacy we leave will be defined by how well we navigate this complex, ever-changing landscape, and by the wisdom we cultivate and pass on to future generations.
`
