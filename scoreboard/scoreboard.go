// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package scoreboard creates a scoreboard based on a provider instance.
package scoreboard

import (
	"bytes"
	"context"
	"embed"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
)

//go:embed testdata/*
var testdataFiles embed.FS

// ProviderFactory is a function that returns a provider instance. The name represents the sub-test name.
//
// This may be used for HTTP recording and replays.
type ProviderFactory func(name string) (genai.Provider, http.RoundTripper)

// CreateScenario calculates the supported Scenario for the given provider and its current model.
//
// ProviderFactory must be concurrent safe.
func CreateScenario(ctx context.Context, pf ProviderFactory) (genai.Scenario, genai.Usage, error) {
	usage := genai.Usage{}
	c, _ := pf("")
	m := c.ModelID()
	if m == "" {
		return genai.Scenario{}, usage, errors.New("provider must have a model")
	}
	mu := sync.Mutex{}
	result := genai.Scenario{
		Models: []string{m},
		In:     map[genai.Modality]genai.ModalCapability{},
		Out:    map[genai.Modality]genai.ModalCapability{},
	}

	eg := errgroup.Group{}
	if _, ok := c.(genai.ProviderGen); ok {
		eg.Go(func() error {
			cs := callState{pf: pf, isStream: false}
			ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenSync"))
			f, err := exerciseGenTextOnly(ctx2, &cs, "GenSync-")
			mu.Lock()
			usage.Add(cs.usage)
			if f != nil {
				result.In[genai.ModalityText] = genai.ModalCapability{Inline: true}
				result.Out[genai.ModalityText] = genai.ModalCapability{Inline: true}
				if cs.isThinking {
					result.Thinking = true
				}
				if cs.hasCitations {
					f.Citations = true
				}
				result.GenSync = f
			}
			mu.Unlock()
			if err != nil {
				return fmt.Errorf("failed with GenSync: %w", err)
			}

			cs = callState{pf: pf, isStream: false}
			ctx2 = internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenSyncMultiModal"))
			in, out, f, err := exerciseGenTextMultiModal(ctx2, &cs, "GenSyncMultiModal-")
			mu.Lock()
			usage.Add(cs.usage)
			if len(in) != 0 {
				result.In = mergeModalities(result.In, in)
				result.Out = mergeModalities(result.Out, out)
				if result.GenSync == nil {
					result.GenSync = f
				} else {
					if f.BrokenTokenUsage != genai.False {
						result.GenSync.BrokenTokenUsage = f.BrokenTokenUsage
					}
					if f.BrokenFinishReason {
						result.GenSync.BrokenFinishReason = f.BrokenFinishReason
					}
				}
				if cs.isThinking {
					result.Thinking = true
				}
				if cs.hasCitations {
					result.GenSync.Citations = true
				}
			}
			mu.Unlock()
			if err != nil {
				return fmt.Errorf("failed with GenSync: %w", err)
			}
			return nil
		})
		eg.Go(func() error {
			cs := callState{pf: pf, isStream: true}
			ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenStream"))
			f, err := exerciseGenTextOnly(ctx2, &cs, "GenStream-")
			mu.Lock()
			usage.Add(cs.usage)
			if f != nil {
				result.In[genai.ModalityText] = genai.ModalCapability{Inline: true}
				result.Out[genai.ModalityText] = genai.ModalCapability{Inline: true}
				if cs.isThinking {
					result.Thinking = true
				}
				if cs.hasCitations {
					f.Citations = true
				}
				result.GenStream = f
			}
			mu.Unlock()
			if err != nil {
				return fmt.Errorf("failed with GenStream: %w", err)
			}

			cs = callState{pf: pf, isStream: true}
			ctx2 = internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenStreamMultiModal"))
			in, out, f, err := exerciseGenTextMultiModal(ctx2, &cs, "GenStreamMultiModal-")
			mu.Lock()
			usage.Add(cs.usage)
			if len(in) != 0 {
				result.In = mergeModalities(result.In, in)
				result.Out = mergeModalities(result.Out, out)
				if result.GenStream == nil {
					result.GenStream = f
				} else {
					if f.BrokenTokenUsage != genai.False {
						result.GenStream.BrokenTokenUsage = f.BrokenTokenUsage
					}
					if f.BrokenFinishReason {
						result.GenStream.BrokenFinishReason = f.BrokenFinishReason
					}
				}
				if cs.isThinking {
					result.Thinking = true
				}
				if cs.hasCitations {
					result.GenStream.Citations = true
				}
			}
			mu.Unlock()
			if err != nil {
				return fmt.Errorf("failed with GenStream: %w", err)
			}
			return nil
		})
	}

	if _, ok := c.(genai.ProviderGenDoc); ok {
		eg.Go(func() error {
			ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenDoc"))
			outDoc, usageDoc, err := exerciseGenDoc(ctx2, pf)
			mu.Lock()
			usage.Add(usageDoc)
			result.In = mergeModalities(result.In, outDoc.In)
			result.Out = mergeModalities(result.Out, outDoc.Out)
			result.GenDoc = outDoc.GenDoc
			mu.Unlock()
			if err != nil {
				return fmt.Errorf("failed with GenDoc: %w", err)
			}
			return nil
		})
	}

	err := eg.Wait()
	return result, usage, err
}

// genai.ProviderGen

func exerciseGenTextOnly(ctx context.Context, cs *callState, prefix string) (*genai.FunctionalityText, error) {
	// Make sure simple text generation works, otherwise there's no point.
	msgs := genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
	resp, err := cs.callGen(ctx, prefix+"Text", msgs, &genai.OptionsText{})
	if err != nil {
		internal.Logger(ctx).DebugContext(ctx, "Text", "err", err)
		// It happens when the model is audio gen only.
		if !isBadError(ctx, err) {
			err = nil
		}
		return nil, err
	}
	f := &genai.FunctionalityText{}
	if resp.InputTokens == 0 || resp.OutputTokens == 0 {
		internal.Logger(ctx).DebugContext(ctx, "Text", "issue", "token usage")
		f.BrokenTokenUsage = genai.True
	}
	if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
		internal.Logger(ctx).DebugContext(ctx, "Text", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
		f.BrokenFinishReason = true
	}
	if strings.Contains(resp.AsText(), "<think") {
		return nil, fmt.Errorf("response contains <think: use adapters.ProviderGenThinking")
	}
	if resp.Logprobs != nil {
		return nil, fmt.Errorf("received Logprobs when not supported")
	}
	f.ReportRateLimits = len(resp.Limits) != 0
	for _, l := range resp.Limits {
		if err := l.Validate(); err != nil {
			return nil, err
		}
	}

	// Seed
	msgs = genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
	resp, err = cs.callGen(ctx, prefix+"Seed", msgs, &genai.OptionsText{Seed: 42})
	if isBadError(ctx, err) {
		return f, err
	}
	var uerr *genai.UnsupportedContinuableError
	if errors.As(err, &uerr) {
		f.Seed = !slices.Contains(uerr.Unsupported, "Seed")
	} else if err == nil {
		f.Seed = true
	}

	// TopLogprobs
	msgs = genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
	resp, err = cs.callGen(ctx, prefix+"TopLogprobs", msgs, &genai.OptionsText{TopLogprobs: 2})
	if isBadError(ctx, err) {
		return f, err
	}
	if errors.As(err, &uerr) {
		f.TopLogprobs = !slices.Contains(uerr.Unsupported, "TopLogprobs")
	} else if err == nil {
		f.TopLogprobs = true
	}
	if f.TopLogprobs {
		if resp.Logprobs == nil {
			// It's not actually supported.
			f.TopLogprobs = false
		} else {
			if len(resp.Logprobs.Content) == 0 {
				return nil, fmt.Errorf("received empty Logprobs")
			}
		}
	} else {
		if resp.Logprobs != nil {
			return nil, fmt.Errorf("received Logprobs when not supported")
		}
	}

	// MaxTokens
	// This will trigger citations on providers with search enabled.
	msgs = genai.Messages{genai.NewTextMessage("Explain the theory of relativity in great details.")}
	resp, err = cs.callGen(ctx, prefix+"MaxTokens", msgs, &genai.OptionsText{MaxTokens: 16})
	if isBadError(ctx, err) {
		return f, err
	}
	f.NoMaxTokens = err != nil || strings.Count(resp.AsText(), " ")+1 > 20
	if err == nil {
		if !f.NoMaxTokens && (resp.InputTokens == 0 || resp.OutputTokens == 0) {
			internal.Logger(ctx).DebugContext(ctx, "MaxTokens", "issue", "token usage")
			f.BrokenTokenUsage = genai.True
		}
		expectedFR := genai.FinishedLength
		if f.NoMaxTokens {
			expectedFR = genai.FinishedStop
		}
		if resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "MaxTokens", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	}

	// Stop
	msgs = genai.Messages{genai.NewTextMessage("Talk about Canada in great details. Start with: Canada is")}
	resp, err = cs.callGen(ctx, prefix+"Stop", msgs, &genai.OptionsText{Stop: []string{"is"}})
	if isBadError(ctx, err) {
		return f, err
	}
	f.NoStopSequence = err != nil || strings.Count(resp.AsText(), " ")+1 > 20
	if !f.NoStopSequence && (resp.InputTokens == 0 || resp.OutputTokens == 0) {
		internal.Logger(ctx).DebugContext(ctx, "Stop", "issue", "token usage")
		f.BrokenTokenUsage = genai.True
	}
	if err == nil {
		// Don't fail if unsupported.
		expectedFR := genai.FinishedStopSequence
		if f.NoStopSequence {
			expectedFR = genai.FinishedStop
		}
		if resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "Stop", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	}

	// JSON
	msgs = genai.Messages{genai.NewTextMessage(`Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON according to the provided schema: {"is_fruit": bool}.`)}
	resp, err = cs.callGen(ctx, prefix+"JSON", msgs, &genai.OptionsText{ReplyAsJSON: true})
	if isBadError(ctx, err) {
		return f, err
	}
	if err == nil {
		var data map[string]any
		// We could check for "is_fruit". In practice the fact that it's JSON is good enough to have the flag set.
		f.JSON = resp.Decode(&data) == nil
		if f.JSON {
			if resp.InputTokens == 0 || resp.OutputTokens == 0 {
				internal.Logger(ctx).DebugContext(ctx, "JSON", "issue", "token usage")
				f.BrokenTokenUsage = genai.True
			}
			if expectedFR := genai.FinishedStop; f.JSON && resp.FinishReason != expectedFR {
				internal.Logger(ctx).DebugContext(ctx, "JSON", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
				f.BrokenFinishReason = true
			}
		}
	}

	// JSONSchema
	msgs = genai.Messages{genai.NewTextMessage(`Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON.`)}
	type schema struct {
		IsFruit bool `json:"is_fruit"`
	}
	resp, err = cs.callGen(ctx, prefix+"JSONSchema", msgs, &genai.OptionsText{DecodeAs: &schema{}})
	if isBadError(ctx, err) {
		return f, err
	}
	if err == nil {
		data := schema{}
		f.JSONSchema = resp.Decode(&data) == nil && data.IsFruit
		if f.JSONSchema {
			if resp.InputTokens == 0 || resp.OutputTokens == 0 {
				internal.Logger(ctx).DebugContext(ctx, "JSONSchema", "issue", "token usage")
				f.BrokenTokenUsage = genai.True
			}
			if expectedFR := genai.FinishedStop; f.JSONSchema && resp.FinishReason != expectedFR {
				internal.Logger(ctx).DebugContext(ctx, "JSONSchema", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
				f.BrokenFinishReason = true
			}
		}
	}

	if err = exerciseGenTools(ctx, cs, f, prefix+"Tools-"); err != nil {
		return f, err
	}

	// Citations
	// Force triggering citations for a document provided.
	msgs = genai.Messages{
		genai.Message{
			Role: genai.User,
			Contents: []genai.Content{
				{
					Document: strings.NewReader("The capital of Quackiland is Quack. The Big Canard Statue is located in Quack."),
					Filename: "capital_info.txt",
				},
				{Text: "What is the capital of Quackiland?"},
			},
		},
	}
	resp, err = cs.callGen(ctx, prefix+"Citations", msgs, &genai.OptionsText{})
	if isBadError(ctx, err) {
		return f, err
	}
	if errors.As(err, &uerr) {
		// Cheap trick to make sure the error is not wrapped. Figure out if there's another way!
		if strings.HasPrefix(err.Error(), "unsupported options: ") {
			err = nil
		}
	}
	if err == nil {
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			internal.Logger(ctx).DebugContext(ctx, "Citations", "issue", "token usage")
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; f.Citations && resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, "Citations", "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
	}
	return f, nil
}

func exerciseGenTextMultiModal(ctx context.Context, cs *callState, prefix string) (map[genai.Modality]genai.ModalCapability, map[genai.Modality]genai.ModalCapability, *genai.FunctionalityText, error) {
	in := map[genai.Modality]genai.ModalCapability{}
	out := map[genai.Modality]genai.ModalCapability{}
	f := &genai.FunctionalityText{}
	m, err := exerciseGenInputDocument(ctx, cs, f, prefix+"Document-")
	if m != nil {
		in[genai.ModalityDocument] = *m
		out[genai.ModalityText] = genai.ModalCapability{Inline: true}
	}
	if err != nil {
		return in, out, f, err
	}
	m, err = exerciseGenInputImage(ctx, cs, f, prefix+"Image-")
	if m != nil {
		in[genai.ModalityImage] = *m
		out[genai.ModalityText] = genai.ModalCapability{Inline: true}
	}
	if err != nil {
		return in, out, f, err
	}
	m, err = exerciseGenInputAudio(ctx, cs, f, prefix+"Audio-")
	if m != nil {
		in[genai.ModalityAudio] = *m
		out[genai.ModalityText] = genai.ModalCapability{Inline: true}
	}
	if err != nil {
		return in, out, f, err
	}
	m, err = exerciseGenInputVideo(ctx, cs, f, prefix+"Video-")
	if m != nil {
		in[genai.ModalityVideo] = *m
		out[genai.ModalityText] = genai.ModalCapability{Inline: true}
	}
	return in, out, f, err
}

func exerciseGenInputDocument(ctx context.Context, cs *callState, f *genai.FunctionalityText, prefix string) (*genai.ModalCapability, error) {
	var m *genai.ModalCapability
	want := "orange"
	for _, format := range []extMime{{"pdf", "application/pdf"}} {
		data, err := testdataFiles.ReadFile("testdata/document." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Role: genai.User, Contents: []genai.Content{
			{Text: "What is the word? Reply with only the word."},
			{Document: bytes.NewReader(data), Filename: "document." + format.ext},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.Inline = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(ctx, err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
		msgs[0].Contents[1] = genai.Content{URL: rootURL + "document." + format.ext}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.URL = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(ctx, err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
	}
	return m, nil
}

func exerciseGenInputImage(ctx context.Context, cs *callState, f *genai.FunctionalityText, prefix string) (*genai.ModalCapability, error) {
	var m *genai.ModalCapability
	want := "banana"
	for _, format := range []extMime{
		{"gif", "image/gif"},
		// TODO: {"heic", "image/heic"},
		{"jpg", "image/jpeg"},
		{"png", "image/png"},
		{"webp", "image/webp"},
	} {
		data, err := testdataFiles.ReadFile("testdata/image." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Role: genai.User, Contents: []genai.Content{
			{Text: "What fruit is it? Reply with only one word."},
			{Document: bytes.NewReader(data), Filename: "image." + format.ext},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.Inline = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(ctx, err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
		msgs[0].Contents[1] = genai.Content{URL: rootURL + "image." + format.ext}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.URL = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(ctx, err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
	}
	return m, nil
}

func exerciseGenInputAudio(ctx context.Context, cs *callState, f *genai.FunctionalityText, prefix string) (*genai.ModalCapability, error) {
	var m *genai.ModalCapability
	want := "orange"
	for _, format := range []extMime{
		{"aac", "audio/aac"},
		{"flac", "audio/flac"},
		{"mp3", "audio/mp3"},
		{"ogg", "audio/ogg"},
		{"wav", "audio/wav"},
	} {
		data, err := testdataFiles.ReadFile("testdata/audio." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Role: genai.User, Contents: []genai.Content{
			{Text: "What is the word said? Reply with only the word."},
			{Document: bytes.NewReader(data), Filename: "audio." + format.ext},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.Inline = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(ctx, err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
		msgs[0].Contents[1] = genai.Content{URL: rootURL + "audio." + format.ext}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.URL = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(ctx, err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
	}
	return m, nil
}

func exerciseGenInputVideo(ctx context.Context, cs *callState, f *genai.FunctionalityText, prefix string) (*genai.ModalCapability, error) {
	var m *genai.ModalCapability
	want := "orange"
	for _, format := range []extMime{
		{"mp4", "video/mp4"},
		{"webm", "video/webm"},
	} {
		data, err := testdataFiles.ReadFile("testdata/video." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Role: genai.User, Contents: []genai.Content{
			{Text: "What is the word said? Reply with only the word."},
			{Document: bytes.NewReader(data), Filename: "video." + format.ext},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.Inline = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(ctx, err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
		msgs[0].Contents[1] = genai.Content{URL: rootURL + "video." + format.ext}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &genai.ModalCapability{}
			}
			m.URL = true
			if !slices.Contains(m.SupportedFormats, format.mime) {
				m.SupportedFormats = append(m.SupportedFormats, format.mime)
			}
		} else if isBadError(ctx, err) {
			return m, err
		} else {
			internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		}
	}
	return m, nil
}

type extMime struct {
	ext  string
	mime string
}

func exerciseModal(ctx context.Context, cs *callState, f *genai.FunctionalityText, name string, msgs genai.Messages, want string) error {
	resp, err := cs.callGen(ctx, name, msgs, &genai.OptionsText{})
	if err == nil {
		got := strings.ToLower(strings.TrimRight(strings.TrimSpace(resp.AsText()), "."))
		if want != "" && got != want {
			return fmt.Errorf("got %q, want %q", got, want)
		}
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "token usage")
			f.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			f.BrokenFinishReason = true
		}
		f.ReportRateLimits = len(resp.Limits) != 0
		for _, l := range resp.Limits {
			if err := l.Validate(); err != nil {
				return err
			}
		}
	}
	return err
}

type callState struct {
	pf       ProviderFactory
	isStream bool
	usage    genai.Usage

	// discovered states
	isThinking   bool
	hasCitations bool
}

func (cs *callState) callGen(ctx context.Context, name string, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	cc, _ := cs.pf(name)
	c := cc.(genai.ProviderGen)
	var err error
	var resp genai.Result
	if cs.isStream {
		chunks := make(chan genai.ContentFragment)
		done := make(chan struct{})
		go func() {
			defer close(done)
			for {
				select {
				case <-done:
					return
				case _, ok := <-chunks:
					if !ok {
						return
					}
				}
			}
		}()
		resp, err = c.GenStream(ctx, msgs, chunks, opts)
		close(chunks)
		<-done
	} else {
		resp, err = c.GenSync(ctx, msgs, opts)
	}
	for _, c := range resp.Contents {
		if len(c.Citations) != 0 {
			cs.hasCitations = true
		}
		if c.Thinking != "" {
			cs.isThinking = true
		}
	}
	if err != nil {
		return resp, err
	}
	cs.usage.InputTokens += resp.InputTokens
	cs.usage.InputCachedTokens += resp.InputCachedTokens
	cs.usage.OutputTokens += resp.OutputTokens
	return resp, err
}

// genai.ProviderGenDoc

// exerciseGenDoc exercises the ProviderGenDoc interface if available.
func exerciseGenDoc(ctx context.Context, pf ProviderFactory) (genai.Scenario, genai.Usage, error) {
	prefix := "GenDoc-"
	out := genai.Scenario{
		In:     map[genai.Modality]genai.ModalCapability{},
		Out:    map[genai.Modality]genai.ModalCapability{},
		GenDoc: &genai.FunctionalityDoc{},
	}
	usage := genai.Usage{}
	if err := exerciseGenDocImage(ctx, pf, prefix+"Image", &out, &usage); err != nil {
		return out, usage, err
	}
	if err := exerciseGenDocAudio(ctx, pf, prefix+"Audio", &out, &usage); err != nil {
		return out, usage, err
	}
	if err := exerciseGenDocVideo(ctx, pf, prefix+"Video", &out, &usage); err != nil {
		return out, usage, err
	}
	if len(out.In) == 0 || len(out.Out) == 0 {
		out.GenDoc = nil
	}
	return out, usage, nil
}

func exerciseGenDocImage(ctx context.Context, pf ProviderFactory, name string, out *genai.Scenario, usage *genai.Usage) error {
	cc, rt := pf(name)
	c := cc.(genai.ProviderGenDoc)
	promptImage := `A doodle animation on a white background of Cartoonish shiba inu with brown fur and a white belly, happily eating a pink ice-cream cone, subtle tail wag. Subtle motion but nothing else moves.`
	contentsImage := `Generate one square, white-background doodle with smooth, vibrantly colored image depicting ` + promptImage + `.

*Mandatory Requirements (Compacted):**

**Style:** Simple, vibrant, varied-colored doodle/hand-drawn sketch.
**Background:** Plain solid white (no background colors/elements). Absolutely no black background.
**Content & Motion:** Clearly depict **` + promptImage + `** action with colored, moving subject (no static images). If there's an action specified, it should be the main difference between frames.
**Format:** Square image (1:1 aspect ratio).
**Cropping:** Absolutely no black bars/letterboxing; colorful doodle fully visible against white.
**Output:** Actual image file for a smooth, colorful doodle-style image on a white background.`
	msg := genai.NewTextMessage(contentsImage)
	resp, err := c.GenDoc(ctx, msg, &genai.OptionsImage{Seed: 42})
	usage.InputTokens += resp.InputTokens
	usage.InputCachedTokens += resp.InputCachedTokens
	usage.OutputTokens += resp.OutputTokens
	out.GenDoc.Seed = true
	var uerr *genai.UnsupportedContinuableError
	if errors.As(err, &uerr) {
		// Cheap trick to make sure the error is not wrapped. Figure out if there's another way!
		if strings.HasPrefix(err.Error(), "unsupported options: ") {
			if slices.Contains(uerr.Unsupported, "Seed") {
				out.GenDoc.Seed = false
			}
			err = nil
		}
	}
	if len(resp.Limits) != 0 {
		out.GenDoc.ReportRateLimits = true
		for _, l := range resp.Limits {
			if err := l.Validate(); err != nil {
				return err
			}
		}
	}
	if err == nil {
		if len(resp.Contents) == 0 {
			return fmt.Errorf("%s: no content", name)
		}
		c := resp.Contents[0]
		fn := c.GetFilename()
		if fn == "" {
			return fmt.Errorf("%s: no content filename", name)
		}
		out.In[genai.ModalityText] = genai.ModalCapability{Inline: true}
		v := out.Out[genai.ModalityImage]
		if c.URL != "" {
			v.URL = true
			// Retrieve the result file.
			internal.Logger(ctx).ErrorContext(ctx, name, "rt", fmt.Sprintf("%T", rt))
			resp2, err2 := (&http.Client{Transport: rt}).Get(c.URL)
			if err2 != nil {
				return fmt.Errorf("failed to download generated result: %w", err2)
			}
			defer resp2.Body.Close()
			if resp2.StatusCode != http.StatusOK {
				return fmt.Errorf("failed to download generated result: %s", resp2.Status)
			}
			body, err2 := io.ReadAll(resp2.Body)
			if err2 != nil {
				return fmt.Errorf("failed to download generated result: %w", err2)
			}
			internal.Logger(ctx).DebugContext(ctx, name, "generated", len(body), "url", c.URL)
		} else {
			v.Inline = true
			_, body, err2 := c.ReadDocument(10 * 1024 * 1024)
			if err2 != nil {
				return fmt.Errorf("failed to download generated result: %w", err2)
			}
			internal.Logger(ctx).DebugContext(ctx, name, "generated", len(body))
		}
		v.SupportedFormats = append(v.SupportedFormats, internal.MimeByExt(filepath.Ext(fn)))
		out.Out[genai.ModalityImage] = v
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "token usage")
			out.GenDoc.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			out.GenDoc.BrokenFinishReason = true
		}
	}
	if isBadError(ctx, err) {
		return err
	}
	return nil
}

func exerciseGenDocAudio(ctx context.Context, pf ProviderFactory, name string, out *genai.Scenario, usage *genai.Usage) error {
	cc, rt := pf(name)
	c := cc.(genai.ProviderGenDoc)
	msg := genai.NewTextMessage("Say hi. Just say this word, nothing else.")
	resp, err := c.GenDoc(ctx, msg, &genai.OptionsAudio{Seed: 42})
	usage.InputTokens += resp.InputTokens
	usage.InputCachedTokens += resp.InputCachedTokens
	usage.OutputTokens += resp.OutputTokens
	out.GenDoc.Seed = true
	var uerr *genai.UnsupportedContinuableError
	if errors.As(err, &uerr) {
		// Cheap trick to make sure the error is not wrapped. Figure out if there's another way!
		if strings.HasPrefix(err.Error(), "unsupported options: ") {
			if slices.Contains(uerr.Unsupported, "Seed") {
				out.GenDoc.Seed = false
			}
			err = nil
		}
	}
	if len(resp.Limits) != 0 {
		out.GenDoc.ReportRateLimits = true
		for _, l := range resp.Limits {
			if err := l.Validate(); err != nil {
				return err
			}
		}
	}
	if err == nil {
		if len(resp.Contents) == 0 {
			return fmt.Errorf("%s: no content", name)
		}
		c := resp.Contents[0]
		fn := c.GetFilename()
		if fn == "" {
			return fmt.Errorf("%s: no content filename", name)
		}
		out.In[genai.ModalityText] = genai.ModalCapability{Inline: true}
		out.Out[genai.ModalityAudio] = genai.ModalCapability{Inline: true}
		v := out.Out[genai.ModalityAudio]
		if c.URL != "" {
			v.URL = true
			// Retrieve the result file.
			internal.Logger(ctx).ErrorContext(ctx, name, "rt", fmt.Sprintf("%T", rt))
			resp2, err2 := (&http.Client{Transport: rt}).Get(c.URL)
			if err2 != nil {
				return fmt.Errorf("failed to download generated result: %w", err2)
			}
			defer resp2.Body.Close()
			if resp2.StatusCode != http.StatusOK {
				return fmt.Errorf("failed to download generated result: %s", resp2.Status)
			}
			body, err2 := io.ReadAll(resp2.Body)
			if err2 != nil {
				return fmt.Errorf("failed to download generated result: %w", err2)
			}
			internal.Logger(ctx).DebugContext(ctx, name, "generated", len(body), "url", c.URL)
		} else {
			v.Inline = true
			_, body, err2 := c.ReadDocument(10 * 1024 * 1024)
			if err2 != nil {
				return fmt.Errorf("failed to download generated result: %w", err2)
			}
			internal.Logger(ctx).DebugContext(ctx, name, "generated", len(body))
		}
		v.SupportedFormats = append(v.SupportedFormats, internal.MimeByExt(filepath.Ext(fn)))
		out.Out[genai.ModalityAudio] = v
		if resp.InputTokens == 0 || resp.OutputTokens == 0 {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "token usage")
			out.GenDoc.BrokenTokenUsage = genai.True
		}
		if expectedFR := genai.FinishedStop; resp.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "finish reason", "expected", expectedFR, "got", resp.FinishReason)
			out.GenDoc.BrokenFinishReason = true
		}
	}
	if isBadError(ctx, err) {
		return err
	}
	return nil
}

func exerciseGenDocVideo(ctx context.Context, pf ProviderFactory, name string, out *genai.Scenario, usage *genai.Usage) error {
	// Will be implemented soon.
	return nil
}

const rootURL = "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/scoreboard/testdata/"

func isBadError(ctx context.Context, err error) bool {
	var uerr *httpjson.UnknownFieldError
	if errors.Is(err, cassette.ErrInteractionNotFound) || errors.As(err, &uerr) {
		internal.Logger(ctx).ErrorContext(ctx, "isBadError", "err", err)
		return true
	}
	// API error are never 'bad'.
	var ua base.ErrAPI
	if errors.As(err, &ua) && ua.IsAPIError() {
		return false
	}
	// Tolerate HTTP 400 as when a model is passed something that it doesn't accept, e.g. sending audio input to
	// a text-only model, the provider often respond with 400. 500s should not be tolerated at all.
	var herr *httpjson.Error
	return errors.As(err, &herr) && herr.StatusCode >= 500
}

func mergeModalities(m1, m2 map[genai.Modality]genai.ModalCapability) map[genai.Modality]genai.ModalCapability {
	out := maps.Clone(m1)
	for k, v2 := range m2 {
		if v1, ok := out[k]; ok {
			v1.Inline = v1.Inline || v2.Inline
			v1.URL = v1.URL || v2.URL
			v1.SupportedFormats = mergeSortedUnique(v1.SupportedFormats, v2.SupportedFormats)
			out[k] = v1
			continue
		}
		out[k] = v2
	}
	return out
}

func mergeSortedUnique(s1, s2 []string) []string {
	combined := append(s1, s2...)
	slices.Sort(combined)
	return slices.Compact(combined)
}
