// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package smoke runs a smoke test to generate a scoreboard.Scenario.
package smoke

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
)

// ProviderFactory is a function that returns a provider instance. The name represents the sub-test name.
//
// This may be used for HTTP recording and replays.
type ProviderFactory func(name string) genai.Provider

// Run runs a smoke test on the provider and model.
//
// It returns the supported Scenario as found in practice.
//
// ProviderFactory must be concurrent safe and never fail.
func Run(ctx context.Context, pf ProviderFactory) (scoreboard.Scenario, genai.Usage, error) {
	usage := genai.Usage{}
	c := pf("")
	m := c.ModelID()
	if m == "" {
		return scoreboard.Scenario{}, usage, errors.New("provider must have a model")
	}
	mods := c.OutputModalities()

	mu := sync.Mutex{}
	result := scoreboard.Scenario{
		Models: []string{m},
		In:     map[genai.Modality]scoreboard.ModalCapability{},
		Out:    map[genai.Modality]scoreboard.ModalCapability{},
	}

	// Make it easy to skip parts of the tests.
	doSyncText := slices.Contains(mods, genai.ModalityText)
	doStream := mods[0] == genai.ModalityText
	doSyncNonText := !slices.Equal(mods, genai.Modalities{genai.ModalityText})

	eg := errgroup.Group{}
	if doSyncText || doSyncNonText {
		eg.Go(func() error {
			if doSyncText {
				cs := callState{pf: pf, isStream: false}
				ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenSync"))
				f, err := exerciseGenTextOnly(ctx2, &cs, "GenSync-")
				mu.Lock()
				usage.Add(cs.usage)
				if f != nil {
					result.In[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
					if cs.isReasoning {
						result.Reason = true
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
						if f.ReportTokenUsage != scoreboard.False {
							result.GenSync.ReportTokenUsage = f.ReportTokenUsage
						}
						if f.ReportFinishReason != scoreboard.False {
							result.GenSync.ReportFinishReason = f.ReportFinishReason
						}
					}
					if cs.isReasoning {
						result.Reason = true
					}
					if cs.hasCitations {
						result.GenSync.Citations = true
					}
				}
				mu.Unlock()
				if err != nil {
					return fmt.Errorf("failed with GenSync: %w", err)
				}
			}

			if doSyncNonText {
				eg.Go(func() error {
					// TODO: Rename because GenDoc is not used in the code anymore.
					ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenDoc"))
					outDoc, usageDoc, err := exerciseGenNonText(ctx2, pf)
					mu.Lock()
					usage.Add(usageDoc)
					result.In = mergeModalities(result.In, outDoc.In)
					result.Out = mergeModalities(result.Out, outDoc.Out)
					if result.GenSync == nil {
						result.GenSync = outDoc.GenSync
					} else if outDoc.GenSync != nil {
						// This is tricky, we need to merge the results. The only case where this is known to happen is
						// gemini-2.5-flash-image-preview. Keep the panic so I don't forget about it.
						panic("implement support for multi-modal output")
					}
					mu.Unlock()
					if err != nil {
						return fmt.Errorf("failed with GenDoc: %w", err)
					}
					return nil
				})
			}

			return nil
		})
	}
	if doStream {
		eg.Go(func() error {
			cs := callState{pf: pf, isStream: true}
			ctx2 := internal.WithLogger(ctx, internal.Logger(ctx).With("fn", "GenStream"))
			f, err := exerciseGenTextOnly(ctx2, &cs, "GenStream-")
			mu.Lock()
			usage.Add(cs.usage)
			if f != nil {
				result.In[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
				result.Out[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
				if cs.isReasoning {
					result.Reason = true
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
					if f.ReportTokenUsage != scoreboard.False {
						result.GenStream.ReportTokenUsage = f.ReportTokenUsage
					}
					if f.ReportFinishReason != scoreboard.False {
						result.GenStream.ReportFinishReason = f.ReportFinishReason
					}
				}
				if cs.isReasoning {
					result.Reason = true
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

	err := eg.Wait()
	return result, usage, err
}

// genai.Provider

func exerciseGenTextOnly(ctx context.Context, cs *callState, prefix string) (*scoreboard.Functionality, error) {
	// Make sure simple text generation works, otherwise there's no point.
	ctxCheck := internal.WithLogger(ctx, internal.Logger(ctx).With("check", "Text"))
	msgs := genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
	resp, err := cs.callGen(ctxCheck, prefix+"Text", msgs)
	if err != nil {
		// It happens when the model is audio gen only.
		if !isBadError(ctxCheck, err) {
			err = nil
		}
		return nil, err
	}
	f := &scoreboard.Functionality{}
	// Optimistically assume it works.
	f.ReportTokenUsage = scoreboard.True
	f.ReportFinishReason = scoreboard.True
	if isZeroUsage(resp.Usage) {
		// If it fails at this one, it's never going to succeed.
		internal.Logger(ctxCheck).DebugContext(ctxCheck, "no usage")
		f.ReportTokenUsage = scoreboard.False
	}
	if expectedFR := genai.FinishedStop; resp.Usage.FinishReason != expectedFR {
		internal.Logger(ctxCheck).DebugContext(ctxCheck, "bad finishreason", "expected", expectedFR, "got", resp.Usage.FinishReason)
		f.ReportFinishReason = scoreboard.False
	}
	if strings.Contains(resp.String(), "<think") {
		return nil, fmt.Errorf("response contains <think: use adapters.ProviderReasoning")
	}
	if len(resp.Logprobs) != 0 {
		return nil, fmt.Errorf("received Logprobs when not supported")
	}
	f.ReportRateLimits = len(resp.Usage.Limits) != 0
	for _, l := range resp.Usage.Limits {
		if err = l.Validate(); err != nil {
			return nil, err
		}
	}

	// Seed
	ctxCheck = internal.WithLogger(ctx, internal.Logger(ctx).With("check", "Seed"))
	msgs = genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
	resp, err = cs.callGen(ctxCheck, prefix+"Seed", msgs, &genai.OptionsText{Seed: 42})
	if isBadError(ctxCheck, err) {
		return f, err
	}
	var uerr *genai.UnsupportedContinuableError
	if errors.As(err, &uerr) {
		f.Seed = !slices.Contains(uerr.Unsupported, "OptionsText.Seed")
	} else if err == nil {
		f.Seed = true
	}

	// TopLogprobs
	ctxCheck = internal.WithLogger(ctx, internal.Logger(ctx).With("check", "TopLogprobs"))
	msgs = genai.Messages{genai.NewTextMessage("Say hello. Use only one word.")}
	resp, err = cs.callGen(ctxCheck, prefix+"TopLogprobs", msgs, &genai.OptionsText{TopLogprobs: 2})
	if isBadError(ctxCheck, err) {
		return f, err
	}
	if errors.As(err, &uerr) {
		f.TopLogprobs = !slices.Contains(uerr.Unsupported, "OptionsText.TopLogprobs")
	} else if err == nil {
		// TODO: We'll need to be more detailed than that. Most don't report the ID or bytes, some only report
		// logprobs, etc.
		f.TopLogprobs = true
	}
	if f.TopLogprobs {
		if len(resp.Logprobs) == 0 {
			// It's not actually supported.
			f.TopLogprobs = false
		} else {
			if len(resp.Logprobs) == 0 {
				return nil, fmt.Errorf("received empty Logprobs")
			}
		}
	} else {
		if len(resp.Logprobs) != 0 {
			// This one is weird, it happens with cohere. Is it "supported" when it returns the logprobs but not
			// output and no alternatives from the one selected?
			// return nil, fmt.Errorf("received Logprobs when not supported")
		}
	}

	// MaxTokens
	// This will trigger citations on providers with search enabled.
	ctxCheck = internal.WithLogger(ctx, internal.Logger(ctx).With("check", "MaxTokens"))
	msgs = genai.Messages{genai.NewTextMessage("Explain the theory of relativity in great details.")}
	resp, err = cs.callGen(ctxCheck, prefix+"MaxTokens", msgs, &genai.OptionsText{MaxTokens: 16})
	if isBadError(ctxCheck, err) {
		return f, err
	}
	// MaxTokens is reported as false if there's no returned output.
	f.MaxTokens = err == nil && strings.Count(resp.String(), " ")+1 <= 20 && (len(resp.String()) > 0 || len(resp.Reasoning()) > 0)
	if err == nil {
		if f.MaxTokens && isZeroUsage(resp.Usage) {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctxCheck).DebugContext(ctxCheck, "no usage")
				f.ReportTokenUsage = scoreboard.Flaky
			}
		}
		expectedFR := genai.FinishedLength
		if !f.MaxTokens {
			expectedFR = genai.FinishedStop
		}
		if resp.Usage.FinishReason != expectedFR {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctxCheck).DebugContext(ctxCheck, "bad finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
				f.ReportFinishReason = scoreboard.Flaky
			}
		}
	}

	// Stop
	ctxCheck = internal.WithLogger(ctx, internal.Logger(ctx).With("check", "Stop"))
	msgs = genai.Messages{genai.NewTextMessage("Talk about Canada in great details. Start with: Canada is")}
	resp, err = cs.callGen(ctxCheck, prefix+"Stop", msgs, &genai.OptionsText{Stop: []string{"is"}})
	if isBadError(ctxCheck, err) {
		return f, err
	}
	// Some model (in particular Gemma3 4B in Q4_K_M) will not start with "Canada is". They will still stop but
	// only after a while.
	f.StopSequence = err == nil && strings.Count(resp.String(), " ")+1 <= 30 && !strings.Contains(resp.String(), " is ")
	if f.StopSequence && isZeroUsage(resp.Usage) {
		if f.ReportTokenUsage != scoreboard.False {
			internal.Logger(ctxCheck).DebugContext(ctxCheck, "no usage")
			f.ReportTokenUsage = scoreboard.Flaky
		}
	}
	if err == nil {
		// Don't fail if unsupported.
		expectedFR := genai.FinishedStopSequence
		if !f.StopSequence {
			expectedFR = genai.FinishedStop
		}
		if resp.Usage.FinishReason != expectedFR {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctxCheck).DebugContext(ctxCheck, "bad finish reason", "expected", genai.FinishedStopSequence, "got", resp.Usage.FinishReason)
				f.ReportFinishReason = scoreboard.Flaky
			}
		}
	}

	// JSON
	ctxCheck = internal.WithLogger(ctx, internal.Logger(ctx).With("check", "JSON"))
	msgs = genai.Messages{genai.NewTextMessage(`Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON according to the provided schema: {"is_fruit": bool}.`)}
	resp, err = cs.callGen(ctxCheck, prefix+"JSON", msgs, &genai.OptionsText{ReplyAsJSON: true})
	if isBadError(ctxCheck, err) {
		return f, err
	}
	if err == nil {
		var data map[string]any
		// We could check for "is_fruit". In practice the fact that it's JSON is good enough to have the flag set.
		f.JSON = resp.Decode(&data) == nil
		if f.JSON {
			if isZeroUsage(resp.Usage) {
				if f.ReportTokenUsage != scoreboard.False {
					internal.Logger(ctxCheck).DebugContext(ctxCheck, "no usage")
					f.ReportTokenUsage = scoreboard.Flaky
				}
			}
			if expectedFR := genai.FinishedStop; f.JSON && resp.Usage.FinishReason != expectedFR {
				if f.ReportTokenUsage != scoreboard.False {
					internal.Logger(ctxCheck).DebugContext(ctxCheck, "bad finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
					f.ReportFinishReason = scoreboard.Flaky
				}
			}
		}
	}

	// JSONSchema
	ctxCheck = internal.WithLogger(ctx, internal.Logger(ctx).With("check", "JSONSchema"))
	msgs = genai.Messages{genai.NewTextMessage(`Is a banana a fruit? Do not include an explanation. Reply ONLY as JSON.`)}
	type schema struct {
		IsFruit bool `json:"is_fruit"`
	}
	resp, err = cs.callGen(ctxCheck, prefix+"JSONSchema", msgs, &genai.OptionsText{DecodeAs: &schema{}})
	if isBadError(ctxCheck, err) {
		return f, err
	}
	if err == nil {
		data := schema{}
		f.JSONSchema = resp.Decode(&data) == nil && data.IsFruit
		if f.JSONSchema {
			if isZeroUsage(resp.Usage) {
				if f.ReportTokenUsage != scoreboard.False {
					internal.Logger(ctxCheck).DebugContext(ctxCheck, "no usage")
					f.ReportTokenUsage = scoreboard.Flaky
				}
			}
			if expectedFR := genai.FinishedStop; f.JSONSchema && resp.Usage.FinishReason != expectedFR {
				if f.ReportTokenUsage != scoreboard.False {
					internal.Logger(ctxCheck).DebugContext(ctxCheck, "bad finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
					f.ReportFinishReason = scoreboard.Flaky
				}
			}
		}
	}

	if err = exerciseGenTools(ctx, cs, f, prefix+"Tools-"); err != nil {
		return f, err
	}

	// Citations
	// Force triggering citations for a document provided.
	ctxCheck = internal.WithLogger(ctx, internal.Logger(ctx).With("check", "Citations", "type", "text/plain"))
	hasCitations := false
	msgs = genai.Messages{
		genai.Message{
			Requests: []genai.Request{
				{
					Doc: genai.Doc{
						Src:      strings.NewReader("The capital of Quackiland is Quack. The Big Canard Statue is located in Quack."),
						Filename: "capital_info.txt",
					},
				},
				{Text: "What is the capital of Quackiland?"},
			},
		},
	}
	resp, err = cs.callGen(ctxCheck, prefix+"Citations-text-plain", msgs)
	if isBadError(ctxCheck, err) {
		return f, err
	}
	if errors.As(err, &uerr) {
		// Cheap trick to make sure the error is not wrapped. Figure out if there's another way!
		if strings.HasPrefix(err.Error(), "unsupported options: ") {
			err = nil
		}
	}
	if err == nil {
		hasCitations = true
		if isZeroUsage(resp.Usage) {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctxCheck).DebugContext(ctxCheck, "no usage")
				f.ReportTokenUsage = scoreboard.Flaky
			}
		}
		if expectedFR := genai.FinishedStop; f.Citations && resp.Usage.FinishReason != expectedFR {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctxCheck).DebugContext(ctxCheck, "bad finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
				f.ReportFinishReason = scoreboard.Flaky
			}
		}
	} else {
		// The client code must provide the text as inline content.
		return f, fmt.Errorf("providers must support text/plain: %w", err)
	}

	// Try again with markdown. This is important to make sure the client doesn't lock onto text/plain.
	ctxCheck = internal.WithLogger(ctx, internal.Logger(ctx).With("check", "Citations", "type", "text/markdown"))
	msgs = genai.Messages{
		genai.Message{
			Requests: []genai.Request{
				{
					Doc: genai.Doc{
						Src:      strings.NewReader("# Quackiland\n\nThe capital of Quackiland is Quack. The Big Canard Statue is located in Quack."),
						Filename: "capital_info.md",
					},
				},
				{Text: "What is the capital of Quackiland?"},
			},
		},
	}
	resp, err = cs.callGen(ctxCheck, prefix+"Citations-text-markdown", msgs)
	if isBadError(ctxCheck, err) {
		return f, err
	}
	if errors.As(err, &uerr) {
		// Cheap trick to make sure the error is not wrapped. Figure out if there's another way!
		if strings.HasPrefix(err.Error(), "unsupported options: ") {
			err = nil
		}
	}
	if err == nil {
		if isZeroUsage(resp.Usage) {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctxCheck).DebugContext(ctxCheck, "no usage")
				f.ReportTokenUsage = scoreboard.Flaky
			}
		}
		if expectedFR := genai.FinishedStop; f.Citations && resp.Usage.FinishReason != expectedFR {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctxCheck).DebugContext(ctxCheck, "bad finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
				f.ReportFinishReason = scoreboard.Flaky
			}
		}
	} else if hasCitations {
		return f, fmt.Errorf("supports text/plain but not text/markdown; this is incorrect: %w", err)
	}

	return f, nil
}

func exerciseGenTextMultiModal(ctx context.Context, cs *callState, prefix string) (map[genai.Modality]scoreboard.ModalCapability, map[genai.Modality]scoreboard.ModalCapability, *scoreboard.Functionality, error) {
	in := map[genai.Modality]scoreboard.ModalCapability{}
	out := map[genai.Modality]scoreboard.ModalCapability{}
	f := &scoreboard.Functionality{}
	m, err := exerciseGenInputDocument(ctx, cs, f, prefix+"Document-")
	if m != nil {
		in[genai.ModalityDocument] = *m
		out[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
	}
	if err != nil {
		return in, out, f, err
	}
	m, err = exerciseGenInputImage(ctx, cs, f, prefix+"Image-")
	if m != nil {
		in[genai.ModalityImage] = *m
		out[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
	}
	if err != nil {
		return in, out, f, err
	}
	m, err = exerciseGenInputAudio(ctx, cs, f, prefix+"Audio-")
	if m != nil {
		in[genai.ModalityAudio] = *m
		out[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
	}
	if err != nil {
		return in, out, f, err
	}
	m, err = exerciseGenInputVideo(ctx, cs, f, prefix+"Video-")
	if m != nil {
		in[genai.ModalityVideo] = *m
		out[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
	}
	return in, out, f, err
}

func exerciseGenInputDocument(ctx context.Context, cs *callState, f *scoreboard.Functionality, prefix string) (*scoreboard.ModalCapability, error) {
	var m *scoreboard.ModalCapability
	want := regexp.MustCompile("^orange$")
	for _, format := range []extMime{{"pdf", "application/pdf"}} {
		data, err := scoreboard.TestdataFiles.ReadFile("testdata/document." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Requests: []genai.Request{
			{Text: "What is the word? Reply with only the word."},
			{Doc: genai.Doc{Src: bytes.NewReader(data), Filename: "document." + format.ext}},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &scoreboard.ModalCapability{}
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
		msgs[0].Requests[1] = genai.Request{Doc: genai.Doc{URL: rootURL + "document." + format.ext}}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &scoreboard.ModalCapability{}
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

func exerciseGenInputImage(ctx context.Context, cs *callState, f *scoreboard.Functionality, prefix string) (*scoreboard.ModalCapability, error) {
	var m *scoreboard.ModalCapability
	want := regexp.MustCompile("^banana$")
	for _, format := range []extMime{
		{"gif", "image/gif"},
		// TODO: {"heic", "image/heic"},
		{"jpg", "image/jpeg"},
		{"png", "image/png"},
		{"webp", "image/webp"},
	} {
		data, err := scoreboard.TestdataFiles.ReadFile("testdata/image." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Requests: []genai.Request{
			{Text: "What fruit is it? Reply with only one word."},
			{Doc: genai.Doc{Src: bytes.NewReader(data), Filename: "image." + format.ext}},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &scoreboard.ModalCapability{}
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
		msgs[0].Requests[1] = genai.Request{Doc: genai.Doc{URL: rootURL + "image." + format.ext}}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &scoreboard.ModalCapability{}
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

func exerciseGenInputAudio(ctx context.Context, cs *callState, f *scoreboard.Functionality, prefix string) (*scoreboard.ModalCapability, error) {
	var m *scoreboard.ModalCapability
	want := regexp.MustCompile("^orange$")
	for _, format := range []extMime{
		{"aac", "audio/aac"},
		{"flac", "audio/flac"},
		{"mp3", "audio/mp3"},
		{"ogg", "audio/ogg"},
		{"wav", "audio/wav"},
	} {
		data, err := scoreboard.TestdataFiles.ReadFile("testdata/audio." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Requests: []genai.Request{
			{Text: "What is the word said? Reply with only the word."},
			{Doc: genai.Doc{Src: bytes.NewReader(data), Filename: "audio." + format.ext}},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &scoreboard.ModalCapability{}
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
		msgs[0].Requests[1] = genai.Request{Doc: genai.Doc{URL: rootURL + "audio." + format.ext}}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &scoreboard.ModalCapability{}
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

func exerciseGenInputVideo(ctx context.Context, cs *callState, f *scoreboard.Functionality, prefix string) (*scoreboard.ModalCapability, error) {
	var m *scoreboard.ModalCapability
	// Relax the check, as the model will try to describe the fact that the word is rotating.
	want := regexp.MustCompile("banana")
	for _, format := range []extMime{
		{"mp4", "video/mp4"},
		{"webm", "video/webm"},
	} {
		data, err := scoreboard.TestdataFiles.ReadFile("testdata/video." + format.ext)
		if err != nil {
			return nil, fmt.Errorf("failed to open input file: %w", err)
		}
		msgs := genai.Messages{genai.Message{Requests: []genai.Request{
			{Text: "What is the word said? Reply with only the word."},
			{Doc: genai.Doc{Src: bytes.NewReader(data), Filename: "video." + format.ext}},
		}}}
		name := prefix + format.ext + "-Inline"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &scoreboard.ModalCapability{}
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
		msgs[0].Requests[1] = genai.Request{Doc: genai.Doc{URL: rootURL + "video." + format.ext}}
		name = prefix + format.ext + "-URL"
		if err = exerciseModal(ctx, cs, f, name, msgs, want); err == nil {
			if m == nil {
				m = &scoreboard.ModalCapability{}
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

func exerciseModal(ctx context.Context, cs *callState, f *scoreboard.Functionality, name string, msgs genai.Messages, want *regexp.Regexp) error {
	resp, err := cs.callGen(ctx, name, msgs)
	if err == nil {
		got := strings.ToLower(strings.TrimRight(strings.TrimSpace(resp.String()), "."))
		if want != nil && !want.MatchString(got) {
			return fmt.Errorf("got %q, want %q", got, want)
		}
		if isZeroUsage(resp.Usage) {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctx).DebugContext(ctx, name, "issue", "token usage")
				f.ReportTokenUsage = scoreboard.Flaky
			}
		}
		if expectedFR := genai.FinishedStop; resp.Usage.FinishReason != expectedFR {
			if f.ReportTokenUsage != scoreboard.False {
				internal.Logger(ctx).DebugContext(ctx, name, "issue", "finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
				f.ReportFinishReason = scoreboard.Flaky
			}
		}
		f.ReportRateLimits = len(resp.Usage.Limits) != 0
		for _, l := range resp.Usage.Limits {
			if err = l.Validate(); err != nil {
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
	isReasoning  bool
	hasCitations bool
}

func (cs *callState) callGen(ctx context.Context, name string, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	// internal.Logger(ctx).DebugContext(ctx, name, "msgs", msgs)
	c := cs.pf(name)
	var err error
	res := genai.Result{}
	if cs.isStream {
		fragments, finish := c.GenStream(ctx, msgs, opts...)
		for range fragments {
		}
		res, err = finish()
	} else {
		res, err = c.GenSync(ctx, msgs, opts...)
	}
	for _, c := range res.Replies {
		if len(c.Citations) != 0 {
			cs.hasCitations = true
		}
		if c.Reasoning != "" {
			cs.isReasoning = true
		}
	}
	if err != nil {
		internal.Logger(ctx).DebugContext(ctx, name, "err", err)
		return res, err
	}
	cs.usage.InputTokens += res.Usage.InputTokens
	cs.usage.InputCachedTokens += res.Usage.InputCachedTokens
	cs.usage.OutputTokens += res.Usage.OutputTokens
	return res, err
}

// Non text modalities

// exerciseGenNonText exercises the non-text functionality of genai.Provider.
//
// It only tests GenSync, not GenStream. Most non-text modalities are fairly expensive to run (some cost a few
// bucks). If you have a business case and would like to pay me to test more, please let me know.
func exerciseGenNonText(ctx context.Context, pf ProviderFactory) (scoreboard.Scenario, genai.Usage, error) {
	prefix := "GenDoc-"
	out := scoreboard.Scenario{
		In:      map[genai.Modality]scoreboard.ModalCapability{},
		Out:     map[genai.Modality]scoreboard.ModalCapability{},
		GenSync: &scoreboard.Functionality{},
	}
	usage := genai.Usage{}
	if err := exerciseGenImage(ctx, pf, prefix+"Image", &out, &usage); err != nil {
		return out, usage, err
	}
	if err := exerciseGenAudio(ctx, pf, prefix+"Audio", &out, &usage); err != nil {
		return out, usage, err
	}
	if err := exerciseGenVideo(ctx, pf, prefix+"Video", &out, &usage); err != nil {
		return out, usage, err
	}
	if len(out.In) == 0 || len(out.Out) == 0 {
		out.GenSync = nil
	}
	return out, usage, nil
}

func exerciseGenImage(ctx context.Context, pf ProviderFactory, name string, out *scoreboard.Scenario, usage *genai.Usage) error {
	c := pf(name)
	if !slices.Contains(c.OutputModalities(), genai.ModalityImage) {
		return nil
	}
	promptImage := `A doodle animation on a white background of Cartoonish shiba inu with brown fur and a white belly, happily eating a pink ice-cream cone, subtle tail wag. Subtle motion but nothing else moves.`
	contentsImage := `Generate one square, white-background doodle with smooth, vibrantly colored image depicting ` + promptImage + `.

*Mandatory Requirements (Compacted):**

**Style:** Simple, vibrant, varied-colored doodle/hand-drawn sketch.
**Background:** Plain solid white (no background colors/elements). Absolutely no black background.
**Content & Motion:** Clearly depict **` + promptImage + `** action with colored, moving subject (no static images). If there's an action specified, it should be the main difference between frames.
**Format:** Square image (1:1 aspect ratio).
**Cropping:** Absolutely no black bars/letterboxing; colorful doodle fully visible against white.
**Output:** Actual image file for a smooth, colorful doodle-style image on a white background.`
	msgs := genai.Messages{genai.NewTextMessage(contentsImage)}
	resp, err := c.GenSync(ctx, msgs, &genai.OptionsImage{Seed: 42})
	usage.InputTokens += resp.Usage.InputTokens
	usage.InputCachedTokens += resp.Usage.InputCachedTokens
	usage.OutputTokens += resp.Usage.OutputTokens
	out.GenSync.Seed = true
	var uerr *genai.UnsupportedContinuableError
	if errors.As(err, &uerr) {
		// Cheap trick to make sure the error is not wrapped. Figure out if there's another way!
		if strings.HasPrefix(err.Error(), "unsupported options: ") {
			if slices.Contains(uerr.Unsupported, "OptionsText.Seed") {
				out.GenSync.Seed = false
			}
			err = nil
		}
	}
	if len(resp.Usage.Limits) != 0 {
		out.GenSync.ReportRateLimits = true
		for _, l := range resp.Usage.Limits {
			if err2 := l.Validate(); err2 != nil {
				return err2
			}
		}
	}
	if err == nil {
		if len(resp.Replies) == 0 {
			return fmt.Errorf("%s: no content", name)
		}
		r := resp.Replies[0]
		fn := r.Doc.GetFilename()
		if fn == "" {
			return fmt.Errorf("%s: no content filename", name)
		}
		out.In[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
		v := out.Out[genai.ModalityImage]
		if r.Doc.URL != "" {
			v.URL = true
			// Retrieve the result file.
			resp2, err2 := c.HTTPClient().Get(r.Doc.URL)
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
			internal.Logger(ctx).DebugContext(ctx, name, "generated", len(body), "url", r.Doc.URL)
		} else {
			v.Inline = true
			_, body, err2 := r.Doc.Read(10 * 1024 * 1024)
			if err2 != nil {
				return fmt.Errorf("failed to download generated result: %w", err2)
			}
			internal.Logger(ctx).DebugContext(ctx, name, "generated", len(body))
		}
		v.SupportedFormats = append(v.SupportedFormats, base.MimeByExt(filepath.Ext(fn)))
		out.Out[genai.ModalityImage] = v
		if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "token usage")
			out.GenSync.ReportTokenUsage = scoreboard.False
		} else {
			// TODO: incorrect.
			out.GenSync.ReportTokenUsage = scoreboard.True
		}
		if expectedFR := genai.FinishedStop; resp.Usage.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
			out.GenSync.ReportFinishReason = scoreboard.False
		} else {
			// TODO: incorrect.
			out.GenSync.ReportFinishReason = scoreboard.True
		}
	} else {
		if isBadError(ctx, err) {
			return err
		}
		internal.Logger(ctx).DebugContext(ctx, name, "err", err)
	}
	return nil
}

func exerciseGenAudio(ctx context.Context, pf ProviderFactory, name string, out *scoreboard.Scenario, usage *genai.Usage) error {
	c := pf(name)
	if !slices.Contains(c.OutputModalities(), genai.ModalityAudio) {
		return nil
	}
	msgs := genai.Messages{genai.NewTextMessage("Say hi. Just say this word, nothing else.")}
	resp, err := c.GenSync(ctx, msgs, &genai.OptionsAudio{Seed: 42})
	usage.InputTokens += resp.Usage.InputTokens
	usage.InputCachedTokens += resp.Usage.InputCachedTokens
	usage.OutputTokens += resp.Usage.OutputTokens
	out.GenSync.Seed = true
	var uerr *genai.UnsupportedContinuableError
	if errors.As(err, &uerr) {
		// Cheap trick to make sure the error is not wrapped. Figure out if there's another way!
		if strings.HasPrefix(err.Error(), "unsupported options: ") {
			if slices.Contains(uerr.Unsupported, "OptionsText.Seed") {
				out.GenSync.Seed = false
			}
			err = nil
		}
	}
	if len(resp.Usage.Limits) != 0 {
		out.GenSync.ReportRateLimits = true
		for _, l := range resp.Usage.Limits {
			if err2 := l.Validate(); err2 != nil {
				return err2
			}
		}
	}
	if err == nil {
		if len(resp.Replies) == 0 {
			return fmt.Errorf("%s: no content", name)
		}
		r := resp.Replies[0]
		fn := r.Doc.GetFilename()
		if fn == "" {
			return fmt.Errorf("%s: no content filename", name)
		}
		out.In[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
		out.Out[genai.ModalityAudio] = scoreboard.ModalCapability{Inline: true}
		v := out.Out[genai.ModalityAudio]
		if r.Doc.URL != "" {
			v.URL = true
			// Retrieve the result file.
			resp2, err2 := c.HTTPClient().Get(r.Doc.URL)
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
			internal.Logger(ctx).DebugContext(ctx, name, "generated", len(body), "url", r.Doc.URL)
		} else {
			v.Inline = true
			_, body, err2 := r.Doc.Read(10 * 1024 * 1024)
			if err2 != nil {
				return fmt.Errorf("failed to download generated result: %w", err2)
			}
			internal.Logger(ctx).DebugContext(ctx, name, "generated", len(body))
		}
		v.SupportedFormats = append(v.SupportedFormats, base.MimeByExt(filepath.Ext(fn)))
		out.Out[genai.ModalityAudio] = v
		if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "token usage")
			out.GenSync.ReportTokenUsage = scoreboard.False
		} else {
			// TODO: incorrect.
			out.GenSync.ReportTokenUsage = scoreboard.True
		}
		if expectedFR := genai.FinishedStop; resp.Usage.FinishReason != expectedFR {
			internal.Logger(ctx).DebugContext(ctx, name, "issue", "finish reason", "expected", expectedFR, "got", resp.Usage.FinishReason)
			out.GenSync.ReportFinishReason = scoreboard.False
		} else {
			// TODO: incorrect.
			out.GenSync.ReportFinishReason = scoreboard.True
		}
	} else {
		if isBadError(ctx, err) {
			return err
		}
		internal.Logger(ctx).DebugContext(ctx, name, "err", err)
	}
	return nil
}

func exerciseGenVideo(ctx context.Context, pf ProviderFactory, name string, out *scoreboard.Scenario, usage *genai.Usage) error {
	// Will be implemented soon.
	return nil
}

const rootURL = "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/scoreboard/testdata/"

func isBadError(ctx context.Context, err error) bool {
	if errors.Is(err, cassette.ErrInteractionNotFound) {
		internal.Logger(ctx).ErrorContext(ctx, "isBadError", "type", "cassette", "err", err)
		return true
	}
	var uerr *httpjson.UnknownFieldError
	if errors.Is(err, cassette.ErrInteractionNotFound) || errors.As(err, &uerr) {
		internal.Logger(ctx).ErrorContext(ctx, "isBadError", "type", "UnknownFieldError", "err", err)
		return true
	}
	var be *internal.BadError
	if errors.As(err, &be) {
		internal.Logger(ctx).ErrorContext(ctx, "isBadError", "type", "BadError", "err", err)
		return true
	}

	// API error are never 'bad'.
	var ua base.ErrAPI
	if errors.As(err, &ua) && ua.IsAPIError() {
		internal.Logger(ctx).DebugContext(ctx, "isBadError", "type", "ErrAPI")
		return false
	}
	// Tolerate HTTP 400 as when a model is passed something that it doesn't accept, e.g. sending audio input to
	// a text-only model, the provider often respond with 400. 500s should not be tolerated at all.
	var herr *httpjson.Error
	if errors.As(err, &herr) && herr.StatusCode >= 500 {
		internal.Logger(ctx).DebugContext(ctx, "isBadError", "status", herr.StatusCode)
		return true
	}
	return false
}

func mergeModalities(m1, m2 map[genai.Modality]scoreboard.ModalCapability) map[genai.Modality]scoreboard.ModalCapability {
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

func isZeroUsage(u genai.Usage) bool {
	// Tolerate cases like OpenAI Responses API with gpt-5 in thinking mode with MaxTokens will only return
	// TotalTokens being set. Good enough.
	return u.InputTokens == 0 &&
		u.InputCachedTokens == 0 &&
		u.ReasoningTokens == 0 &&
		u.OutputTokens == 0 &&
		u.TotalTokens == 0
}
