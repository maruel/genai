// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Transcription session probes for genai.RealtimeOptionTranscription.

package smoke

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
)

// RealtimeProbeTranscription transcribes speech in a transcription session.
const RealtimeProbeTranscription RealtimeProbe = "transcription"

// transcriptionWords are words of the code fixture's English speech that a
// transcript must contain.
const transcriptionWords = "access code"

// RunTranscriptionScenario qualifies a transcription session for the
// provider's configured model.
//
// The first session transcribes the English speech fixture and must return a
// final transcript containing its words, followed by TurnDone. Drafts records
// whether transcripts arrived before the final one. A second session sets a
// mismatched InputLanguage, French: a provider that rejects it with
// base.ErrNotSupported does not accept a language; otherwise the session must
// end its turn, and ForcesLanguage records whether the transcript lost the
// English words. Audio is paced at playback speed unless the session replays a
// recording, so a live server decodes it as it is spoken.
func RunTranscriptionScenario(ctx context.Context, c genai.Provider) (scoreboard.Realtime, genai.Usage, error) {
	model := c.ModelID()
	sc := scoreboard.Realtime{Models: []string{model}}
	if model == "" {
		return sc, genai.Usage{}, errors.New("provider must have a model")
	}
	if !c.Capabilities().Realtime {
		return sc, genai.Usage{}, errors.New("provider does not support realtime")
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	t := scoreboard.RealtimeTranscription{}
	info, final, drafts, err := transcribe(ctx, c, "")
	if err != nil {
		return sc, genai.Usage{}, err
	}
	if !strings.Contains(realtimeNormalizeAnswer(final), transcriptionWords) {
		return sc, genai.Usage{}, fmt.Errorf("transcript %q lacks %q", final, transcriptionWords)
	}
	t.Drafts = drafts
	_, final, _, err = transcribe(ctx, c, "fr")
	if _, ok := errors.AsType[*base.ErrNotSupported](err); !ok {
		if err != nil {
			return sc, genai.Usage{}, fmt.Errorf("mismatched language: %w", err)
		}
		t.InputLanguage = true
		t.ForcesLanguage = !strings.Contains(realtimeNormalizeAnswer(final), transcriptionWords)
	}
	sc.In = map[genai.Modality]scoreboard.ModalCapability{genai.ModalityAudio: {Inline: true}}
	sc.Out = map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}}
	sc.Realtime = &scoreboard.RealtimeFunctionality{InputSampleRate: info.InputFormat.SampleRate, Transcription: &t}
	return sc, genai.Usage{}, nil
}

// transcribe sends the English speech fixture through one transcription
// session in lang and returns the final transcript, which may be empty, and
// whether transcripts preceded it.
func transcribe(ctx context.Context, c genai.Provider, lang string) (genai.RealtimeSessionInfo, string, bool, error) {
	opts := []genai.RealtimeOption{genai.RealtimeOptionTranscription{}}
	if lang != "" {
		opts = append(opts, &genai.RealtimeOptionAudio{InputLanguage: lang})
	}
	s, err := c.ConnectRealtime(ctx, opts...)
	if err != nil {
		return genai.RealtimeSessionInfo{}, "", false, err
	}
	final, drafts, err := transcribeSession(ctx, s)
	return s.Info(), final, drafts, errors.Join(err, s.Close())
}

func transcribeSession(ctx context.Context, s genai.RealtimeSession) (string, bool, error) {
	info := s.Info()
	if err := realtimeValidateInfo(&info); err != nil {
		return "", false, err
	}
	if info.OutputFormat != (genai.AudioFormat{}) || !slices.Equal(info.InputModalities, []genai.Modality{genai.ModalityAudio}) || info.Capabilities != (genai.RealtimeCapabilities{AudioCommit: true}) {
		return "", false, fmt.Errorf("transcription session info %+v", info)
	}
	audio, err := speech(ctx, speechFixtureCode, info.InputFormat.SampleRate)
	if err != nil {
		return "", false, err
	}
	report := RealtimeReport{}
	ev := newRealtimeEvents(ctx, s, &report)
	defer ev.release()
	paced := true
	if r, ok := s.(RealtimeReplay); ok && r.IsReplay() {
		paced = false
	}
	// Send while receiving so a server that decodes audio inline is not blocked
	// by unread events.
	sendCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	sent := make(chan error, 1)
	go func() { sent <- sendRealtimeAudio(sendCtx, s, audio, info.InputFormat.SampleRate, paced, true, false) }()
	final, drafts := "", false
	_, err = ev.await(RealtimeProbeTranscription, func(e *genai.RealtimeEvent) bool {
		switch e.Kind {
		case genai.RealtimeEventTranscriptInput, genai.RealtimeEventTranscriptInputInterim:
			if e.Content.Final {
				final = e.Content.Text
			} else if final == "" && e.Content.Text != "" {
				drafts = true
			}
		case genai.RealtimeEventTurnDone:
			return true
		default:
		}
		return false
	}, nil)
	if err != nil {
		// The sender may wait on a session that stopped reading.
		cancel()
	}
	return final, drafts, errors.Join(err, <-sent)
}
