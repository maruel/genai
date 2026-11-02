// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Core realtime session probes, gated on reported capabilities.

package smoke

import (
	"context"
	"errors"
	"fmt"
	"io"
	"iter"
	"slices"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/scoreboard"
)

// RealtimeSessionFactory opens the realtime session RunRealtime qualifies.
//
// The factory owns model selection, credentials, instructions and provider
// options. probe identifies the probe that will use the session. tools lists the
// session tools the probe needs; an empty list opens a session without tools.
// The runner opens the core session with no tools, then opens separate sessions
// with tools for the tool probes. The factory must select the output modality the
// probe needs: text for RealtimeProbeTools and audio for
// RealtimeProbeAudioTool.
type RealtimeSessionFactory func(ctx context.Context, probe RealtimeProbe, tools []genai.ToolDef) (genai.RealtimeSession, error)

// RealtimeProbe identifies one core realtime probe.
type RealtimeProbe string

// Core realtime probes, in execution order.
const (
	RealtimeProbeSetup        RealtimeProbe = "setup"
	RealtimeProbeText         RealtimeProbe = "text"
	RealtimeProbeReuse        RealtimeProbe = "reuse"
	RealtimeProbeCancellation RealtimeProbe = "cancellation"
	RealtimeProbeAudio        RealtimeProbe = "audio"
	RealtimeProbeAudioTool    RealtimeProbe = "audio_tool"
	RealtimeProbeTools        RealtimeProbe = "tools"
)

// RealtimeProbeResult is the capability-gated outcome of one core probe.
type RealtimeProbeResult struct {
	Probe RealtimeProbe
	// Skipped reports that the session's reported capabilities exclude the probe.
	Skipped bool
	// Failed reports that the probe ran and did not pass. The audio-tool probe is
	// optional and reports failure instead of failing the run.
	Failed bool
	// Reason explains a skipped or failed probe.
	Reason string
}

// RealtimeReport reports what RunRealtime observed.
type RealtimeReport struct {
	// Info describes the qualified session. It is not reported when the setup
	// probe failed.
	Info genai.RealtimeSessionInfo
	// Probes lists the core probes that passed, failed, or were skipped, in
	// execution order. Most failing probes are reported by the returned error
	// instead; the optional audio-tool probe reports failure here.
	Probes []RealtimeProbeResult
	// OutputAudio reports that a probe observed nonempty audio output.
	OutputAudio bool
	// OutputText reports that a probe observed nonempty text output. Output
	// transcripts are not text output.
	OutputText bool
	// Usage accumulates non-cumulative usage snapshots and keeps the latest rate
	// limits. It is partial when the run fails. Duplicate response IDs are counted
	// once.
	Usage genai.Usage
}

// RunRealtime qualifies a realtime provider with the core probes.
//
// The runner opens one session without tools and qualifies it: setup validates
// Info, text and reuse qualify two complete turns that produce output,
// cancellation qualifies that a running response is interrupted, and audio
// qualifies synthesized speech input. Separate sessions with tools qualify a
// text tool call and an audio tool call, including the model's answer after the
// tool results. The text probe requires a text answer; the audio probe requires
// a spoken answer. Audio probes use recorded speech and are skipped when the
// session does not accept audio turns. A provider that rejects tools or omits
// ToolResults skips both tool probes; a provider that rejects one tool output
// modality skips only that probe. The audio-tool probe is optional: a failed
// audio-tool probe is reported in Probes instead of failing the run. Probes the
// reported capabilities exclude are skipped instead of failing.
//
// Every received event is validated with RealtimeContent.Validate. Provider
// error events, invalid content, a session that ends before the awaited event, a
// turn that ends without the output or tool call the probe expects, mismatched
// tool correlation, output after an interruption, and a cancel that never
// interrupts fail the run. Event kinds the runner does not handle, including
// provider-specific ones, are ignored. Each wait consumes at most 1000 events;
// a probe can wait more than once. Every call uses ctx, so the caller bounds wall
// clock time with a deadline. A session Close error is joined into the returned
// error.
//
// The returned report is partial when the run fails.
func RunRealtime(ctx context.Context, f RealtimeSessionFactory) (report RealtimeReport, err error) {
	// The core probes run without tools. The tool probes use separate sessions so
	// their tool instructions and text output cannot affect text or audio turns.
	s, err := f(ctx, RealtimeProbeSetup, nil)
	if err != nil {
		return report, fmt.Errorf("probe %s: %w", RealtimeProbeSetup, err)
	}
	closed := false
	defer func() {
		if !closed {
			err = errors.Join(err, s.Close())
		}
	}()
	info := s.Info()
	if err := realtimeValidateInfo(&info); err != nil {
		return report, fmt.Errorf("probe %s: %w", RealtimeProbeSetup, err)
	}
	report.Info = info
	report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeSetup})
	ev := newRealtimeEvents(ctx, s, &report)
	defer ev.release()

	if err := realtimeTurn(ctx, ev, s, RealtimeProbeText, realtimeTextPrompt); err != nil {
		return report, err
	}
	report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeText})
	if err := realtimeTurn(ctx, ev, s, RealtimeProbeReuse, realtimeReusePrompt); err != nil {
		return report, err
	}
	report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeReuse})

	if !info.Capabilities.CancelResponse {
		report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeCancellation, Skipped: true, Reason: "session does not report CancelResponse"})
	} else {
		if err := realtimeCancel(ctx, ev, s); err != nil {
			return report, err
		}
		report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeCancellation})
	}

	if !realtimeAudioSupported(&info) {
		report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeAudio, Skipped: true, Reason: "session does not accept audio turns"})
	} else {
		if err := realtimeAudioTurn(ctx, ev, s, &info, speechFixtureOrange, RealtimeProbeAudio, false); err != nil {
			return report, err
		}
		report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeAudio})
	}

	ts, _, skip, err := openRealtimeToolSession(ctx, f, RealtimeProbeTools)
	if err != nil {
		return report, err
	}
	if skip != "" {
		report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeTools, Skipped: true, Reason: skip})
	} else {
		tev := newRealtimeEvents(ctx, ts, &report)
		err := realtimeTools(ctx, tev, ts)
		tev.release()
		err = errors.Join(err, ts.Close())
		if err != nil {
			return report, err
		}
		report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeTools})
	}

	{
		as, ainfo, skip, err := openRealtimeToolSession(ctx, f, RealtimeProbeAudioTool)
		if err != nil {
			return report, err
		}
		switch {
		case skip != "":
			report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeAudioTool, Skipped: true, Reason: skip})
		case !realtimeAudioSupported(&ainfo):
			if err := as.Close(); err != nil {
				return report, err
			}
			report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeAudioTool, Skipped: true, Reason: "audio-tool session does not accept audio turns"})
		default:
			aev := newRealtimeEvents(ctx, as, &report)
			turnErr := realtimeAudioTurn(ctx, aev, as, &ainfo, speechFixtureCode, RealtimeProbeAudioTool, true)
			aev.release()
			closeErr := as.Close()
			if closeErr != nil {
				return report, errors.Join(turnErr, closeErr)
			}
			if turnErr != nil {
				report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeAudioTool, Failed: true, Reason: turnErr.Error()})
			} else {
				report.Probes = append(report.Probes, RealtimeProbeResult{Probe: RealtimeProbeAudioTool})
			}
		}
	}
	return report, nil
}

// RunRealtimeScenario qualifies bidirectional realtime support for the
// provider's configured model.
//
// It requires the provider to implement ConnectRealtime. reason selects the
// reasoning mode the returned scenario declares. The text tool probe requests
// text output; the audio tool probe requests audio output. The returned scenario
// declares text input and audio or text output and records the measured formats
// and capabilities the probes exercised. It returns the accumulated
// non-cumulative usage; usage is partial when the run fails. Realtime sessions
// are WebSocket conversations, not HTTP requests; provider tests record them
// separately.
// A previously recorded tool result must keep passing. A model that never
// recorded one records a failed audio probe instead of failing the run. A Flaky
// result stays Flaky across later passes and failures.
func RunRealtimeScenario(ctx context.Context, c genai.Provider, reason bool) (scoreboard.Scenario, genai.Usage, error) {
	var usage genai.Usage
	model := c.ModelID()
	sc := scoreboard.Scenario{Models: []string{model}, Reason: reason}
	sc.In = map[genai.Modality]scoreboard.ModalCapability{
		genai.ModalityText: {Inline: true},
	}
	if model == "" {
		return sc, usage, errors.New("provider must have a model")
	}
	if !c.Capabilities().Realtime {
		return sc, usage, errors.New("provider does not support realtime")
	}
	ctx, cancel := context.WithTimeout(ctx, 2*time.Minute)
	defer cancel()
	report, err := RunRealtime(ctx, func(ctx context.Context, probe RealtimeProbe, tools []genai.ToolDef) (genai.RealtimeSession, error) {
		instructions := "Follow the user's request exactly."
		if len(tools) != 0 {
			// LFM3-Audio and similar models need their documented text-only
			// function-calling system prompt to emit tool calls reliably.
			instructions = "List of tools:"
		}
		connOpts := []genai.RealtimeOption{genai.RealtimeOptionInstructions(instructions)}
		switch probe {
		case RealtimeProbeTools:
			connOpts = append(connOpts, genai.Modalities{genai.ModalityText})
		case RealtimeProbeAudioTool:
			connOpts = append(connOpts, genai.Modalities{genai.ModalityAudio})
		case RealtimeProbeSetup, RealtimeProbeText, RealtimeProbeReuse, RealtimeProbeCancellation, RealtimeProbeAudio:
		}
		if len(tools) != 0 {
			connOpts = append(connOpts, &genai.RealtimeOptionTools{Tools: tools})
		}
		return c.ConnectRealtime(ctx, connOpts...)
	})
	usage = report.Usage
	if err != nil {
		return sc, usage, err
	}
	previousTools := realtimePreviousTools(c, model)
	tools := scoreboard.False
	if realtimeProbePassed(&report, RealtimeProbeTools) || realtimeProbePassed(&report, RealtimeProbeAudioTool) {
		tools = scoreboard.True
	}
	switch {
	case previousTools == scoreboard.Flaky:
		tools = scoreboard.Flaky
	case previousTools == scoreboard.True && tools == scoreboard.False:
		for i := range report.Probes {
			if report.Probes[i].Failed {
				return sc, usage, fmt.Errorf("tool result regressed: %s", report.Probes[i].Reason)
			}
		}
		return sc, usage, errors.New("tool result regressed")
	}
	sc.Realtime = &scoreboard.RealtimeFunctionality{
		InputSampleRate:  report.Info.InputFormat.SampleRate,
		OutputSampleRate: report.Info.OutputFormat.SampleRate,
		CancelResponse:   realtimeProbePassed(&report, RealtimeProbeCancellation),
		Tools:            tools,
		ReportRateLimits: len(report.Usage.Limits) != 0,
		ReportTokenUsage: scoreboard.False,
	}
	if !isZeroUsage(&report.Usage) {
		sc.Realtime.ReportTokenUsage = scoreboard.True
	}
	if realtimeProbePassed(&report, RealtimeProbeAudio) {
		sc.In[genai.ModalityAudio] = scoreboard.ModalCapability{Inline: true}
	}
	sc.Out = map[genai.Modality]scoreboard.ModalCapability{}
	if report.OutputAudio {
		sc.Out[genai.ModalityAudio] = scoreboard.ModalCapability{Inline: true}
	}
	if report.OutputText {
		sc.Out[genai.ModalityText] = scoreboard.ModalCapability{Inline: true}
	}
	return sc, usage, nil
}

// realtimeProbePassed reports whether the named probe ran and passed.
func realtimeProbePassed(report *RealtimeReport, probe RealtimeProbe) bool {
	for i := range report.Probes {
		if report.Probes[i].Probe == probe {
			return !report.Probes[i].Skipped && !report.Probes[i].Failed
		}
	}
	return false
}

// realtimePreviousTools returns the tool result a scoreboard already records for
// model.
func realtimePreviousTools(c genai.Provider, model string) scoreboard.TriState {
	score := c.Scoreboard()
	for i := range score.Scenarios {
		s := &score.Scenarios[i]
		if s.Realtime != nil && slices.Contains(s.Models, model) {
			return s.Realtime.Tools
		}
	}
	return scoreboard.False
}

// realtimeEventBudget bounds the events one wait consumes so a stalled or chatty
// session fails instead of running until the caller's deadline.
const realtimeEventBudget = 1000

const (
	realtimeTextPrompt   = "Say hello. Use only one word."
	realtimeReusePrompt  = "Say goodbye. Use only one word."
	realtimeToolPrompt   = "What is the access code? Call lookup_code, then use its result. Reply with only the code."
	realtimeToolName     = "lookup_code"
	realtimeToolResult   = `{"code":"violet seven"}`
	realtimeToolCode     = "violet seven"
	realtimeCancelPrompt = "Count slowly from one to one hundred. Say each number as a separate word."
)

// realtimeToolDefs returns the session tools the tool probe expects.
func realtimeToolDefs() []genai.ToolDef {
	return []genai.ToolDef{{
		Name:                realtimeToolName,
		Description:         "Look up the current access code",
		InputSchemaOverride: genai.JSONSchema(`{"type":"object","properties":{}}`),
	}}
}

// realtimeValidateInfo checks the formats and the turn mode the session accepted.
func realtimeValidateInfo(info *genai.RealtimeSessionInfo) error {
	if err := info.InputFormat.Validate(); err != nil {
		return fmt.Errorf("input format: %w", err)
	}
	if err := info.OutputFormat.Validate(); err != nil {
		return fmt.Errorf("output format: %w", err)
	}
	switch info.TurnDetection {
	case genai.RealtimeTurnManual, genai.RealtimeTurnServer:
		return nil
	}
	return fmt.Errorf("turn detection: must be %q or %q, got %q", genai.RealtimeTurnManual, genai.RealtimeTurnServer, info.TurnDetection)
}

// realtimeTurn sends one text turn and requires output before the turn completes.
func realtimeTurn(ctx context.Context, ev *realtimeEvents, s genai.RealtimeSession, probe RealtimeProbe, text string) error {
	if err := s.Send(ctx, genai.RealtimeInputContent{Text: text, Generate: true}); err != nil {
		return fmt.Errorf("probe %s: %w", probe, err)
	}
	if _, err := ev.await(probe, realtimeGenerating, realtimeAbort("turn completed without audio or text output", genai.RealtimeEventGenerationDone, genai.RealtimeEventTurnDone)); err != nil {
		return err
	}
	_, err := ev.await(probe, realtimeTurnDone, nil)
	return err
}

// realtimeAudioSupported reports whether the session accepts an audio turn.
func realtimeAudioSupported(info *genai.RealtimeSessionInfo) bool {
	return info.TurnDetection == genai.RealtimeTurnServer || info.Capabilities.AudioCommit
}

// sendRealtimeAudio sends audio in 100 ms chunks, as live APIs expect. Server
// VAD also needs the chunks paced at playback speed to detect silence. The last
// chunk commits the turn and optionally requests a reply.
func sendRealtimeAudio(ctx context.Context, s genai.RealtimeSession, audio []byte, rate int, paced, generate bool) error {
	chunk := max(2, rate/10*2)
	for len(audio) != 0 {
		n := min(chunk, len(audio))
		if err := s.Send(ctx, genai.RealtimeInputContent{Audio: audio[:n], Commit: len(audio) == n, Generate: len(audio) == n && generate}); err != nil {
			return err
		}
		audio = audio[n:]
		if paced && len(audio) != 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(100 * time.Millisecond):
			}
		}
	}
	return nil
}

// realtimeAudioTurn sends one synthesized audio turn.
func realtimeAudioTurn(ctx context.Context, ev *realtimeEvents, s genai.RealtimeSession, info *genai.RealtimeSessionInfo, fixture speechFixture, probe RealtimeProbe, expectTool bool) error {
	rate := info.InputFormat.SampleRate
	audio, err := speech(ctx, fixture, rate)
	if err != nil {
		return fmt.Errorf("probe %s: synthesize audio: %w", probe, err)
	}
	if len(audio) == 0 || len(audio)%2 != 0 {
		return fmt.Errorf("probe %s: synthesized audio must contain nonempty PCM16 samples", probe)
	}
	manual := info.TurnDetection == genai.RealtimeTurnManual
	if manual && !info.Capabilities.AudioCommit {
		return fmt.Errorf("probe %s: session does not accept manual audio turns", probe)
	}
	if info.TurnDetection == genai.RealtimeTurnServer {
		// Give server VAD 500 ms of silence to close the utterance. Padding in
		// whole samples keeps PCM16 valid for odd sample rates.
		audio = append(audio, make([]byte, (rate/2)*2)...)
	}
	if err := sendRealtimeAudio(ctx, s, audio, rate, info.TurnDetection == genai.RealtimeTurnServer, manual); err != nil {
		return fmt.Errorf("probe %s: %w", probe, err)
	}
	if expectTool {
		// A non-blocking provider may finish filler utterances before the tool call.
		calls, err := ev.await(probe, realtimeToolCalls, realtimeAbort("turn completed without a tool call", genai.RealtimeEventTurnDone))
		if err != nil {
			return err
		}
		return realtimeToolResults(ctx, ev, s, &calls, probe, true)
	}
	if _, err := ev.await(probe, realtimeGenerating, realtimeAbort("turn completed without audio or text output", genai.RealtimeEventGenerationDone, genai.RealtimeEventTurnDone)); err != nil {
		return err
	}
	_, err = ev.await(probe, realtimeTurnDone, nil)
	return err
}

// openRealtimeToolSession opens a session with tools for one tool probe. A
// provider setup error or missing ToolResults capability returns a skip reason
// instead of an error, so providers without tool support remain qualifiable.
func openRealtimeToolSession(ctx context.Context, f RealtimeSessionFactory, probe RealtimeProbe) (genai.RealtimeSession, genai.RealtimeSessionInfo, string, error) {
	s, err := f(ctx, probe, realtimeToolDefs())
	if err != nil {
		return nil, genai.RealtimeSessionInfo{}, err.Error(), nil //nolint:nilerr // A rejected tool setup is a skipped capability.
	}
	info := s.Info()
	if err := realtimeValidateInfo(&info); err != nil {
		return nil, info, "", errors.Join(fmt.Errorf("probe %s: %w", probe, err), s.Close())
	}
	if !info.Capabilities.ToolResults {
		return nil, info, "session does not report ToolResults", s.Close()
	}
	return s, info, "", nil
}

// realtimeTools qualifies a tool call and the tool results sent back for it.
func realtimeTools(ctx context.Context, ev *realtimeEvents, s genai.RealtimeSession) error {
	if err := s.Send(ctx, genai.RealtimeInputContent{Text: realtimeToolPrompt, Generate: true}); err != nil {
		return fmt.Errorf("probe %s: %w", RealtimeProbeTools, err)
	}
	// A non-blocking provider may finish filler utterances before it emits the
	// tool call. GenerationDone describes that filler, not a completed request.
	calls, err := ev.await(RealtimeProbeTools, realtimeToolCalls, realtimeAbort("turn completed without a tool call", genai.RealtimeEventTurnDone))
	if err != nil {
		return err
	}
	return realtimeToolResults(ctx, ev, s, &calls, RealtimeProbeTools, false)
}

// realtimeToolResults validates and sends one correlated result batch, then
// requires the model's answer to the result. wantAudio selects a spoken answer
// instead of a text answer.
func realtimeToolResults(ctx context.Context, ev *realtimeEvents, s genai.RealtimeSession, calls *genai.RealtimeEvent, probe RealtimeProbe, wantAudio bool) error {
	if len(calls.Content.ToolCalls) == 0 {
		return fmt.Errorf("probe %s: tool_calls event carried no call", probe)
	}
	results := make([]genai.ToolCallResult, len(calls.Content.ToolCalls))
	seen := make(map[string]struct{}, len(results))
	for i := range calls.Content.ToolCalls {
		call := &calls.Content.ToolCalls[i]
		if call.Name != realtimeToolName {
			return fmt.Errorf("probe %s: mismatched tool correlation: got tool %q, want %q", probe, call.Name, realtimeToolName)
		}
		if call.ID == "" {
			return fmt.Errorf("probe %s: mismatched tool correlation: call %q has no ID", probe, call.Name)
		}
		if _, ok := seen[call.ID]; ok {
			return fmt.Errorf("probe %s: mismatched tool correlation: duplicate call ID %q", probe, call.ID)
		}
		seen[call.ID] = struct{}{}
		if err := call.Validate(); err != nil {
			return fmt.Errorf("probe %s: tool call %q: %w", probe, call.ID, err)
		}
		results[i] = genai.ToolCallResult{ID: call.ID, Name: call.Name, Result: realtimeToolResult}
	}
	if err := s.Send(ctx, genai.RealtimeInputContent{ToolResults: results, Generate: true}); err != nil {
		return fmt.Errorf("probe %s: %w", probe, err)
	}
	return realtimeToolAnswer(probe, ev, calls, wantAudio)
}

// realtimeToolAnswer consumes the turn that answers a tool result and requires
// the access code in its output. The tool-call response may leave a queued
// TurnDone behind; output and completion that carry the tool-call response ID
// are not part of the answer.
func realtimeToolAnswer(probe RealtimeProbe, ev *realtimeEvents, calls *genai.RealtimeEvent, wantAudio bool) error {
	var text, transcript strings.Builder
	sawAudio := false
	stale := func(e *genai.RealtimeEvent) bool {
		return calls.ResponseID != "" && e.ResponseID == calls.ResponseID
	}
	collect := func(e *genai.RealtimeEvent) {
		if stale(e) {
			return
		}
		switch e.Kind {
		case genai.RealtimeEventContentAudio:
			sawAudio = sawAudio || len(e.Content.Audio) != 0
		case genai.RealtimeEventTranscriptOutput:
			if e.Content.Text != "" {
				if e.Content.Final {
					transcript.Reset()
				}
				transcript.WriteString(e.Content.Text)
			}
		case genai.RealtimeEventContentText:
			if e.Content.Text != "" {
				if e.Content.Final {
					text.Reset()
				}
				text.WriteString(e.Content.Text)
			}
		case genai.RealtimeEventError, genai.RealtimeEventGenerationDone, genai.RealtimeEventInputSpeechEnd, genai.RealtimeEventInputSpeechStart, genai.RealtimeEventTranscriptInput, genai.RealtimeEventInterrupted, genai.RealtimeEventProvider, genai.RealtimeEventContentToolCalls, genai.RealtimeEventTurnDone, genai.RealtimeEventUsage:
		}
	}
	// The first await skips a queued tool-call TurnDone and stale output from the
	// tool-call response.
	started := false
	_, err := ev.await(probe, func(e *genai.RealtimeEvent) bool {
		collect(e)
		if stale(e) {
			return false
		}
		if wantAudio {
			started = len(e.Content.Audio) != 0 || (e.Kind == genai.RealtimeEventTranscriptOutput && e.Content.Text != "")
		} else {
			started = e.Kind == genai.RealtimeEventContentText && e.Content.Text != ""
		}
		return started || e.Kind == genai.RealtimeEventTurnDone
	}, realtimeAbort("tool result answer was interrupted", genai.RealtimeEventInterrupted))
	if err != nil {
		return err
	}
	if !started {
		if wantAudio {
			return fmt.Errorf("probe %s: tool result turn completed without spoken output", probe)
		}
		return fmt.Errorf("probe %s: tool result turn completed without text output", probe)
	}
	_, err = ev.await(probe, func(e *genai.RealtimeEvent) bool {
		collect(e)
		return e.Kind == genai.RealtimeEventTurnDone && !stale(e)
	}, realtimeAbort("tool result answer was interrupted", genai.RealtimeEventInterrupted))
	if err != nil {
		return err
	}
	if wantAudio {
		if !sawAudio {
			return fmt.Errorf("probe %s: tool result answer had no spoken output", probe)
		}
		if transcript.Len() != 0 && !realtimeHasToolAnswer(transcript.String()) {
			return fmt.Errorf("probe %s: tool result transcript %q does not contain %q", probe, transcript.String(), realtimeToolCode)
		}
		return nil
	}
	if !realtimeHasToolAnswer(text.String()) {
		return fmt.Errorf("probe %s: tool result text %q does not contain %q", probe, text.String(), realtimeToolCode)
	}
	return nil
}

// realtimeHasToolAnswer reports whether s contains the expected tool answer,
// ignoring case and punctuation.
func realtimeHasToolAnswer(s string) bool {
	normalize := func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			return r
		case r >= 'A' && r <= 'Z':
			return r + ('a' - 'A')
		default:
			return ' '
		}
	}
	return strings.Contains(strings.Join(strings.Fields(strings.Map(normalize, s)), " "), realtimeToolCode)
}

// realtimeCancel qualifies that a cancel interrupts a generating response.
func realtimeCancel(ctx context.Context, ev *realtimeEvents, s genai.RealtimeSession) error {
	if err := s.Send(ctx, genai.RealtimeInputContent{Text: realtimeCancelPrompt, Generate: true}); err != nil {
		return fmt.Errorf("probe %s: %w", RealtimeProbeCancellation, err)
	}
	if _, err := ev.await(RealtimeProbeCancellation, realtimeGenerating, realtimeAbort("turn completed without audio or text output", genai.RealtimeEventGenerationDone, genai.RealtimeEventTurnDone)); err != nil {
		return err
	}
	if err := s.Send(ctx, genai.RealtimeCancel{}); err != nil {
		return fmt.Errorf("probe %s: %w", RealtimeProbeCancellation, err)
	}
	if _, err := ev.await(RealtimeProbeCancellation, realtimeInterrupted, realtimeAbort("turn completed without interruption after cancel", genai.RealtimeEventTurnDone)); err != nil {
		return err
	}
	// The turn must drain to its terminal event without further output.
	_, err := ev.await(RealtimeProbeCancellation, realtimeTurnDone, realtimeAbort("output after interruption", genai.RealtimeEventContentAudio, genai.RealtimeEventTranscriptOutput, genai.RealtimeEventContentText))
	return err
}

// realtimeEvents pulls validated events from one session.
type realtimeEvents struct {
	next     func() (genai.RealtimeEvent, error, bool)
	release  func()
	report   *RealtimeReport
	usageIDs map[string]struct{}
}

// newRealtimeEvents starts consuming the session's events.
func newRealtimeEvents(ctx context.Context, s genai.RealtimeSession, report *RealtimeReport) *realtimeEvents {
	next, release := iter.Pull2(s.Receive(ctx))
	return &realtimeEvents{next: next, release: release, report: report, usageIDs: make(map[string]struct{})}
}

// await returns the first event matching match. abort rejects an event that ends
// the wait early; it runs before match, so it wins over a matching event.
func (e *realtimeEvents) await(probe RealtimeProbe, match func(*genai.RealtimeEvent) bool, abort func(*genai.RealtimeEvent) error) (genai.RealtimeEvent, error) {
	var zero genai.RealtimeEvent
	for range realtimeEventBudget {
		ev, err, ok := e.next()
		if !ok || errors.Is(err, io.EOF) {
			return zero, fmt.Errorf("probe %s: session ended before the awaited event", probe)
		}
		if err != nil {
			return zero, fmt.Errorf("probe %s: receive: %w", probe, err)
		}
		if err := ev.Content.Validate(); err != nil {
			return zero, fmt.Errorf("probe %s: invalid %s content: %w", probe, ev.Kind, err)
		}
		switch ev.Kind {
		case genai.RealtimeEventContentAudio:
			e.report.OutputAudio = e.report.OutputAudio || len(ev.Content.Audio) != 0
		case genai.RealtimeEventContentText:
			e.report.OutputText = e.report.OutputText || ev.Content.Text != ""
		case genai.RealtimeEventUsage:
			e.addUsage(&ev)
		default:
		}
		if ev.Kind == genai.RealtimeEventError {
			return zero, fmt.Errorf("probe %s: provider error: %w", probe, realtimeEventError(ev.Error))
		}
		if abort != nil {
			if err := abort(&ev); err != nil {
				return zero, fmt.Errorf("probe %s: %w", probe, err)
			}
		}
		if match(&ev) {
			return ev, nil
		}
	}
	return zero, fmt.Errorf("probe %s: %d events without a matching event", probe, realtimeEventBudget)
}

// addUsage accumulates one non-cumulative snapshot. A duplicate response ID
// does not affect the total. Rate limits replace the previous snapshot.
func (e *realtimeEvents) addUsage(ev *genai.RealtimeEvent) {
	if ev.Usage == nil {
		return
	}
	if ev.ResponseID != "" {
		if _, ok := e.usageIDs[ev.ResponseID]; ok {
			return
		}
		e.usageIDs[ev.ResponseID] = struct{}{}
	}
	e.report.Usage.Add(ev.Usage)
	if e.report.Usage.ServiceTier == "" {
		e.report.Usage.ServiceTier = ev.Usage.ServiceTier
	}
	if len(ev.Usage.Limits) != 0 {
		e.report.Usage.Limits = slices.Clone(ev.Usage.Limits)
	}
}

// realtimeAbort returns an abort predicate that rejects the event kinds that end
// a wait early with reason.
func realtimeAbort(reason string, kinds ...genai.RealtimeEventKind) func(*genai.RealtimeEvent) error {
	return func(e *genai.RealtimeEvent) error {
		if slices.Contains(kinds, e.Kind) {
			return errors.New(reason)
		}
		return nil
	}
}

// realtimeEventError returns a non-nil error for an Error event.
func realtimeEventError(e *genai.RealtimeError) error {
	if e == nil {
		return errors.New("event carried no detail")
	}
	return e
}

// realtimeTurnDone reports a completed provider turn.
func realtimeTurnDone(e *genai.RealtimeEvent) bool {
	return e.Kind == genai.RealtimeEventTurnDone
}

// realtimeToolCalls reports complete tool calls.
func realtimeToolCalls(e *genai.RealtimeEvent) bool {
	return e.Kind == genai.RealtimeEventContentToolCalls
}

// realtimeInterrupted reports that the provider stopped generating.
func realtimeInterrupted(e *genai.RealtimeEvent) bool {
	return e.Kind == genai.RealtimeEventInterrupted
}

// realtimeGenerating reports that the response produced non-empty output.
func realtimeGenerating(e *genai.RealtimeEvent) bool {
	switch e.Kind {
	case genai.RealtimeEventContentAudio:
		return len(e.Content.Audio) != 0
	case genai.RealtimeEventTranscriptOutput, genai.RealtimeEventContentText:
		return e.Content.Text != ""
	default:
		return false
	}
}
