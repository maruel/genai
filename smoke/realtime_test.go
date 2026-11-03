// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the core realtime smoke probes.

package smoke_test

import (
	"context"
	"errors"
	"net/http"
	"slices"
	"strings"
	"testing"
	"testing/synctest"
	"time"

	"github.com/google/go-cmp/cmp"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/genai/smoke"
)

// fakeRealtimeInfo is the info of a session that reports every capability.
var fakeRealtimeInfo = genai.RealtimeSessionInfo{
	ID:              "fake",
	InputFormat:     genai.AudioFormat{SampleRate: 16000},
	OutputFormat:    genai.AudioFormat{SampleRate: 24000},
	InputModalities: []genai.Modality{genai.ModalityAudio, genai.ModalityImage, genai.ModalityText},
	TurnDetection:   genai.RealtimeTurnManual,
	Capabilities:    genai.RealtimeCapabilities{CancelResponse: true, ToolResults: true},
}

// fakeImageTurn is the answer to the image prompt.
var fakeImageTurn = []genai.RealtimeEvent{
	{Kind: genai.RealtimeEventContentText, Content: genai.RealtimeContent{Text: "banana"}},
	{Kind: genai.RealtimeEventGenerationDone},
	{Kind: genai.RealtimeEventTurnDone},
}

// fakeOutputTurn is a complete turn with text and audio output.
var fakeOutputTurn = []genai.RealtimeEvent{
	{Kind: genai.RealtimeEventInputSpeechStart},
	{Kind: genai.RealtimeEventProvider, ProviderEvent: fakeProviderEvent{}},
	{Kind: genai.RealtimeEventContentText, Content: genai.RealtimeContent{Text: "hello"}},
	{Kind: genai.RealtimeEventContentAudio, Content: genai.RealtimeContent{Audio: []byte{1, 0}}},
	{Kind: genai.RealtimeEventGenerationDone},
	{Kind: genai.RealtimeEventTurnDone},
}

// fakeAudioTurn is a turn that produced audio only.
var fakeAudioTurn = []genai.RealtimeEvent{
	{Kind: genai.RealtimeEventContentAudio, Content: genai.RealtimeContent{Audio: []byte{1, 0}}},
}

// fakeToolAnswer is the model's answer after a tool result. It carries both text
// and spoken output; the probe selects the form it expects.
var fakeToolAnswer = []genai.RealtimeEvent{
	{Kind: genai.RealtimeEventContentText, ResponseID: "answer", Content: genai.RealtimeContent{Text: "violet seven"}},
	{Kind: genai.RealtimeEventTranscriptOutput, ResponseID: "answer", Content: genai.RealtimeContent{Text: "violet seven"}},
	{Kind: genai.RealtimeEventContentAudio, ResponseID: "answer", Content: genai.RealtimeContent{Audio: []byte{1, 0}}},
	{Kind: genai.RealtimeEventTurnDone, ResponseID: "answer"},
}

// fakeInterruption is the reply to a cancel that stops the response.
var fakeInterruption = []genai.RealtimeEvent{
	{Kind: genai.RealtimeEventInterrupted},
	{Kind: genai.RealtimeEventTurnDone},
}

// fakeToolCalls is a turn that calls tool once and ends output generation, the
// way providers end a turn that requests tools.
func fakeToolCalls(tool string) []genai.RealtimeEvent {
	return []genai.RealtimeEvent{
		{Kind: genai.RealtimeEventProvider, ProviderEvent: fakeProviderEvent{}},
		{Kind: genai.RealtimeEventContentToolCalls, ResponseID: "call_response", Content: genai.RealtimeContent{ToolCalls: []genai.ToolCall{{ID: "call_1", Name: tool, Arguments: "{}"}}}},
		{Kind: genai.RealtimeEventGenerationDone, ResponseID: "call_response"},
	}
}

// fakeProviderEvent is a provider-specific event the runner must ignore.
type fakeProviderEvent struct{}

// RealtimeEventName implements genai.RealtimeProviderEvent.
func (fakeProviderEvent) RealtimeEventName() string { return "fake.event" }

var _ genai.RealtimeProviderEvent = fakeProviderEvent{}

// fakeRealtimeFactory scripts and records the sessions RunRealtime opens.
//
// Each field scripts the reply to one command. A session opened without tools
// skips the tool prompt. Text prompts are identified by content, so a change in
// probe order cannot desync the script.
type fakeRealtimeFactory struct {
	t *testing.T
	// text and reuse script the two text probes.
	text, reuse []genai.RealtimeEvent
	// toolPrompt scripts the tool probe's prompt with the requested tool name.
	toolPrompt func(tool string) []genai.RealtimeEvent
	// toolResults scripts the text tool probe's reply to the tool results.
	toolResults []genai.RealtimeEvent
	// audioToolResults scripts the audio tool probe's reply to the tool results.
	audioToolResults []genai.RealtimeEvent
	// cancelPrompt scripts the cancellation probe's prompt.
	cancelPrompt []genai.RealtimeEvent
	// cancel scripts the reply to the cancel command.
	cancel []genai.RealtimeEvent
	// audio scripts the reply to an audio-only turn.
	audio []genai.RealtimeEvent
	// audioTool scripts the reply to an audio turn that calls a tool.
	audioTool func(tool string) []genai.RealtimeEvent
	// image scripts the reply to the image turn.
	image []genai.RealtimeEvent
	// info is the info of the session. secondInfo overrides the audio-tool
	// session's info when set.
	info       genai.RealtimeSessionInfo
	secondInfo *genai.RealtimeSessionInfo
	// replay makes the sessions report that they replay a recording.
	replay bool
	// closeErr fails Close.
	closeErr error
	// err fails every factory call.
	err error
	// toolErr fails a factory call that requests tools.
	toolErr error
	// opens records the tools requested by each factory call, in order.
	opens [][]genai.ToolDef
	// probes records the probe requested by each factory call, in order.
	probes []smoke.RealtimeProbe
	// sessions lists the sessions opened, in order.
	sessions []*realtimetest.FakeSession
}

func newFakeRealtimeFactory(t *testing.T) *fakeRealtimeFactory {
	return &fakeRealtimeFactory{
		t:                t,
		text:             fakeOutputTurn,
		reuse:            fakeOutputTurn,
		toolPrompt:       fakeToolCalls,
		toolResults:      fakeToolAnswer,
		audioToolResults: fakeToolAnswer,
		cancelPrompt:     fakeAudioTurn,
		cancel:           fakeInterruption,
		audio:            fakeOutputTurn,
		image:            fakeImageTurn,
		info:             fakeRealtimeInfo,
	}
}

func (f *fakeRealtimeFactory) open(ctx context.Context, probe smoke.RealtimeProbe, tools []genai.ToolDef) (genai.RealtimeSession, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.opens = append(f.opens, tools)
	f.probes = append(f.probes, probe)
	if f.err != nil {
		return nil, f.err
	}
	tool := ""
	if len(tools) != 0 {
		if f.toolErr != nil {
			return nil, f.toolErr
		}
		tool = tools[0].Name
	}
	audioToolSession := probe == smoke.RealtimeProbeAudioTool
	audioReplied := false
	info := f.info
	if audioToolSession && f.secondInfo != nil {
		info = *f.secondInfo
	}
	s := realtimetest.NewFakeSession(info)
	s.CloseErr = f.closeErr
	s.Replay = f.replay
	s.Reply = func(_ int, in genai.RealtimeInput) []genai.RealtimeEvent {
		switch v := in.(type) {
		case genai.RealtimeInputContent:
			switch {
			case v.ToolResults != nil:
				if audioToolSession {
					return f.audioToolResults
				}
				return f.toolResults
			case v.Audio != nil || v.Commit:
				if audioReplied {
					return nil
				}
				audioReplied = true
				if audioToolSession && f.audioTool != nil {
					return f.audioTool(tool)
				}
				return f.audio
			case !v.Doc.IsZero():
				return f.image
			}
			switch {
			case strings.Contains(v.Text, "Say hello"):
				return f.text
			case strings.Contains(v.Text, "Say goodbye"):
				return f.reuse
			case strings.Contains(v.Text, "access code"):
				return f.toolPrompt(tool)
			case strings.Contains(v.Text, "Count slowly"):
				return f.cancelPrompt
			}
			f.t.Fatalf("unexpected text prompt %q", v.Text)
			return nil
		case genai.RealtimeCancel:
			return f.cancel
		default:
			return nil
		}
	}
	f.sessions = append(f.sessions, s)
	return s, nil
}

// audioChunks returns every audio input in sent.
func audioChunks(sent []genai.RealtimeInput) [][]byte {
	var out [][]byte
	for i := range sent {
		if in, ok := sent[i].(genai.RealtimeInputContent); ok && in.Audio != nil {
			out = append(out, in.Audio)
		}
	}
	return out
}

// ranProbes returns the name of every probe reported by r, in order.
func ranProbes(r *smoke.RealtimeReport) []smoke.RealtimeProbe {
	got := make([]smoke.RealtimeProbe, len(r.Probes))
	for i := range r.Probes {
		got[i] = r.Probes[i].Probe
	}
	return got
}

// probeResult returns the result of probe p.
func probeResult(t *testing.T, r *smoke.RealtimeReport, p smoke.RealtimeProbe) smoke.RealtimeProbeResult {
	i := slices.IndexFunc(r.Probes, func(x smoke.RealtimeProbeResult) bool { return x.Probe == p })
	if i == -1 {
		t.Fatalf("probe %s missing from %v", p, ranProbes(r))
	}
	return r.Probes[i]
}

func TestRunRealtime(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		t.Run("Replay", func(t *testing.T) {
			// A replayed recording does not depend on when audio arrives, so its audio
			// turns skip the playback-speed pacing a live server needs.
			elapsed := func(replay bool) time.Duration {
				var d time.Duration
				synctest.Test(t, func(t *testing.T) {
					f := newFakeRealtimeFactory(t)
					f.info.TurnDetection = genai.RealtimeTurnServer
					f.replay = replay
					start := time.Now()
					if _, err := smoke.RunRealtime(t.Context(), f.open); err != nil {
						t.Fatal(err)
					}
					d = time.Since(start)
				})
				return d
			}
			if got := elapsed(true); got != 0 {
				t.Fatalf("replayed recording paced audio for %v", got)
			}
			if got := elapsed(false); got == 0 {
				t.Fatal("live session did not pace audio")
			}
		})
		t.Run("success", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if !cmp.Equal(report.Info, fakeRealtimeInfo) {
				t.Fatalf("info=%+v", report.Info)
			}
			want := []smoke.RealtimeProbe{smoke.RealtimeProbeSetup, smoke.RealtimeProbeText, smoke.RealtimeProbeReuse, smoke.RealtimeProbeCancellation, smoke.RealtimeProbeAudio, smoke.RealtimeProbeImage, smoke.RealtimeProbeTools, smoke.RealtimeProbeAudioTool}
			if got := ranProbes(&report); !slices.Equal(got, want) {
				t.Fatalf("probes=%v, want %v", got, want)
			}
			for _, p := range report.Probes[:4] {
				if p.Skipped {
					t.Fatalf("probe %s skipped: %s", p.Probe, p.Reason)
				}
			}
			for _, p := range []smoke.RealtimeProbeResult{report.Probes[4], report.Probes[7]} {
				if !p.Skipped {
					t.Fatalf("probe %s did not skip", p.Probe)
				}
			}
			for _, p := range report.Probes[5:7] {
				if p.Skipped || p.Failed {
					t.Fatalf("probe %s did not pass: %s", p.Probe, p.Reason)
				}
			}
			// The core and image probes open without tools; each tool probe opens a
			// session with tools.
			if len(f.opens) != 4 || len(f.opens[0]) != 0 || len(f.opens[1]) != 0 || len(f.opens[2]) != 1 || len(f.opens[3]) != 1 {
				t.Fatalf("opens=%v", f.opens)
			}
			if tool := f.opens[2][0]; tool.Name == "" || tool.Validate() != nil {
				t.Fatalf("tool=%+v", tool)
			}
			if len(f.sessions) != 4 {
				t.Fatalf("sessions=%d", len(f.sessions))
			}
			sent := f.sessions[0].Sent
			if len(sent) != 4 {
				t.Fatalf("sent=%v", sent)
			}
			for _, in := range sent[:2] {
				if v, ok := in.(genai.RealtimeInputContent); !ok || !v.Generate {
					t.Fatalf("input=%v", in)
				}
			}
			if v, ok := sent[2].(genai.RealtimeInputContent); !ok || !v.Generate {
				t.Fatalf("cancel prompt=%v", sent[2])
			}
			if v, ok := sent[3].(genai.RealtimeCancel); !ok || v.ResponseID != "" {
				t.Fatalf("cancel=%v", sent[3])
			}
			toolSent := f.sessions[2].Sent
			text, ok := toolSent[0].(genai.RealtimeInputContent)
			if len(toolSent) != 2 || !ok || !text.Generate {
				t.Fatalf("tool sent=%v", toolSent)
			}
			results, ok := toolSent[1].(genai.RealtimeInputContent)
			if !ok || !results.Generate {
				t.Fatalf("tool sent=%v", toolSent)
			}
			if want := []genai.ToolCallResult{{ID: "call_1", Name: f.opens[2][0].Name, Result: `{"code":"violet seven"}`}}; !slices.Equal(results.ToolResults, want) {
				t.Fatalf("results=%v, want %v", results.ToolResults, want)
			}
			imageSent, ok := f.sessions[1].Sent[0].(genai.RealtimeInputContent)
			if len(f.sessions[1].Sent) != 1 || !ok || imageSent.Doc.GetFilename() != "image.jpg" || !imageSent.Generate {
				t.Fatalf("image sent=%v", f.sessions[1].Sent)
			}
		})

		t.Run("usage", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.text = []genai.RealtimeEvent{
				{Kind: genai.RealtimeEventContentText, ResponseID: "r1", Content: genai.RealtimeContent{Text: "hello"}},
				{Kind: genai.RealtimeEventUsage, ResponseID: "r1", Usage: &genai.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12, ServiceTier: "standard"}},
				{Kind: genai.RealtimeEventUsage, ResponseID: "r1", Usage: &genai.Usage{InputTokens: 10, OutputTokens: 2, TotalTokens: 12}},
				{Kind: genai.RealtimeEventUsage, ResponseID: "r2", Usage: &genai.Usage{InputTokens: 3, OutputTokens: 1, TotalTokens: 4}},
				{Kind: genai.RealtimeEventUsage, ResponseID: "r3", Usage: &genai.Usage{Limits: []genai.RateLimit{{Type: genai.Tokens, Period: genai.PerOther, Limit: 10, Remaining: 9, Reset: time.Unix(1, 0)}}}},
				{Kind: genai.RealtimeEventUsage, ResponseID: "r4", Usage: &genai.Usage{Limits: []genai.RateLimit{{Type: genai.Tokens, Period: genai.PerOther, Limit: 20, Remaining: 19, Reset: time.Unix(2, 0)}}}},
				{Kind: genai.RealtimeEventGenerationDone},
				{Kind: genai.RealtimeEventTurnDone},
			}
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if u := report.Usage; u.InputTokens != 13 || u.OutputTokens != 3 || u.TotalTokens != 16 || u.ServiceTier != "standard" || len(u.Limits) != 1 || u.Limits[0].Limit != 20 {
				t.Fatalf("usage=%+v", u)
			}
		})

		t.Run("audio", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.Capabilities.AudioCommit = true
			f.audioTool = fakeToolCalls
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			want := []smoke.RealtimeProbe{smoke.RealtimeProbeSetup, smoke.RealtimeProbeText, smoke.RealtimeProbeReuse, smoke.RealtimeProbeCancellation, smoke.RealtimeProbeAudio, smoke.RealtimeProbeImage, smoke.RealtimeProbeTools, smoke.RealtimeProbeAudioTool}
			if got := ranProbes(&report); !slices.Equal(got, want) {
				t.Fatalf("probes=%v, want %v", got, want)
			}
			if len(f.sessions) != 4 {
				t.Fatalf("sessions=%d", len(f.sessions))
			}
			sent := f.sessions[0].Sent
			lastAudio, ok := sent[len(sent)-1].(genai.RealtimeInputContent)
			if len(sent) < 4 || len(audioChunks(sent)) == 0 || !ok || !lastAudio.Commit || !lastAudio.Generate {
				t.Fatalf("audio inputs=%v", sent)
			}
			toolSent := f.sessions[2].Sent
			if len(toolSent) != 2 {
				t.Fatalf("text tool inputs=%v", toolSent)
			}
			if _, ok := toolSent[0].(genai.RealtimeInputContent); !ok {
				t.Fatalf("text tool inputs=%v", toolSent)
			}
			if _, ok := toolSent[1].(genai.RealtimeInputContent); !ok {
				t.Fatalf("text tool inputs=%v", toolSent)
			}
			audioToolSent := f.sessions[3].Sent
			if len(audioToolSent) < 4 || len(audioChunks(audioToolSent)) == 0 {
				t.Fatalf("audio tool inputs=%v", audioToolSent)
			}
			lastAudio, ok = audioToolSent[len(audioToolSent)-2].(genai.RealtimeInputContent)
			if !ok || !lastAudio.Commit || !lastAudio.Generate {
				t.Fatalf("audio tool inputs=%v", audioToolSent)
			}
			if _, ok := audioToolSent[len(audioToolSent)-1].(genai.RealtimeInputContent); !ok {
				t.Fatalf("audio tool inputs=%v", audioToolSent)
			}
		})

		t.Run("audio tool gates second session", func(t *testing.T) {
			for name, mutate := range map[string]func(*genai.RealtimeSessionInfo){
				"no audio": func(i *genai.RealtimeSessionInfo) {
					i.Capabilities.AudioCommit = false
				},
				"no tools": func(i *genai.RealtimeSessionInfo) { i.Capabilities.ToolResults = false },
			} {
				t.Run(name, func(t *testing.T) {
					f := newFakeRealtimeFactory(t)
					f.audioTool = fakeToolCalls
					second := fakeRealtimeInfo
					mutate(&second)
					f.secondInfo = &second
					report, err := smoke.RunRealtime(t.Context(), f.open)
					if err != nil {
						t.Fatal(err)
					}
					if p := probeResult(t, &report, smoke.RealtimeProbeAudioTool); !p.Skipped {
						t.Fatalf("audio tool probe ran: %+v", p)
					}
				})
			}
		})

		t.Run("audio tool requires spoken output", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.Capabilities.AudioCommit = true
			f.audioTool = fakeToolCalls
			f.audioToolResults = []genai.RealtimeEvent{
				{Kind: genai.RealtimeEventTranscriptOutput, ResponseID: "answer", Content: genai.RealtimeContent{Text: "violet seven"}},
				{Kind: genai.RealtimeEventTurnDone, ResponseID: "answer"},
			}
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			p := probeResult(t, &report, smoke.RealtimeProbeAudioTool)
			if !p.Failed || !strings.Contains(p.Reason, "spoken output") {
				t.Fatalf("audio tool probe=%+v", p)
			}
		})

		t.Run("audio tool checks transcript", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.Capabilities.AudioCommit = true
			f.audioTool = fakeToolCalls
			f.audioToolResults = []genai.RealtimeEvent{
				{Kind: genai.RealtimeEventTranscriptOutput, ResponseID: "answer", Content: genai.RealtimeContent{Text: "something else"}},
				{Kind: genai.RealtimeEventContentAudio, ResponseID: "answer", Content: genai.RealtimeContent{Audio: []byte{1, 0}}},
				{Kind: genai.RealtimeEventTurnDone, ResponseID: "answer"},
			}
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			p := probeResult(t, &report, smoke.RealtimeProbeAudioTool)
			if !p.Failed || !strings.Contains(p.Reason, "does not contain") {
				t.Fatalf("audio tool probe=%+v", p)
			}
		})

		t.Run("audio tool uses its own audio capability", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			second := fakeRealtimeInfo
			second.Capabilities.AudioCommit = true
			f.secondInfo = &second
			f.audioTool = fakeToolCalls
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if p := probeResult(t, &report, smoke.RealtimeProbeAudioTool); p.Skipped {
				t.Fatalf("audio tool probe skipped: %s", p.Reason)
			}
		})

		t.Run("tool fallback skips audio tool", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.toolErr = errors.New("no tools")
			f.audioTool = fakeToolCalls
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if len(f.sessions) != 2 {
				t.Fatalf("sessions=%d", len(f.sessions))
			}
			for _, p := range report.Probes {
				if p.Probe == smoke.RealtimeProbeAudioTool && !p.Skipped {
					t.Fatalf("audio tool probe ran: %+v", p)
				}
			}
		})

		t.Run("audio tool queued completion", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.Capabilities.AudioCommit = true
			f.audioTool = func(tool string) []genai.RealtimeEvent {
				return append(fakeToolCalls(tool), genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone, ResponseID: "call_response"})
			}
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if len(f.sessions) != 4 {
				t.Fatalf("sessions=%d", len(f.sessions))
			}
			if got := ranProbes(&report); !slices.Contains(got, smoke.RealtimeProbeTools) || !slices.Contains(got, smoke.RealtimeProbeAudioTool) {
				t.Fatalf("probes=%v", got)
			}
		})

		t.Run("manual audio requires commit", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.audioTool = fakeToolCalls
			second := fakeRealtimeInfo
			second.Capabilities.AudioCommit = true
			f.secondInfo = &second
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if p := probeResult(t, &report, smoke.RealtimeProbeAudio); !p.Skipped {
				t.Fatalf("audio probe ran: %+v", p)
			}
			if p := probeResult(t, &report, smoke.RealtimeProbeAudioTool); p.Skipped {
				t.Fatalf("audio tool probe skipped: %s", p.Reason)
			}
		})

		t.Run("multiple tool calls", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.toolPrompt = func(tool string) []genai.RealtimeEvent {
				return []genai.RealtimeEvent{
					{Kind: genai.RealtimeEventContentToolCalls, Content: genai.RealtimeContent{ToolCalls: []genai.ToolCall{
						{ID: "call_2", Name: tool, Arguments: "{}"},
						{ID: "call_1", Name: tool, Arguments: "{}"},
					}}},
					{Kind: genai.RealtimeEventGenerationDone},
				}
			}
			if _, err := smoke.RunRealtime(t.Context(), f.open); err != nil {
				t.Fatal(err)
			}
			results := f.sessions[2].Sent[1].(genai.RealtimeInputContent)
			want := []genai.ToolCallResult{
				{ID: "call_2", Name: f.opens[2][0].Name, Result: `{"code":"violet seven"}`},
				{ID: "call_1", Name: f.opens[2][0].Name, Result: `{"code":"violet seven"}`},
			}
			if !slices.Equal(results.ToolResults, want) {
				t.Fatalf("results=%v, want %v", results.ToolResults, want)
			}
		})

		t.Run("filler before tool call", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.toolPrompt = func(tool string) []genai.RealtimeEvent {
				// A non-blocking provider may finish a filler utterance before it
				// emits the tool call.
				return append([]genai.RealtimeEvent{
					{Kind: genai.RealtimeEventContentAudio, Content: genai.RealtimeContent{Audio: []byte{0, 0}}},
					{Kind: genai.RealtimeEventGenerationDone},
				}, fakeToolCalls(tool)...)
			}
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if r := probeResult(t, &report, smoke.RealtimeProbeTools); r.Skipped {
				t.Fatalf("probe tools=%+v", report.Probes)
			}
		})

		t.Run("queued tool turn done", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			// The provider queues the tool-call turn's TurnDone after the call, so the
			// run must not require output from the turn answering the results.
			f.toolPrompt = func(tool string) []genai.RealtimeEvent {
				return append(fakeToolCalls(tool), genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone, ResponseID: "call_response"})
			}
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if r := probeResult(t, &report, smoke.RealtimeProbeTools); r.Skipped {
				t.Fatalf("probe tools=%+v", report.Probes)
			}
		})

		t.Run("image refusal", func(t *testing.T) {
			for name, turn := range map[string][]genai.RealtimeEvent{
				"provider error": {{Kind: genai.RealtimeEventError, Error: &genai.RealtimeError{Code: "invalid_request", Message: "image input is not supported"}}},
				"wrong answer":   {{Kind: genai.RealtimeEventContentText, Content: genai.RealtimeContent{Text: "apple"}}, {Kind: genai.RealtimeEventGenerationDone}, {Kind: genai.RealtimeEventTurnDone}},
			} {
				t.Run(name, func(t *testing.T) {
					f := newFakeRealtimeFactory(t)
					f.image = turn
					report, err := smoke.RunRealtime(t.Context(), f.open)
					if err != nil {
						t.Fatal(err)
					}
					if r := probeResult(t, &report, smoke.RealtimeProbeImage); !r.Failed || r.Reason == "" {
						t.Fatalf("probe image=%+v", r)
					}
				})
			}
		})

		t.Run("skipped capabilities", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.Capabilities = genai.RealtimeCapabilities{}
			f.info.InputModalities = []genai.Modality{genai.ModalityAudio}
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			for _, p := range []smoke.RealtimeProbe{smoke.RealtimeProbeTools, smoke.RealtimeProbeCancellation, smoke.RealtimeProbeImage} {
				if r := probeResult(t, &report, p); !r.Skipped || r.Reason == "" {
					t.Fatalf("probe %s=%+v", p, report.Probes)
				}
			}
			if len(f.sessions) != 3 {
				t.Fatalf("sessions=%d", len(f.sessions))
			}
			if sent := f.sessions[0].Sent; len(sent) != 2 {
				t.Fatalf("sent=%v", sent)
			}
		})

		t.Run("skipped session tools", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.toolErr = &base.ErrNotSupported{Options: []string{"RealtimeOptionTools"}}
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err != nil {
				t.Fatal(err)
			}
			if r := probeResult(t, &report, smoke.RealtimeProbeTools); !r.Skipped || r.Reason == "" {
				t.Fatalf("probe tools=%+v", report.Probes)
			}
			if len(f.opens) != 4 || len(f.opens[0]) != 0 || len(f.opens[1]) != 0 || len(f.opens[2]) != 1 || len(f.opens[3]) != 1 {
				t.Fatalf("opens=%v", f.opens)
			}
			if len(f.sessions) != 2 {
				t.Fatalf("sessions=%d", len(f.sessions))
			}
			sent := f.sessions[0].Sent
			if len(sent) != 4 {
				t.Fatalf("sent=%v", sent)
			}
			if _, ok := sent[2].(genai.RealtimeInputContent); !ok {
				t.Fatalf("sent=%v", sent)
			}
			if _, ok := sent[3].(genai.RealtimeCancel); !ok {
				t.Fatalf("sent=%v", sent)
			}
		})
	})

	t.Run("error", func(t *testing.T) {
		t.Run("factory", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.err = errors.New("connect failed")
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err == nil || !strings.Contains(err.Error(), "probe setup") {
				t.Fatalf("err=%v", err)
			}
			if len(report.Probes) != 0 {
				t.Fatalf("probes=%v", report.Probes)
			}
		})

		t.Run("output format", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.OutputFormat = genai.AudioFormat{}
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "probe setup") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("input modalities", func(t *testing.T) {
			for name, mods := range map[string][]genai.Modality{
				"empty":       nil,
				"unsupported": {genai.ModalityAudio, genai.ModalityVideo},
				"duplicate":   {genai.ModalityAudio, genai.ModalityAudio},
			} {
				t.Run(name, func(t *testing.T) {
					f := newFakeRealtimeFactory(t)
					f.info.InputModalities = mods
					if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "input modalities") {
						t.Fatalf("err=%v", err)
					}
				})
			}
		})

		t.Run("empty turn detection", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.TurnDetection = ""
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if err == nil || !strings.Contains(err.Error(), "turn detection") {
				t.Fatalf("err=%v", err)
			}
			if len(report.Probes) != 0 {
				t.Fatalf("probes=%v", report.Probes)
			}
		})

		t.Run("timeout", func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				f := newFakeRealtimeFactory(t)
				f.text = nil
				ctx, cancel := context.WithTimeout(t.Context(), time.Hour)
				t.Cleanup(cancel)
				report, err := smoke.RunRealtime(ctx, f.open)
				if !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("err=%v", err)
				}
				if got := ranProbes(&report); !slices.Equal(got, []smoke.RealtimeProbe{smoke.RealtimeProbeSetup}) {
					t.Fatalf("probes=%v", got)
				}
				// The factory rejects the expired context as a real one would.
				if _, err := f.open(ctx, smoke.RealtimeProbeSetup, nil); !errors.Is(err, context.DeadlineExceeded) {
					t.Fatalf("factory err=%v", err)
				}
			})
		})

		t.Run("ended", func(t *testing.T) {
			s := realtimetest.NewFakeSession(fakeRealtimeInfo)
			s.Reply = func(turn int, _ genai.RealtimeInput) []genai.RealtimeEvent {
				if turn != 1 {
					t.Fatalf("turn=%d", turn)
				}
				s.End()
				return nil
			}
			f := func(context.Context, smoke.RealtimeProbe, []genai.ToolDef) (genai.RealtimeSession, error) {
				return s, nil
			}
			if _, err := smoke.RunRealtime(t.Context(), f); err == nil || !strings.Contains(err.Error(), "session ended") {
				t.Fatalf("err=%v", err)
			}
			// The first input was accepted, then the stream ended.
			if len(s.Sent) != 1 {
				t.Fatalf("sent=%v", s.Sent)
			}
		})

		t.Run("close error", func(t *testing.T) {
			want := errors.New("close handshake failed")
			f := newFakeRealtimeFactory(t)
			f.closeErr = want
			report, err := smoke.RunRealtime(t.Context(), f.open)
			if !errors.Is(err, want) {
				t.Fatalf("err=%v", err)
			}
			// The core session reported four probes and skipped audio before Close failed.
			if got := len(report.Probes); got != 5 {
				t.Fatalf("probes=%v", report.Probes)
			}
		})

		t.Run("provider error", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.text = []genai.RealtimeEvent{{Kind: genai.RealtimeEventError, Error: &genai.RealtimeError{Code: "invalid_request", Message: "rejected"}}}
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "provider error") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("invalid content", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.text = []genai.RealtimeEvent{{Kind: genai.RealtimeEventContentText, Content: genai.RealtimeContent{Audio: []byte{1, 0}, Text: "hello"}}}
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "invalid text content") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("silent turn", func(t *testing.T) {
			for name, turn := range map[string][]genai.RealtimeEvent{
				"no output":               {{Kind: genai.RealtimeEventTurnDone}},
				"empty text":              {{Kind: genai.RealtimeEventContentText}, {Kind: genai.RealtimeEventTurnDone}},
				"empty audio":             {{Kind: genai.RealtimeEventContentAudio}, {Kind: genai.RealtimeEventTurnDone}},
				"empty output transcript": {{Kind: genai.RealtimeEventTranscriptOutput}, {Kind: genai.RealtimeEventTurnDone}},
			} {
				t.Run(name, func(t *testing.T) {
					f := newFakeRealtimeFactory(t)
					f.text = turn
					if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "without audio or text output") {
						t.Fatalf("err=%v", err)
					}
				})
			}
		})

		t.Run("tool answer missing code", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.toolResults = fakeOutputTurn
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "does not contain") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("no tool call", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.toolPrompt = func(string) []genai.RealtimeEvent {
				return []genai.RealtimeEvent{{Kind: genai.RealtimeEventGenerationDone}, {Kind: genai.RealtimeEventTurnDone}}
			}
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "without a tool call") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("tool call after turn done", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.toolPrompt = func(tool string) []genai.RealtimeEvent {
				return append([]genai.RealtimeEvent{{Kind: genai.RealtimeEventTurnDone}}, fakeToolCalls(tool)...)
			}
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "without a tool call") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("tool mismatch", func(t *testing.T) {
			for name, tc := range map[string]struct {
				calls func(tool string) []genai.ToolCall
				want  string
			}{
				"unknown tool": {
					func(string) []genai.ToolCall {
						return []genai.ToolCall{{ID: "call_1", Name: "other_tool", Arguments: "{}"}}
					},
					"mismatched tool correlation",
				},
				"missing id": {
					func(tool string) []genai.ToolCall { return []genai.ToolCall{{Name: tool, Arguments: "{}"}} },
					"mismatched tool correlation",
				},
				"duplicate id": {
					func(tool string) []genai.ToolCall {
						return []genai.ToolCall{{ID: "call_1", Name: tool, Arguments: "{}"}, {ID: "call_1", Name: tool, Arguments: "{}"}}
					},
					"mismatched tool correlation",
				},
				"invalid arguments": {
					func(tool string) []genai.ToolCall {
						return []genai.ToolCall{{ID: "call_1", Name: tool, Arguments: "{"}}
					},
					"failed to decode tool call arguments",
				},
			} {
				t.Run(name, func(t *testing.T) {
					f := newFakeRealtimeFactory(t)
					f.toolPrompt = func(tool string) []genai.RealtimeEvent {
						return []genai.RealtimeEvent{{Kind: genai.RealtimeEventContentToolCalls, Content: genai.RealtimeContent{ToolCalls: tc.calls(tool)}}}
					}
					if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), tc.want) {
						t.Fatalf("err=%v", err)
					}
				})
			}
		})

		t.Run("late output after interruption", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.cancel = []genai.RealtimeEvent{
				{Kind: genai.RealtimeEventInterrupted},
				{Kind: genai.RealtimeEventContentAudio, Content: genai.RealtimeContent{Audio: []byte{1, 0}}},
				{Kind: genai.RealtimeEventTurnDone},
			}
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "output after interruption") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("event budget", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.text = slices.Repeat([]genai.RealtimeEvent{{Kind: genai.RealtimeEventInputSpeechStart}}, 1001)
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "1000 events without a matching event") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("missing cancellation", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.cancel = []genai.RealtimeEvent{{Kind: genai.RealtimeEventGenerationDone}, {Kind: genai.RealtimeEventTurnDone}}
			if _, err := smoke.RunRealtime(t.Context(), f.open); err == nil || !strings.Contains(err.Error(), "without interruption") {
				t.Fatalf("err=%v", err)
			}
		})
	})
}

// realtimeProvider routes ConnectRealtime to a scripted session.
type realtimeProvider struct {
	base.NotImplemented
	f     *fakeRealtimeFactory
	model string
	caps  genai.ProviderCapabilities
	score scoreboard.Score
	opens [][]genai.RealtimeOption
}

func (p *realtimeProvider) Close() error    { return nil }
func (p *realtimeProvider) Name() string    { return "realtime" }
func (p *realtimeProvider) ModelID() string { return p.model }
func (p *realtimeProvider) OutputModalities() genai.Modalities {
	return genai.Modalities{genai.ModalityAudio}
}
func (p *realtimeProvider) HTTPClient() *http.Client     { return nil }
func (p *realtimeProvider) Scoreboard() scoreboard.Score { return p.score }
func (p *realtimeProvider) Capabilities() genai.ProviderCapabilities {
	return p.caps
}

func (p *realtimeProvider) ConnectRealtime(ctx context.Context, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	p.opens = append(p.opens, opts)
	probe := smoke.RealtimeProbeSetup
	var tools []genai.ToolDef
	for _, o := range opts {
		switch v := o.(type) {
		case *genai.RealtimeOptionTools:
			tools = v.Tools
		case genai.Modalities:
			if slices.Contains(v, genai.ModalityText) {
				probe = smoke.RealtimeProbeTools
			} else if slices.Contains(v, genai.ModalityAudio) {
				probe = smoke.RealtimeProbeAudioTool
			}
		}
	}
	return p.f.open(ctx, probe, tools)
}

// rejectedRealtimeProvider turns down every session.
type rejectedRealtimeProvider struct {
	base.NotImplemented
	caps  genai.ProviderCapabilities
	opens int
}

func (p *rejectedRealtimeProvider) Close() error             { return nil }
func (p *rejectedRealtimeProvider) Name() string             { return "rejected" }
func (p *rejectedRealtimeProvider) ModelID() string          { return "model" }
func (p *rejectedRealtimeProvider) HTTPClient() *http.Client { return nil }
func (p *rejectedRealtimeProvider) OutputModalities() genai.Modalities {
	return genai.Modalities{genai.ModalityAudio}
}
func (p *rejectedRealtimeProvider) Scoreboard() scoreboard.Score             { return scoreboard.Score{} }
func (p *rejectedRealtimeProvider) Capabilities() genai.ProviderCapabilities { return p.caps }

func (p *rejectedRealtimeProvider) ConnectRealtime(context.Context, ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	p.opens++
	return nil, errors.New("model does not support realtime")
}

func TestRealtimeSupported(t *testing.T) {
	t.Run("provider without realtime", func(t *testing.T) {
		p := &rejectedRealtimeProvider{}
		ok, reason, err := smoke.RealtimeSupported(t.Context(), p)
		if ok || err != nil || reason == "" || p.opens != 0 {
			t.Fatalf("ok=%v reason=%q err=%v opens=%d", ok, reason, err, p.opens)
		}
	})
	t.Run("model rejected", func(t *testing.T) {
		p := &rejectedRealtimeProvider{caps: genai.ProviderCapabilities{Realtime: true}}
		ok, reason, err := smoke.RealtimeSupported(t.Context(), p)
		if ok || err != nil || !strings.Contains(reason, "does not support realtime") || p.opens != 1 {
			t.Fatalf("ok=%v reason=%q err=%v opens=%d", ok, reason, err, p.opens)
		}
	})
	t.Run("accepted", func(t *testing.T) {
		p := &realtimeProvider{f: newFakeRealtimeFactory(t), model: "model", caps: genai.ProviderCapabilities{Realtime: true}}
		ok, reason, err := smoke.RealtimeSupported(t.Context(), p)
		if !ok || err != nil || reason != "" || len(p.opens) != 1 {
			t.Fatalf("ok=%v reason=%q err=%v opens=%d", ok, reason, err, len(p.opens))
		}
	})
}

func TestRunRealtimeScenario(t *testing.T) {
	realtimeScoreboard := func() scoreboard.Score {
		return scoreboard.Score{Scenarios: []scoreboard.Scenario{{
			Models:   []string{"model"},
			In:       map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}, scoreboard.ModalityText: {Inline: true}},
			Out:      map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityAudio: {Inline: true}, scoreboard.ModalityText: {Inline: true}},
			Realtime: &scoreboard.RealtimeFunctionality{},
		}}}
	}

	t.Run("valid", func(t *testing.T) {
		t.Run("success", func(t *testing.T) {
			p := &realtimeProvider{f: newFakeRealtimeFactory(t), model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: realtimeScoreboard()}
			sc, _, err := smoke.RunRealtimeScenario(t.Context(), p, false)
			if err != nil {
				t.Fatal(err)
			}
			if err := sc.Validate(); err != nil {
				t.Fatal(err)
			}
			if sc.Reason {
				t.Fatal("scenario declares thinking")
			}
			if !slices.Equal(sc.Models, []string{"model"}) {
				t.Fatalf("models=%v", sc.Models)
			}
			if !sc.In[genai.ModalityText].Inline || !sc.In[genai.ModalityImage].Inline || sc.In[genai.ModalityAudio].Inline {
				t.Fatalf("input modalities: %+v", sc.In)
			}
			for _, m := range []genai.Modality{genai.ModalityAudio, genai.ModalityText} {
				if !sc.Out[m].Inline {
					t.Fatalf("output modality %s: %+v", m, sc.Out[m])
				}
			}
			want := scoreboard.RealtimeFunctionality{InputSampleRate: 16000, OutputSampleRate: 24000, CancelResponse: true, Tools: scoreboard.True}
			if sc.Realtime == nil || *sc.Realtime != want {
				t.Fatalf("realtime=%+v, want %+v", sc.Realtime, want)
			}
			// The core session opens without tools. The image probe opens a text session
			// without tools; each tool probe opens a session with its own output modality
			// and text-only function-calling instructions.
			if len(p.opens) != 4 || len(p.opens[0]) != 1 || len(p.opens[1]) != 2 || len(p.opens[2]) != 3 || len(p.opens[3]) != 3 {
				t.Fatalf("opens=%v", p.opens)
			}
			if in, ok := p.opens[2][0].(genai.RealtimeOptionInstructions); !ok || in != "List of tools:" {
				t.Fatalf("instructions=%q", p.opens[2][0])
			}
			if m, ok := p.opens[1][1].(genai.Modalities); !ok || !slices.Equal(m, genai.Modalities{genai.ModalityText}) {
				t.Fatalf("image modalities=%v", p.opens[1][1])
			}
			if m, ok := p.opens[2][1].(genai.Modalities); !ok || !slices.Equal(m, genai.Modalities{genai.ModalityText}) {
				t.Fatalf("modalities=%v", p.opens[2][1])
			}
			if _, ok := p.opens[2][2].(*genai.RealtimeOptionTools); !ok {
				t.Fatalf("options=%v", p.opens[2])
			}
			if m, ok := p.opens[3][1].(genai.Modalities); !ok || !slices.Equal(m, genai.Modalities{genai.ModalityAudio}) {
				t.Fatalf("audio modalities=%v", p.opens[3][1])
			}

			// Run routes a scoreboard realtime scenario before GenSync.
			sc, usage, err := smoke.Run(t.Context(), func(string) genai.Provider { return p })
			if err != nil {
				t.Fatal(err)
			}
			if usage.TotalTokens != 0 || len(usage.Limits) != 0 || sc.Realtime == nil || *sc.Realtime != want {
				t.Fatalf("usage=%+v scenario=%+v", usage, sc.Realtime)
			}
		})

		t.Run("usage", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.text = []genai.RealtimeEvent{
				{Kind: genai.RealtimeEventContentText, Content: genai.RealtimeContent{Text: "hello"}},
				{Kind: genai.RealtimeEventUsage, ResponseID: "r1", Usage: &genai.Usage{InputTokens: 5, OutputTokens: 1, TotalTokens: 6}},
				{Kind: genai.RealtimeEventGenerationDone},
				{Kind: genai.RealtimeEventTurnDone},
			}
			p := &realtimeProvider{f: f, model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: realtimeScoreboard()}
			sc, usage, err := smoke.RunRealtimeScenario(t.Context(), p, false)
			if err != nil {
				t.Fatal(err)
			}
			if sc.Realtime == nil || usage.TotalTokens != 6 {
				t.Fatalf("scenario=%+v usage=%+v", sc, usage)
			}
			_, usage, err = smoke.Run(t.Context(), func(string) genai.Provider { return p })
			if err != nil {
				t.Fatal(err)
			}
			if usage.TotalTokens != 6 {
				t.Fatalf("usage=%+v", usage)
			}
		})

		t.Run("audio", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.Capabilities.AudioCommit = true
			f.audioTool = fakeToolCalls
			p := &realtimeProvider{f: f, model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: realtimeScoreboard()}
			sc, _, err := smoke.RunRealtimeScenario(t.Context(), p, false)
			if err != nil {
				t.Fatal(err)
			}
			if sc.Realtime == nil || !sc.In[genai.ModalityAudio].Inline || sc.Realtime.Tools != scoreboard.True {
				t.Fatalf("scenario=%+v", sc)
			}
		})

		t.Run("tool probe modalities ignore scoreboard", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.Capabilities.AudioCommit = true
			f.audioTool = fakeToolCalls
			score := realtimeScoreboard()
			delete(score.Scenarios[0].Out, genai.ModalityText)
			delete(score.Scenarios[0].Out, genai.ModalityAudio)
			p := &realtimeProvider{f: f, model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: score}
			sc, _, err := smoke.RunRealtimeScenario(t.Context(), p, false)
			if err != nil {
				t.Fatal(err)
			}
			if sc.Realtime == nil || sc.Realtime.Tools != scoreboard.True {
				t.Fatalf("scenario=%+v", sc)
			}
		})

		t.Run("reason", func(t *testing.T) {
			score := realtimeScoreboard()
			score.Scenarios[0].Reason = true
			p := &realtimeProvider{f: newFakeRealtimeFactory(t), model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: score}
			sc, _, err := smoke.RunRealtimeScenario(t.Context(), p, true)
			if err != nil {
				t.Fatal(err)
			}
			if !sc.Reason {
				t.Fatal("scenario does not declare thinking")
			}

			// Run declares the reasoning mode from the matching scoreboard scenario.
			sc, _, err = smoke.Run(t.Context(), func(string) genai.Provider { return p })
			if err != nil {
				t.Fatal(err)
			}
			if !sc.Reason {
				t.Fatal("scoreboard reason not applied")
			}
		})
	})

	t.Run("error", func(t *testing.T) {
		t.Run("unsupported", func(t *testing.T) {
			p := &realtimeProvider{f: newFakeRealtimeFactory(t), model: "model"}
			if sc, _, err := smoke.RunRealtimeScenario(t.Context(), p, false); err == nil || sc.Realtime != nil {
				t.Fatalf("scenario=%+v err=%v", sc, err)
			}
		})

		t.Run("no model", func(t *testing.T) {
			p := &realtimeProvider{f: newFakeRealtimeFactory(t), caps: genai.ProviderCapabilities{Realtime: true}}
			if _, _, err := smoke.RunRealtimeScenario(t.Context(), p, false); err == nil {
				t.Fatal("missing model accepted")
			}
		})

		t.Run("flaky audio tool results", func(t *testing.T) {
			for name, events := range map[string][]genai.RealtimeEvent{
				"pass": fakeToolAnswer,
				"fail": {
					{Kind: genai.RealtimeEventContentText, ResponseID: "answer", Content: genai.RealtimeContent{Text: "violet seven"}},
					{Kind: genai.RealtimeEventTurnDone, ResponseID: "answer"},
				},
			} {
				t.Run(name, func(t *testing.T) {
					f := newFakeRealtimeFactory(t)
					f.info.Capabilities.AudioCommit = true
					f.audioTool = fakeToolCalls
					f.audioToolResults = events
					score := realtimeScoreboard()
					score.Scenarios[0].Realtime.Tools = scoreboard.Flaky
					p := &realtimeProvider{f: f, model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: score}
					sc, _, err := smoke.RunRealtimeScenario(t.Context(), p, false)
					if err != nil {
						t.Fatal(err)
					}
					if sc.Realtime == nil || sc.Realtime.Tools != scoreboard.Flaky {
						t.Fatalf("scenario=%+v", sc)
					}
				})
			}
		})

		t.Run("image regression", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.InputModalities = []genai.Modality{genai.ModalityAudio, genai.ModalityText}
			score := realtimeScoreboard()
			score.Scenarios[0].In[genai.ModalityImage] = scoreboard.ModalCapability{Inline: true}
			p := &realtimeProvider{f: f, model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: score}
			if _, _, err := smoke.RunRealtimeScenario(t.Context(), p, false); err == nil || !strings.Contains(err.Error(), "image input regressed") {
				t.Fatalf("err=%v", err)
			}
		})

		t.Run("image regression reason", func(t *testing.T) {
			// Each reasoning mode has its own scoreboard entry; a recorded image in
			// one mode applies to that mode only.
			f := newFakeRealtimeFactory(t)
			f.info.InputModalities = []genai.Modality{genai.ModalityAudio, genai.ModalityText}
			score := scoreboard.Score{Scenarios: []scoreboard.Scenario{
				{Models: []string{"model"}, Reason: true, In: map[scoreboard.Modality]scoreboard.ModalCapability{scoreboard.ModalityImage: {Inline: true}}, Realtime: &scoreboard.RealtimeFunctionality{}},
				{Models: []string{"model"}, In: map[scoreboard.Modality]scoreboard.ModalCapability{}, Realtime: &scoreboard.RealtimeFunctionality{}},
			}}
			p := &realtimeProvider{f: f, model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: score}
			if _, _, err := smoke.RunRealtimeScenario(t.Context(), p, false); err != nil {
				t.Fatalf("non-thinking mode inherited the thinking result: %v", err)
			}
			if _, _, err := smoke.RunRealtimeScenario(t.Context(), p, true); err == nil || !strings.Contains(err.Error(), "image input regressed") {
				t.Fatalf("thinking mode err=%v", err)
			}
		})

		t.Run("tool regression", func(t *testing.T) {
			f := newFakeRealtimeFactory(t)
			f.info.Capabilities.ToolResults = false
			score := realtimeScoreboard()
			score.Scenarios[0].Realtime.Tools = scoreboard.True
			p := &realtimeProvider{f: f, model: "model", caps: genai.ProviderCapabilities{Realtime: true}, score: score}
			if _, _, err := smoke.RunRealtimeScenario(t.Context(), p, false); err == nil || !strings.Contains(err.Error(), "regressed") {
				t.Fatalf("err=%v", err)
			}
		})
	})
}
