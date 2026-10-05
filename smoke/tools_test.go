// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for interactive chronological conversations with deferred tool results.

package smoke

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"slices"
	"testing"

	"github.com/maruel/httpjson"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

func TestExerciseGenConversation(t *testing.T) {
	for _, stream := range []bool{false, true} {
		mode := "GenSync"
		if stream {
			mode = "GenStream"
		}
		t.Run(mode, func(t *testing.T) {
			t.Run("valid", func(t *testing.T) {
				for _, tc := range []struct {
					name           string
					initial, later []int
				}{
					{"single_call", []int{7}, nil},
					{"batch_calls", []int{3, 7, 11}, nil},
					{"consecutive_calls", []int{3}, []int{7}},
					{"repeated_task", []int{7}, []int{7}},
				} {
					t.Run(tc.name, func(t *testing.T) {
						p := &conversationProvider{t: t, initial: tc.initial, later: tc.later}
						cs := callState{isStream: stream, pf: func(label string) genai.Provider {
							if label != fmt.Sprintf("%s-OutOfOrder-%d", mode, p.calls+1) {
								t.Fatalf("label=%q", label)
							}
							return p
						}}
						f := scoreboard.Functionality{Tools: scoreboard.True, ReportTokenUsage: scoreboard.True, ReportFinishReason: scoreboard.True}
						got, err := exerciseGenConversation(t.Context(), &cs, &f, mode+"-OutOfOrder")
						if err != nil || got != scoreboard.True || p.calls != 3 {
							t.Fatalf("score=%s calls=%d err=%v", got, p.calls, err)
						}
						if f.Tools != scoreboard.True || f.ReportTokenUsage != scoreboard.True || f.ReportFinishReason != scoreboard.True {
							t.Fatalf("changed unrelated capabilities: %+v", f)
						}
					})
				}
			})
			t.Run("error", func(t *testing.T) {
				for _, tc := range []struct {
					name           string
					initial, later []int
					err            error
					abort          bool
					duplicateID    bool
					calls          int
				}{
					{name: "no_calls", calls: 1},
					{name: "duplicate_call_id", initial: []int{7}, later: []int{7}, duplicateID: true, calls: 2},
					{name: "invalid_task", initial: []int{99}, calls: 1},
					{name: "initial_rejection", err: &httpjson.Error{StatusCode: http.StatusBadRequest}, abort: true, calls: 1},
					{name: "continuation_rejection", initial: []int{7}, err: &httpjson.Error{StatusCode: http.StatusBadRequest}, calls: 2},
					{name: "unprocessable_continuation", initial: []int{7}, err: &httpjson.Error{StatusCode: http.StatusUnprocessableEntity}, calls: 2},
					{name: "unsupported_tools", initial: []int{7}, err: &base.ErrNotSupported{Options: []string{"GenOptionTools"}}, calls: 2},
					{name: "unsupported_option", initial: []int{7}, err: &base.ErrNotSupported{Options: []string{"GenOptionText.Seed"}}, abort: true, calls: 2},
					{name: "decode", initial: []int{7}, err: errors.Join(&httpjson.Error{StatusCode: http.StatusBadRequest}, &httpjson.UnknownFieldError{Field: "new_field"}), abort: true, calls: 2},
					{name: "authentication", initial: []int{7}, err: &httpjson.Error{StatusCode: 401}, abort: true, calls: 2},
					{name: "quota", initial: []int{7}, err: &httpjson.Error{StatusCode: 429}, abort: true, calls: 2},
					{name: "network", initial: []int{7}, err: context.DeadlineExceeded, abort: true, calls: 2},
					{name: "recording", initial: []int{7}, err: cassette.ErrInteractionNotFound, abort: true, calls: 2},
					{name: "library", initial: []int{7}, err: errors.Join(&httpjson.Error{StatusCode: http.StatusBadRequest}, &internal.BadError{Err: errors.New("conversion failure")}), abort: true, calls: 2},
					{name: "server", initial: []int{7}, err: &httpjson.Error{StatusCode: http.StatusInternalServerError}, abort: true, calls: 2},
				} {
					t.Run(tc.name, func(t *testing.T) {
						p := &conversationProvider{t: t, initial: tc.initial, later: tc.later, err: tc.err, errStep: tc.calls - 1, duplicateID: tc.duplicateID}
						cs := callState{isStream: stream, pf: func(string) genai.Provider { return p }}
						var f scoreboard.Functionality
						got, err := exerciseGenConversation(t.Context(), &cs, &f, mode+"-OutOfOrder")
						if (err != nil) != tc.abort || got != scoreboard.False || p.calls != tc.calls {
							t.Fatalf("score=%s calls=%d err=%v", got, p.calls, err)
						}
					})
				}
			})
		})
	}
}

func TestConversationAnswer(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		pending := []conversationResult{{Task: conversationTask{ID: 7, Reference: "violet"}}, {Task: conversationTask{ID: 3, Reference: "amber"}}}
		m := genai.NewTextMessage("I'm still waiting for task 7's result.")
		if err := conversationAnswer(&m, nil, pending); err != nil {
			t.Fatal(err)
		}
		m = genai.NewTextMessage("Task seven's reference is Violet.")
		if err := conversationAnswer(&m, []conversationTask{{ID: 7, Status: "waiting", Reference: "violet"}}, nil); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct{ name, text string }{
			{"undelivered", "Task 7's reference is violet."},
			{"empty", ""},
			{"json", `{"reference":"violet"}`},
		} {
			t.Run(tc.name, func(t *testing.T) {
				m := genai.NewTextMessage(tc.text)
				if err := conversationAnswer(&m, nil, []conversationResult{{Task: conversationTask{ID: 7, Reference: "violet"}}}); err == nil {
					t.Fatal("accepted incorrect answer")
				}
			})
		}
		t.Run("delivered", func(t *testing.T) {
			for _, tc := range []struct{ name, text string }{
				{"wrong_task", "Task 7's reference is amber."},
				{"extra_task", "Task 7 is violet; task 3 is amber."},
				{"missing_reference", "I have the result now."},
				{"partial_word", "The reference is ultraviolet."},
			} {
				t.Run(tc.name, func(t *testing.T) {
					m := genai.NewTextMessage(tc.text)
					if err := conversationAnswer(&m, []conversationTask{{ID: 7, Reference: "violet"}, {ID: 3, Reference: "amber"}}, nil); err == nil {
						t.Fatalf("accepted incorrect answer %q", tc.text)
					}
				})
			}
		})
	})
}

type conversationProvider struct {
	base.NotImplemented
	t              *testing.T
	initial, later []int
	err            error
	errStep        int
	duplicateID    bool
	calls          int
}

func (p *conversationProvider) Close() error    { return nil }
func (p *conversationProvider) Name() string    { return "conversation" }
func (p *conversationProvider) ModelID() string { return "model" }
func (p *conversationProvider) OutputModalities() genai.Modalities {
	return genai.Modalities{genai.ModalityText}
}
func (p *conversationProvider) HTTPClient() *http.Client     { return nil }
func (p *conversationProvider) Scoreboard() scoreboard.Score { return scoreboard.Score{} }
func (p *conversationProvider) GenSync(_ context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	return p.next(msgs, opts)
}
func (p *conversationProvider) GenStream(_ context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	r, err := p.next(msgs, opts)
	return func(yield func(genai.Reply) bool) {
		for i := range r.Replies {
			if !yield(r.Replies[i]) {
				return
			}
		}
	}, func() (genai.Result, error) { return r, err }
}
func (p *conversationProvider) next(msgs genai.Messages, opts []genai.GenOption) (genai.Result, error) {
	if err := msgs.Validate(); err != nil {
		p.t.Fatal(err)
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			p.t.Fatal(err)
		}
		if text, ok := opt.(*genai.GenOptionText); ok && (text.ReplyAsJSON || text.DecodeAs != nil) {
			p.t.Fatal("requires provider JSON mode")
		}
	}
	step := p.calls
	p.calls++
	r := genai.Result{Usage: genai.Usage{InputTokens: 1, OutputTokens: 1, FinishReason: genai.FinishedStop}}
	if p.err != nil && step == p.errStep {
		return r, p.err
	}
	calls := func(tasks []int) {
		for _, task := range tasks {
			callStep := step
			if p.duplicateID {
				callStep = 0
			}
			r.Replies = append(r.Replies, genai.Reply{ToolCall: genai.ToolCall{ID: fmt.Sprintf("native-%d-%d", task, callStep), Name: "task_status", Arguments: fmt.Sprintf(`{"task_id":%d}`, task), Opaque: map[string]any{"provenance": "native"}}})
		}
		if len(tasks) > 0 {
			r.Usage.FinishReason = genai.FinishedToolCalls
		}
	}
	switch step {
	case 0:
		if len(msgs) != 2 || msgs[0].Role() != "user" || msgs[1].Role() != "user" {
			p.t.Fatalf("initial history=%+v", msgs)
		}
		calls(p.initial)
	case 1:
		if len(msgs) < 5 || msgs[len(msgs)-2].Role() != "user" || msgs[len(msgs)-1].Role() != "user" {
			p.t.Fatal("lost consecutive user corrections")
		}
		if len(msgs[2].Replies) == 0 || msgs[2].Replies[0].ToolCall.Opaque["provenance"] != "native" {
			p.t.Fatal("lost real assistant provenance")
		}
		for _, m := range msgs {
			if len(m.ToolCallResults) != 0 {
				p.t.Fatal("did not defer tool results across a generation")
			}
		}
		if len(p.later) > 0 {
			calls(p.later)
		} else {
			r.Replies = []genai.Reply{{Text: "I'm still waiting for task 7's result."}}
		}
	case 2:
		want := append(slices.Clone(p.initial), p.later...)
		slices.Reverse(want)
		var got []int
		for _, m := range msgs {
			for _, result := range m.ToolCallResults {
				var task conversationTask
				if err := json.Unmarshal([]byte(result.Result), &task); err != nil {
					p.t.Fatal(err)
				}
				callStep := 0
				if len(got) < len(p.later) {
					callStep = 1
				}
				if result.ID != fmt.Sprintf("native-%d-%d", task.ID, callStep) || result.Name != "task_status" {
					p.t.Fatal("lost native result correlation")
				}
				got = append(got, task.ID)
			}
		}
		if !slices.Equal(want, got) {
			p.t.Fatalf("result order=%v want=%v", got, want)
		}
		r.Replies = []genai.Reply{{Text: "Task 7's reference is violet."}}
	default:
		p.t.Fatalf("unexpected generation %d", step+1)
	}
	return r, nil
}
