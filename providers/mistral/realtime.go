// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Realtime transcription sessions over Mistral's WebSocket API.

package mistral

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/url"
	"strconv"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/realtime"
)

const (
	realtimeURL = "wss://api.mistral.ai/v1/audio/transcriptions/realtime"
	// realtimeInputRate is the service default input sample rate.
	realtimeInputRate = 16000
	// maxRealtimeMessageBytes bounds a server message; transcription.done
	// carries the whole transcript and its segments.
	maxRealtimeMessageBytes = 4 << 20
)

// RealtimeOption configures a Mistral realtime session beyond the portable
// contract.
type RealtimeOption struct {
	// Endpoint replaces the WebSocket URL with a trusted proxy. The client's
	// authentication transport is used for this endpoint too. No query is
	// allowed; ConnectRealtime adds the model.
	Endpoint string
}

// Validate implements genai.RealtimeOption.
func (o *RealtimeOption) Validate() error {
	if o.Endpoint != "" {
		u, err := url.Parse(o.Endpoint)
		if err != nil || (u.Scheme != "ws" && u.Scheme != "wss") || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
			return errors.New("mistral realtime endpoint must be an absolute ws/wss URL without credentials, query or fragment")
		}
	}
	return nil
}

// realtimeSession recognizes one utterance.
type realtimeSession struct {
	ws   *realtime.Conn
	info genai.RealtimeSessionInfo
	// pending is receiver-owned and holds the untaken events of one server
	// message, so a consumer that stops early loses none.
	pending []genai.RealtimeEvent

	mu        sync.Mutex
	committed bool
}

// Info implements genai.RealtimeSession.
func (s *realtimeSession) Info() genai.RealtimeSessionInfo { return s.info }

// Close implements genai.RealtimeSession.
func (s *realtimeSession) Close() error { return s.ws.Close() }

// Send implements genai.RealtimeSession. It accepts audio until Commit.
func (s *realtimeSession) Send(ctx context.Context, in genai.RealtimeInput) error {
	if in == nil {
		return errors.New("realtime input is nil")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	v, ok := in.(genai.RealtimeInputContent)
	if !ok || v.Audio == nil && !v.Commit || v.Generate {
		return fmt.Errorf("mistral transcription accepts only audio and commits: %w", &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}})
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.committed {
		return errors.New("mistral transcription already committed its utterance")
	}
	if len(v.Audio) != 0 {
		if err := s.send(ctx, &RealtimeClientMessage{Type: RealtimeInputAudioAppend, Audio: v.Audio}); err != nil {
			return err
		}
	}
	if v.Commit {
		// The official client flushes before it ends the input.
		if err := s.send(ctx, &RealtimeClientMessage{Type: RealtimeInputAudioFlush}); err != nil {
			return err
		}
		if err := s.send(ctx, &RealtimeClientMessage{Type: RealtimeInputAudioEnd}); err != nil {
			return err
		}
		s.committed = true
	}
	return nil
}

// Receive implements genai.RealtimeSession. Text deltas become input
// transcript deltas; transcription.done is final and ends the turn.
func (s *realtimeSession) Receive(ctx context.Context) iter.Seq2[genai.RealtimeEvent, error] {
	return func(yield func(genai.RealtimeEvent, error) bool) {
		for {
			if len(s.pending) != 0 {
				e := s.pending[0]
				s.pending[0] = genai.RealtimeEvent{}
				s.pending = s.pending[1:]
				if !yield(e, nil) {
					return
				}
				continue
			}
			m, err := s.receive(ctx)
			if err != nil {
				yield(genai.RealtimeEvent{}, err)
				return
			}
			switch m.Type {
			case RealtimeTranscriptionTextDelta:
				if m.Text != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: m.Text}})
				}
			case RealtimeTranscriptionDone:
				if m.Text != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, Content: genai.RealtimeContent{Text: m.Text, Final: true}})
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone})
			case RealtimeError:
				e := &genai.RealtimeError{Message: "mistral transcription failed"}
				if m.Error != nil {
					e = &genai.RealtimeError{Code: strconv.Itoa(m.Error.Code), Message: m.Error.Text()}
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventError, Error: e})
			case RealtimeSessionCreated, RealtimeSessionUpdated, RealtimeTranscriptionLanguage, RealtimeTranscriptionSegment,
				RealtimeTurnEagerEnd, RealtimeTurnEnd, RealtimeTurnResume, RealtimeTurnStart:
				// Segments repeat the deltas' text; turn events need server turns.
			default:
				// The official client ignores unknown messages too.
			}
		}
	}
}

func (s *realtimeSession) send(ctx context.Context, m *RealtimeClientMessage) error {
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return s.ws.Write(ctx, data)
}

// receive reads one server message; invalid JSON terminates the connection.
func (s *realtimeSession) receive(ctx context.Context) (RealtimeServerMessage, error) {
	var m RealtimeServerMessage
	data, err := s.ws.Read(ctx)
	if err != nil {
		return m, err
	}
	if err := internal.UnmarshalJSON(data, &m); err != nil {
		s.ws.Abort(err)
		return RealtimeServerMessage{}, err
	}
	return m, nil
}

var _ genai.RealtimeSession = (*realtimeSession)(nil)
