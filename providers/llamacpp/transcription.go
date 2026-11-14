// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Dedicated live transcription through llama.cpp's streaming audio encoder,
// natively and as a genai.RealtimeSession.

package llamacpp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/realtime"
)

// TranscriptionConn owns native llamacpp WebSocket I/O, without transcript accumulation
// or utterance-commit policy. One receiver and one sender may run concurrently.
// Canceling an I/O context aborts the connection. The caller closes it once.
type TranscriptionConn struct {
	// Initial is the complete native setup acknowledgement, including negotiated
	// session fields. Treat it as immutable; it does not drive transport policy.
	Initial ServerMessage

	ws *realtime.Conn
}

// Send writes a native client message. Audio encoding and turn controls belong
// to the caller; no text accumulation or commit policy is applied.
func (c *TranscriptionConn) Send(ctx context.Context, in *ClientMessage) error {
	if in == nil {
		return errors.New("native client message is required")
	}
	data, err := json.Marshal(in)
	if err != nil {
		return err
	}
	return c.ws.Write(ctx, data)
}

// Receive returns one complete native server message. Invalid JSON terminates
// the connection; native provider error payloads remain available to the caller.
func (c *TranscriptionConn) Receive(ctx context.Context) (ServerMessage, error) {
	var msg ServerMessage
	data, err := c.ws.Read(ctx)
	if err != nil {
		return msg, err
	}
	if err := internal.UnmarshalJSON(data, &msg); err != nil {
		c.ws.Abort(err)
		return ServerMessage{}, err
	}
	return msg, nil
}

// Abort terminates pending reads/writes with the caller's cause. Close is still
// required to wait for transport cleanup.
func (c *TranscriptionConn) Abort(err error) { c.ws.Abort(err) }

// Close terminates I/O and waits for socket cleanup.
func (c *TranscriptionConn) Close() error { return c.ws.Close() }

// newTranscriptionSession adapts a native transcription connection to
// genai.RealtimeSession. It owns conn, closing it on error.
func newTranscriptionSession(conn *TranscriptionConn, format genai.AudioFormat) (genai.RealtimeSession, error) {
	native, err := nativeFormat(conn.Initial.Session.Audio.Input.Format)
	if err != nil {
		return nil, errors.Join(err, conn.Close())
	}
	if format.SampleRate != 0 && format != native {
		return nil, errors.Join(fmt.Errorf("llama.cpp transcription input is PCM16 at %d Hz: %w", native.SampleRate, &base.ErrNotSupported{Options: []string{"input sample rate"}}), conn.Close())
	}
	info := genai.RealtimeSessionInfo{
		ID:              conn.Initial.Session.ID,
		InputFormat:     native,
		InputModalities: []genai.Modality{genai.ModalityAudio},
		TurnDetection:   genai.RealtimeTurnManual,
		Capabilities:    genai.RealtimeCapabilities{AudioCommit: true},
	}
	return &transcriptionSession{conn: conn, info: info}, nil
}

// transcriptionSession recognizes one utterance.
type transcriptionSession struct {
	conn *TranscriptionConn
	info genai.RealtimeSessionInfo

	mu        sync.Mutex
	committed bool
}

// Info implements genai.RealtimeSession.
func (s *transcriptionSession) Info() genai.RealtimeSessionInfo { return s.info }

// Close implements genai.RealtimeSession.
func (s *transcriptionSession) Close() error { return s.conn.Close() }

// Send implements genai.RealtimeSession. It accepts audio until Commit.
func (s *transcriptionSession) Send(ctx context.Context, in genai.RealtimeInput) error {
	if in == nil {
		return errors.New("realtime input is nil")
	}
	if err := in.Validate(); err != nil {
		return err
	}
	v, ok := in.(genai.RealtimeInputContent)
	if !ok || v.Audio == nil && !v.Commit || v.Generate {
		return fmt.Errorf("llama.cpp transcription accepts only audio and commits: %w", &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}})
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.committed {
		return errors.New("llama.cpp transcription already committed its utterance")
	}
	if len(v.Audio) != 0 {
		if err := s.conn.Send(ctx, &ClientMessage{Type: EventAudioAppend, Audio: v.Audio}); err != nil {
			return err
		}
	}
	if v.Commit {
		if err := s.conn.Send(ctx, &ClientMessage{Type: EventAudioCommit, Final: true}); err != nil {
			return err
		}
		s.committed = true
	}
	return nil
}

// Receive implements genai.RealtimeSession. Deltas become input transcript
// deltas; the completed transcript is final and ends the turn.
func (s *transcriptionSession) Receive(ctx context.Context) iter.Seq2[genai.RealtimeEvent, error] {
	return func(yield func(genai.RealtimeEvent, error) bool) {
		for {
			m, err := s.conn.Receive(ctx)
			if err != nil {
				yield(genai.RealtimeEvent{}, err)
				return
			}
			var events []genai.RealtimeEvent
			switch m.Type {
			case EventInputTranscriptDelta:
				if m.Delta != "" {
					events = append(events, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, ItemID: m.ItemID, Content: genai.RealtimeContent{Text: m.Delta}})
				}
			case EventInputTranscriptDone:
				if m.Transcript != "" {
					events = append(events, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, ItemID: m.ItemID, Content: genai.RealtimeContent{Text: m.Transcript, Final: true}})
				}
				events = append(events, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone, ItemID: m.ItemID})
			case EventInputTranscriptFailed, EventError:
				e := &genai.RealtimeError{Message: "llama.cpp transcription failed"}
				if m.Error != nil {
					e = &genai.RealtimeError{Code: m.Error.Code, Message: m.Error.Message}
				}
				events = append(events, genai.RealtimeEvent{Kind: genai.RealtimeEventError, ItemID: m.ItemID, Error: e})
				if m.Type == EventInputTranscriptFailed {
					events = append(events, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone, ItemID: m.ItemID})
				}
			default:
				// Acknowledgements carry no transcript.
			}
			for i := range events {
				if !yield(events[i], nil) {
					return
				}
			}
		}
	}
}

var _ genai.RealtimeSession = (*transcriptionSession)(nil)
