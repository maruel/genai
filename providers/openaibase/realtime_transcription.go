// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// OpenAI GA realtime transcription sessions as genai.RealtimeSession.

package openaibase

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"reflect"
	"sync"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/realtime"
)

// transcriptionRate is the only GA PCM input rate.
const transcriptionRate = 24000

// connectTranscriptionSession opens a GA transcription session
// (/v1/realtime?intent=transcription) where model is the transcription model,
// such as gpt-live-transcribe. The session transcribes one utterance of PCM16
// at 24 kHz with manual turns: Commit commits the input buffer and later audio
// is rejected. RealtimeOptionAudio.InputLanguage is sent as its ISO-639-1 code.
// RealtimeOption accepts Endpoint and the keepalive durations.
//
// Protocol:
// https://developers.openai.com/api/reference/resources/realtime/client-events#session.update
func connectTranscriptionSession(ctx context.Context, hc *http.Client, model string, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	if model == "" {
		return nil, errors.New("OpenAI realtime transcription requires an explicit model")
	}
	tr := Transcription{Model: model}
	o := RealtimeOption{}
	var unsupported []string
	seen := make(map[reflect.Type]struct{}, len(opts))
	for _, opt := range opts {
		if opt == nil || (reflect.ValueOf(opt).Kind() == reflect.Pointer && reflect.ValueOf(opt).IsNil()) {
			return nil, errors.New("realtime option must not be nil")
		}
		t := reflect.TypeOf(opt)
		if _, ok := seen[t]; ok {
			return nil, fmt.Errorf("duplicate realtime option type %s", t)
		}
		seen[t] = struct{}{}
		if err := opt.Validate(); err != nil {
			return nil, fmt.Errorf("realtime option %T: %w", opt, err)
		}
		switch v := opt.(type) {
		case genai.RealtimeOptionTranscription:
		case *genai.RealtimeOptionAudio:
			if r := v.InputFormat.SampleRate; r != 0 && r != transcriptionRate {
				unsupported = append(unsupported, "input sample rate")
			}
			if v.Voice != "" {
				unsupported = append(unsupported, "voice")
			}
			if v.OutputTranscription {
				unsupported = append(unsupported, "output transcription")
			}
			lang, err := realtime.PrimaryLanguage("OpenAI", v.InputLanguage)
			if err != nil {
				return nil, err
			}
			tr.Language = lang
		case genai.RealtimeTurnDetection:
			if v == genai.RealtimeTurnServer {
				unsupported = append(unsupported, "server turns")
			}
		case *RealtimeOption:
			o = *v
			if o.InputTranscriptionModel != "" {
				unsupported = append(unsupported, "input transcription model")
			}
			if o.MaxOutputTokens != 0 {
				unsupported = append(unsupported, "max output tokens")
			}
			if o.Truncation != nil {
				unsupported = append(unsupported, "truncation")
			}
		default:
			unsupported = append(unsupported, fmt.Sprintf("%T", opt))
		}
	}
	if len(unsupported) != 0 {
		return nil, fmt.Errorf("OpenAI realtime transcription: %w", &base.ErrNotSupported{Options: unsupported})
	}
	if o.Endpoint == "" {
		o.Endpoint = "wss://api.openai.com/v1/realtime"
	}
	if o.PingInterval == 0 {
		o.PingInterval = 30 * time.Second
	}
	if o.PingTimeout == 0 {
		o.PingTimeout = 10 * time.Second
	}
	u, err := url.Parse(o.Endpoint)
	if err != nil {
		return nil, errors.New("invalid OpenAI realtime endpoint")
	}
	u.RawQuery = url.Values{"intent": {"transcription"}}.Encode()
	format := AudioFormat{Type: AudioPCM, Rate: transcriptionRate}
	update, err := json.Marshal(SessionUpdateMessage{Type: EventSessionUpdate, Session: &SessionUpdate{
		Type: SessionTranscription,
		Audio: SessionUpdateAudio{Input: SessionUpdateAudioInput{
			Format:        format,
			Transcription: Nullable[Transcription]{Set: true, Value: &tr},
			// Manual turns: Commit ends the utterance.
			TurnDetection: Nullable[TurnDetection]{Set: true},
		}},
	}})
	if err != nil {
		return nil, err
	}
	ws, err := realtime.Dial(ctx, u.String(), hc, maxMessageBytes, o.PingInterval, o.PingTimeout)
	if err != nil {
		return nil, err
	}
	s := &transcriptionSession{ws: ws}
	if err := s.setup(ctx, update, format); err != nil {
		return nil, errors.Join(err, ws.Close())
	}
	return s, nil
}

// transcriptionSession recognizes one utterance.
type transcriptionSession struct {
	ws   *realtime.Conn
	info genai.RealtimeSessionInfo
	// pending is receiver-owned and holds the untaken events of one server
	// message, so a consumer that stops early loses none.
	pending []genai.RealtimeEvent

	mu        sync.Mutex
	committed bool
}

// Info implements genai.RealtimeSession.
func (s *transcriptionSession) Info() genai.RealtimeSessionInfo { return s.info }

// Close implements genai.RealtimeSession.
func (s *transcriptionSession) Close() error { return s.ws.Close() }

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
		return fmt.Errorf("OpenAI transcription accepts only audio and commits: %w", &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", in)}})
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.committed {
		return errors.New("OpenAI transcription already committed its utterance")
	}
	if len(v.Audio) != 0 {
		if err := s.send(ctx, &ClientMessage{Type: EventAudioAppend, Audio: v.Audio}); err != nil {
			return err
		}
	}
	if v.Commit {
		if err := s.send(ctx, &ClientMessage{Type: EventAudioCommit}); err != nil {
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
			case EventInputTranscriptDelta:
				if m.Delta != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, ItemID: m.ItemID, Content: genai.RealtimeContent{Text: m.Delta}})
				}
			case EventInputTranscriptDone:
				if m.Transcript != "" {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTranscriptInput, ItemID: m.ItemID, Content: genai.RealtimeContent{Text: m.Transcript, Final: true}})
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone, ItemID: m.ItemID})
			case EventInputTranscriptFailed, EventError:
				e := &genai.RealtimeError{Message: "OpenAI transcription failed"}
				if m.Error != nil {
					e.Code = m.Error.Code
					e.EventID = m.Error.EventID
				}
				s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventError, ItemID: m.ItemID, Error: e})
				if m.Type == EventInputTranscriptFailed {
					s.pending = append(s.pending, genai.RealtimeEvent{Kind: genai.RealtimeEventTurnDone, ItemID: m.ItemID})
				}
			default:
				// Acknowledgements, segments and item lifecycle carry no new transcript.
			}
		}
	}
}

// setup requires session.created, sends update, and requires session.updated
// with the requested input format. Unknown messages are ignored within a
// bounded budget.
func (s *transcriptionSession) setup(ctx context.Context, update []byte, format AudioFormat) error {
	created := false
	for range 32 {
		m, err := s.receive(ctx)
		if err != nil {
			return err
		}
		switch m.Type {
		case EventSessionCreated:
			if created || m.Session == nil || m.Session.ID == "" || len(m.Session.ID) > 256 || m.Session.Type != SessionTranscription {
				return errors.New("invalid OpenAI transcription session creation")
			}
			created = true
			s.info.ID = m.Session.ID
			if err := s.ws.Write(ctx, update); err != nil {
				return err
			}
		case EventSessionUpdated:
			if !created || m.Session == nil || m.Session.Type != SessionTranscription {
				return errors.New("OpenAI transcription setup acknowledgement missing")
			}
			if m.Session.Audio.Input.Format != format || m.Session.Audio.Input.TurnDetection != nil {
				return errors.New("OpenAI transcription kept another input format or turn detection")
			}
			s.info.InputFormat = genai.AudioFormat{SampleRate: format.Rate}
			s.info.InputModalities = []genai.Modality{genai.ModalityAudio}
			s.info.TurnDetection = genai.RealtimeTurnManual
			s.info.Capabilities = genai.RealtimeCapabilities{AudioCommit: true}
			return nil
		case EventError:
			e := &genai.RealtimeError{Message: "OpenAI rejected the transcription setup"}
			if m.Error != nil {
				e.Code = m.Error.Code
			}
			return e
		default:
		}
	}
	return errors.New("OpenAI transcription setup exceeded event limit")
}

func (s *transcriptionSession) send(ctx context.Context, m *ClientMessage) error {
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return s.ws.Write(ctx, data)
}

// receive reads one server message; invalid JSON terminates the connection.
func (s *transcriptionSession) receive(ctx context.Context) (ServerMessage, error) {
	var m ServerMessage
	data, err := s.ws.Read(ctx)
	if err != nil {
		return m, err
	}
	if err := internal.UnmarshalJSON(data, &m); err != nil {
		s.ws.Abort(err)
		return ServerMessage{}, err
	}
	return m, nil
}

var _ genai.RealtimeSession = (*transcriptionSession)(nil)
