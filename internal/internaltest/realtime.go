// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Realtime WebSocket recording for provider tests.

package internaltest

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/coder/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/websocketrecord"
)

// RealtimeRecorder records or replays each realtime WebSocket session opened
// through Provider.
//
// One fixture is created per ConnectRealtime call. Dir must contain the
// fixtures named Realtime-1.ndjson, Realtime-2.ndjson, and so on. Mutation adds
// the recording proxy endpoint to opts before the provider connects.
//
// When Updating is true, existing fixtures are replaced, and a session is
// recorded even where no fixture exists. Otherwise a missing fixture fails
// unless RECORD is set. Set it from [UpdatingRecordings] so that cassettes
// follow RECORD and not -update-scoreboard, which only refreshes the scoreboard.
// A failed session keeps its recording as Realtime-N.failed.ndjson. Audio
// payloads are replaced with a two-byte marker so fixtures stay small; event
// structure and transcripts are preserved.
type RealtimeRecorder struct {
	genai.Provider

	T        testing.TB
	Target   string
	Dir      string
	Updating bool
	Mutation func(opts []genai.RealtimeOption, endpoint string) []genai.RealtimeOption

	mu    sync.Mutex
	calls int
}

// ConnectRealtime implements genai.Provider.
func (r *RealtimeRecorder) ConnectRealtime(ctx context.Context, opts ...genai.RealtimeOption) (genai.RealtimeSession, error) {
	r.mu.Lock()
	r.calls++
	call := r.calls
	r.mu.Unlock()
	path := filepath.Join(r.Dir, fmt.Sprintf("Realtime-%d", call))
	if r.Updating {
		if err := os.Remove(path + ".ndjson"); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return nil, fmt.Errorf("remove realtime recording: %w", err)
		}
	} else if _, err := os.Stat(path + ".ndjson"); errors.Is(err, fs.ErrNotExist) {
		// A realtime fixture is numbered by position, so a missing number means the
		// probe flow opens one more session than when it was recorded. Failing says
		// so; skipping would make a gap look like a confirmation. Recording takes a
		// deliberate RECORD so a flow change cannot turn into a live call.
		if os.Getenv("RECORD") == "" {
			r.T.Fatalf("missing realtime recording %s; record it with RECORD=failure_only scoped to this model", path+".ndjson")
		}
	} else if err != nil {
		return nil, fmt.Errorf("stat realtime recording: %w", err)
	}
	markRecording(r.T, path+".ndjson")
	rec, err := websocketrecord.New(path)
	if err != nil {
		return nil, err
	}
	rec.Normalize = MaskRealtimeAudio
	r.T.Cleanup(func() {
		err := rec.Close()
		if r.T.Failed() && !rec.IsReplay() {
			err = errors.Join(err, retainFailedRecording(path+".ndjson"))
		}
		if err != nil {
			r.T.Error(err)
		}
	})
	if err := rec.Start(r.Target); err != nil {
		return nil, errors.Join(err, rec.Discard())
	}
	if r.Mutation == nil {
		return nil, errors.Join(errors.New("realtime recorder requires an endpoint mutation"), rec.Discard())
	}
	opts = r.Mutation(opts, rec.WSURL())
	s, err := r.Provider.ConnectRealtime(ctx, opts...)
	if err != nil {
		if !r.Updating {
			return nil, err
		}
		// A refused session is a result. Keep the exchange, and keep the call number
		// that names it, so replay reproduces the refusal instead of asking again.
		return nil, errors.Join(err, rec.Close())
	}
	if rec.IsReplay() {
		return &replaySession{RealtimeSession: s}, nil
	}
	return s, nil
}

// Unwrap returns the wrapped provider.
func (r *RealtimeRecorder) Unwrap() genai.Provider {
	return r.Provider
}

// replaySession marks a session that replays a recording. Consumers skip the
// playback-speed pacing that only a live server needs.
type replaySession struct {
	genai.RealtimeSession
}

// IsReplay reports that the session replays a recording.
func (r *replaySession) IsReplay() bool { return true }

// realtimeAudioMarker is base64 for two zero PCM16 bytes.
const realtimeAudioMarker = "AAA="

// MaskRealtimeAudio replaces realtime PCM payloads with a two-byte marker.
//
// It preserves event types and ordering while removing the large base64 audio
// blobs from recordings. Text, transcripts, tool calls and metadata are
// unchanged. Unrecognized payloads are returned unchanged.
func MaskRealtimeAudio(typ websocket.MessageType, data []byte) []byte {
	v, err := decodeJSONObject(data)
	if err != nil {
		return data
	}
	if !maskRealtimeAudio(v) {
		return data
	}
	out, err := json.Marshal(v)
	if err != nil {
		return data
	}
	return out
}

func decodeJSONObject(data []byte) (any, error) {
	d := json.NewDecoder(bytes.NewReader(data))
	d.UseNumber()
	var v any
	if err := d.Decode(&v); err != nil {
		return nil, err
	}
	return v, nil
}

func maskRealtimeAudio(v any) bool {
	changed := false
	switch x := v.(type) {
	case map[string]any:
		event := ""
		if s, ok := x["type"].(string); ok {
			event = s
		}
		mimeType, _ := x["mimeType"].(string)
		for k, value := range x {
			s, isString := value.(string)
			if isString && s != realtimeAudioMarker {
				switch {
				case k == "audio":
					x[k] = realtimeAudioMarker
					changed = true
				case k == "delta" && strings.HasSuffix(event, "audio.delta"):
					x[k] = realtimeAudioMarker
					changed = true
				case k == "data" && strings.HasPrefix(mimeType, "audio/"):
					x[k] = realtimeAudioMarker
					changed = true
				default:
					changed = maskRealtimeAudio(value) || changed
				}
				continue
			}
			changed = maskRealtimeAudio(value) || changed
		}
	case []any:
		for _, value := range x {
			changed = maskRealtimeAudio(value) || changed
		}
	}
	return changed
}

// UpdatingRecordings reports whether the run re-records cassettes.
//
// Only RECORD=all replaces existing fixtures; RECORD=failure_only records the
// missing ones and replays the rest. Both leave -update-scoreboard as a pure
// scoreboard refresh.
func UpdatingRecordings() bool {
	return os.Getenv("RECORD") == "all"
}

func retainFailedRecording(fixture string) error {
	failed := failedRecordingPath(fixture)
	if err := os.Remove(failed); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("remove failed recording: %w", err)
	}
	if err := os.Rename(fixture, failed); err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("retain failed recording: %w", err)
	}
	return nil
}
