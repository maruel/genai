// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Mistral realtime transcription WebSocket API.
//
// Sourced from github.com/mistralai/client-python at commit
// d3da3bd3c63d0502449e1e765bc2d3c16d0d2f1d: src/mistralai/client/models/realtimetranscription*.py,
// transcriptionstream*.py, audioformat.py, audioencoding.py and turndetectionconfig.py, and the message
// flow in src/mistralai/extra/realtime/connection.py.

package mistral

import "encoding/json"

// RealtimeClientEvent is a client message type.
type RealtimeClientEvent string

// Client message types.
const (
	RealtimeInputAudioAppend RealtimeClientEvent = "input_audio.append"
	RealtimeInputAudioEnd    RealtimeClientEvent = "input_audio.end"
	RealtimeInputAudioFlush  RealtimeClientEvent = "input_audio.flush"
	RealtimeSessionUpdate    RealtimeClientEvent = "session.update"
)

// RealtimeServerEvent is a server message type.
type RealtimeServerEvent string

// Server message types.
const (
	RealtimeError                  RealtimeServerEvent = "error"
	RealtimeSessionCreated         RealtimeServerEvent = "session.created"
	RealtimeSessionUpdated         RealtimeServerEvent = "session.updated"
	RealtimeTranscriptionDone      RealtimeServerEvent = "transcription.done"
	RealtimeTranscriptionLanguage  RealtimeServerEvent = "transcription.language"
	RealtimeTranscriptionSegment   RealtimeServerEvent = "transcription.segment"
	RealtimeTranscriptionTextDelta RealtimeServerEvent = "transcription.text.delta"
	RealtimeTurnEagerEnd           RealtimeServerEvent = "turn.eager_end"
	RealtimeTurnEnd                RealtimeServerEvent = "turn.end"
	RealtimeTurnResume             RealtimeServerEvent = "turn.resume"
	RealtimeTurnStart              RealtimeServerEvent = "turn.start"
)

// RealtimeAudioEncoding is an input audio encoding.
type RealtimeAudioEncoding string

// Input audio encodings.
const (
	RealtimeAudioALaw   RealtimeAudioEncoding = "pcm_alaw"
	RealtimeAudioF16LE  RealtimeAudioEncoding = "pcm_f16le"
	RealtimeAudioF32LE  RealtimeAudioEncoding = "pcm_f32le"
	RealtimeAudioMuLaw  RealtimeAudioEncoding = "pcm_mulaw"
	RealtimeAudioPCMS16 RealtimeAudioEncoding = "pcm_s16le"
	RealtimeAudioPCMS32 RealtimeAudioEncoding = "pcm_s32le"
)

// RealtimeAudioFormat is the input audio format.
type RealtimeAudioFormat struct {
	Encoding   RealtimeAudioEncoding `json:"encoding"`
	SampleRate int                   `json:"sample_rate"`
}

// RealtimeTurnDetection configures server turn detection.
type RealtimeTurnDetection struct {
	StartThreshold    float64 `json:"start_threshold,omitzero"`
	EagerEndThreshold float64 `json:"eager_end_threshold,omitzero"`
	EndThreshold      float64 `json:"end_threshold,omitzero"`
	MinSilenceMS      int     `json:"min_silence_ms,omitzero"`
	EndTimeoutMS      int     `json:"end_timeout_ms,omitzero"`
}

// RealtimeSession is the session state the server reports.
type RealtimeSession struct {
	RequestID              string              `json:"request_id"`
	Model                  string              `json:"model"`
	AudioFormat            RealtimeAudioFormat `json:"audio_format"`
	TargetStreamingDelayMS int                 `json:"target_streaming_delay_ms,omitzero"`
	Diarize                bool                `json:"diarize,omitzero"`
	// Verbatim is sent by the service but absent from the SDK commit above.
	Verbatim          bool                   `json:"verbatim,omitzero"`
	ContextBias       []string               `json:"context_bias,omitzero"`
	Language          string                 `json:"language,omitzero"`
	SpeakerReferences []string               `json:"speaker_references,omitzero"`
	TurnDetection     *RealtimeTurnDetection `json:"turn_detection,omitzero"`
}

// RealtimeSessionConfig is the session.update payload. Omitted fields keep
// their current value.
type RealtimeSessionConfig struct {
	AudioFormat            *RealtimeAudioFormat   `json:"audio_format,omitzero"`
	TargetStreamingDelayMS int                    `json:"target_streaming_delay_ms,omitzero"`
	Diarize                bool                   `json:"diarize,omitzero"`
	ContextBias            []string               `json:"context_bias,omitzero"`
	Language               string                 `json:"language,omitzero"`
	SpeakerReferences      map[string]string      `json:"speaker_references,omitzero"`
	TurnDetection          *RealtimeTurnDetection `json:"turn_detection,omitzero"`
}

// RealtimeClientMessage is one client message. Audio is base64 encoded on the
// wire.
type RealtimeClientMessage struct {
	Type    RealtimeClientEvent    `json:"type"`
	Session *RealtimeSessionConfig `json:"session,omitzero"`
	Audio   []byte                 `json:"audio,omitzero"`
}

// RealtimeSegment is a timed transcript segment.
type RealtimeSegment struct {
	Type      string  `json:"type,omitzero"`
	Text      string  `json:"text"`
	Start     float64 `json:"start"`
	End       float64 `json:"end"`
	Score     float64 `json:"score,omitzero"`
	SpeakerID string  `json:"speaker_id,omitzero"`
}

// RealtimeErrorDetail is a server error.
type RealtimeErrorDetail struct {
	// Message is a string or an object.
	Message json.RawMessage `json:"message"`
	Code    int             `json:"code"`
}

// Text returns the message, unquoted when it is a JSON string.
func (e *RealtimeErrorDetail) Text() string {
	var s string
	if json.Unmarshal(e.Message, &s) == nil {
		return s
	}
	return string(e.Message)
}

// RealtimeServerMessage is one server message; Type selects the fields set.
type RealtimeServerMessage struct {
	Type    RealtimeServerEvent  `json:"type"`
	Session *RealtimeSession     `json:"session,omitzero"`
	Error   *RealtimeErrorDetail `json:"error,omitzero"`

	// transcription.text.delta, transcription.segment and transcription.done.
	Text string `json:"text,omitzero"`
	// transcription.segment.
	Start     float64 `json:"start,omitzero"`
	End       float64 `json:"end,omitzero"`
	SpeakerID string  `json:"speaker_id,omitzero"`
	// transcription.language.
	AudioLanguage string `json:"audio_language,omitzero"`
	// transcription.done.
	Model    string            `json:"model,omitzero"`
	Usage    *Usage            `json:"usage,omitzero"`
	Language string            `json:"language,omitzero"`
	Segments []RealtimeSegment `json:"segments,omitzero"`
}
