// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for Gemini Interactions speech, sourced from github.com/googleapis/go-genai
// commit 087281541b864840a1ff51a14d16eea23f0d8d83, interactions/models/interactions.
// This subset supports text/audio inputs, speech generation and transcription;
// agent, tool, image, video and conversation-message unions are not implemented.

package gemini

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
)

// InteractionRequest creates a model interaction with text/audio content blocks.
// Stream is controlled by the raw operation, which does not mutate the request.
// Store is a pointer because omitting it uses the server's default retention policy.
type InteractionRequest struct {
	Model                 string                       `json:"model"`
	Input                 []InteractionContent         `json:"input,omitzero"`
	ResponseFormat        *InteractionResponseFormat   `json:"response_format,omitzero"`
	GenerationConfig      *InteractionGenerationConfig `json:"generation_config,omitzero"`
	Store                 *bool                        `json:"store,omitzero"`
	Stream                bool                         `json:"stream,omitzero"`
	SystemInstruction     *string                      `json:"system_instruction,omitzero"`
	PreviousInteractionID string                       `json:"previous_interaction_id,omitzero"`
	Labels                map[string]string            `json:"labels,omitzero"`
	// MaxResponseBytes is a Go-only batch-response bound. Zero uses 1 MiB; raise
	// it explicitly for base64 batch audio. Streaming events remain limited to 1 MiB.
	MaxResponseBytes int64 `json:"-"`
}

// Validate checks the supported text/audio request shape. Provider-specific model,
// voice, codec and transcription-mode constraints remain the server's authority.
func (r *InteractionRequest) Validate() error {
	if r.Model == "" || len(r.Input) == 0 {
		return errors.New("interaction model and input are required")
	}
	if r.MaxResponseBytes < 0 {
		return errors.New("interaction response limit must be positive")
	}
	for i := range r.Input {
		c := &r.Input[i]
		switch c.Type {
		case InteractionText:
			if c.Source != nil || c.Data != "" || c.URI != "" {
				return errors.New("text content cannot carry audio")
			}
		case InteractionAudio:
			n := 0
			if c.Source != nil {
				n++
			}
			if c.Data != "" {
				n++
			}
			if c.URI != "" {
				n++
			}
			if n != 1 || c.MIMEType == "" {
				return errors.New("audio requires MIME type and exactly one of Source, Data or URI")
			}
		default:
			return fmt.Errorf("unsupported interaction input type %q", c.Type)
		}
	}
	return nil
}

// InteractionContentType identifies a supported content block or delta.
type InteractionContentType string

// Native wire values.
const (
	InteractionAudio               InteractionContentType = "audio"
	InteractionText                InteractionContentType = "text"
	InteractionTextAnnotationDelta InteractionContentType = "text_annotation_delta"
)

// InteractionContent is a text/audio content block or streamed content delta.
// Data is base64-encoded audio. Source is a Go-only alternative, mutually exclusive with Data and URI:
// an encoded audio file, rewound and consumed synchronously by the raw call.
// Its position changes; the caller retains ownership and must close it. HTTP
// retries use independent private spooled readers and never read Source again.
// Total encoded audio read from Sources is limited to 14 MB; JSON to 20 MB.
type InteractionContent struct {
	Type        InteractionContentType  `json:"type"`
	Text        string                  `json:"text,omitzero"`
	Data        string                  `json:"data,omitzero"`
	URI         string                  `json:"uri,omitzero"`
	MIMEType    string                  `json:"mime_type,omitzero"`
	SampleRate  int                     `json:"sample_rate,omitzero"`
	Channels    int                     `json:"channels,omitzero"`
	Annotations []InteractionAnnotation `json:"annotations,omitzero"`
	Source      io.ReadSeeker           `json:"-"`
}

// InteractionAnnotationType discriminates supported speech annotations.
type InteractionAnnotationType string

// Native wire values.
const (
	InteractionSpeechMetadata InteractionAnnotationType = "speech_metadata"
	InteractionWordInfo       InteractionAnnotationType = "word_info"
)

// InteractionAnnotation carries word timing/diarization or speech style metadata.
// Indexes count UTF-8 bytes; time offsets retain the provider's duration strings
// (for example "0.100s"). Pointer indexes preserve a present zero.
type InteractionAnnotation struct {
	Type        InteractionAnnotationType `json:"type"`
	Text        string                    `json:"text,omitzero"`
	Speaker     string                    `json:"speaker,omitzero"`
	StartIndex  *int                      `json:"start_index,omitzero"`
	EndIndex    *int                      `json:"end_index,omitzero"`
	StartOffset string                    `json:"start_offset,omitzero"`
	EndOffset   string                    `json:"end_offset,omitzero"`
	Style       string                    `json:"style,omitzero"`
}

// InteractionResponseFormat selects a single text or audio response format.
// Audio MIME type, sample rate, bit rate and delivery are provider/model dependent.
// Schema is a JSON Schema object used with text/application/json output.
// The provider's array-of-formats alternative is not implemented.
type InteractionResponseFormat struct {
	Type       InteractionContentType   `json:"type"`
	MIMEType   string                   `json:"mime_type,omitzero"`
	SampleRate int                      `json:"sample_rate,omitzero"`
	BitRate    int                      `json:"bit_rate,omitzero"`
	Delivery   InteractionAudioDelivery `json:"delivery,omitzero"`
	Schema     json.RawMessage          `json:"schema,omitzero"`
}

// InteractionAudioDelivery controls inline audio versus an output URI.
type InteractionAudioDelivery string

// Native wire values.
const (
	InteractionAudioInline InteractionAudioDelivery = "inline"
	InteractionAudioURI    InteractionAudioDelivery = "uri"
)

// InteractionGenerationConfig configures speech generation or transcription.
// SpeechConfig uses the native array form; the legacy speakers-object form is
// not implemented. Deprecated sampling settings and unrelated tools are omitted.
type InteractionGenerationConfig struct {
	SpeechConfig        []InteractionSpeechConfig       `json:"speech_config,omitzero"`
	TranscriptionConfig *InteractionTranscriptionConfig `json:"transcription_config,omitzero"`
	MaxOutputTokens     int                             `json:"max_output_tokens,omitzero"`
	Seed                *int                            `json:"seed,omitzero"`
	StopSequences       []string                        `json:"stop_sequences,omitzero"`
}

// InteractionSpeechConfig selects a voice and optional named speaker/language.
type InteractionSpeechConfig struct {
	Voice    string `json:"voice,omitzero"`
	Speaker  string `json:"speaker,omitzero"`
	Language string `json:"language,omitzero"`
}

// InteractionTranscriptionConfig sets language hints, vocabulary and recognition
// mode. CustomVocabulary cannot be combined with diarization or word timestamps.
type InteractionTranscriptionConfig struct {
	LanguageCodes    []string                      `json:"language_codes,omitzero"`
	CustomVocabulary []string                      `json:"custom_vocabulary,omitzero"`
	Mode             *InteractionTranscriptionMode `json:"mode,omitzero"`
}

// InteractionTranscriptionModeType identifies the native recognition mode.
type InteractionTranscriptionModeType string

// Native wire values.
const (
	InteractionTranscriptionSmart    InteractionTranscriptionModeType = "smart"
	InteractionTranscriptionVerbatim InteractionTranscriptionModeType = "verbatim"
)

// InteractionTranscriptionMode uses the native object form of mode. Verbatim
// supports "speaker" diarization and "word" timestamp granularities; smart can
// supply SourceText for voice-driven editing. Legacy string modes are omitted.
type InteractionTranscriptionMode struct {
	Type                   InteractionTranscriptionModeType `json:"type"`
	DiarizationMode        string                           `json:"diarization_mode,omitzero"`
	TimestampGranularities []string                         `json:"timestamp_granularities,omitzero"`
	SourceText             *string                          `json:"source_text,omitzero"`
}

// InteractionState is the provider's interaction state.
type InteractionState string

// Native wire values.
const (
	InteractionBudgetExceeded       InteractionState = "budget_exceeded"
	InteractionCancelled            InteractionState = "cancelled"
	InteractionCompleted            InteractionState = "completed"
	InteractionFailed               InteractionState = "failed"
	InteractionInProgress           InteractionState = "in_progress"
	InteractionIncomplete           InteractionState = "incomplete"
	InteractionQueued               InteractionState = "queued"
	InteractionRequiresAction       InteractionState = "requires_action"
	InteractionRequiresContinuation InteractionState = "requires_continuation"
)

// InteractionResponse is a speech-scoped interaction resource. Lifecycle events
// contain partial resources and may omit fields present in batch responses.
// Text is in Steps, not the official SDK's derived output_text convenience field.
type InteractionResponse struct {
	ID                    string                       `json:"id,omitzero"`
	Model                 string                       `json:"model,omitzero"`
	Status                InteractionState             `json:"status,omitzero"`
	Created               string                       `json:"created,omitzero"`
	Updated               string                       `json:"updated,omitzero"`
	Errors                []InteractionError           `json:"errors,omitzero"`
	Steps                 []InteractionStep            `json:"steps,omitzero"`
	Usage                 *InteractionUsage            `json:"usage,omitzero"`
	ContinuationToken     string                       `json:"continuation_token,omitzero"`
	PreviousInteractionID string                       `json:"previous_interaction_id,omitzero"`
	GenerationConfig      *InteractionGenerationConfig `json:"generation_config,omitzero"`
	ResponseFormat        *InteractionResponseFormat   `json:"response_format,omitzero"`
	Labels                map[string]string            `json:"labels,omitzero"`
}

func (s *InteractionResponse) checkCompleted() error {
	if len(s.Errors) != 0 {
		errs := make([]error, 0, len(s.Errors))
		for i := range s.Errors {
			errs = append(errs, &s.Errors[i])
		}
		return errors.Join(errs...)
	}
	if s.Status != InteractionCompleted {
		return fmt.Errorf("speech interaction ended with status %q", s.Status)
	}
	return nil
}

// InteractionStepType identifies supported speech step payloads.
type InteractionStepType string

// Native wire values.
const (
	InteractionModelOutput InteractionStepType = "model_output"
	InteractionUserInput   InteractionStepType = "user_input"
)

// InteractionStep carries content from a model output or user input step.
// Error preserves the native google.rpc.Status error reported on model output.
type InteractionStep struct {
	ID      string                `json:"id,omitzero"`
	Type    InteractionStepType   `json:"type"`
	Content []InteractionContent  `json:"content,omitzero"`
	Error   *InteractionStepError `json:"error,omitzero"`
}

// InteractionStepError is a native google.rpc.Status on a model-output step.
type InteractionStepError struct {
	Code    int               `json:"code,omitzero"`
	Message string            `json:"message,omitzero"`
	Details []json.RawMessage `json:"details,omitzero"`
}

// InteractionUsage contains native token accounting for speech interactions.
type InteractionUsage struct {
	TotalTokens             int                         `json:"total_tokens,omitzero"`
	TotalInputTokens        int                         `json:"total_input_tokens,omitzero"`
	TotalOutputTokens       int                         `json:"total_output_tokens,omitzero"`
	TotalCachedTokens       int                         `json:"total_cached_tokens,omitzero"`
	TotalThoughtTokens      int                         `json:"total_thought_tokens,omitzero"`
	TotalToolUseTokens      int                         `json:"total_tool_use_tokens,omitzero"`
	InitialInputTokens      int                         `json:"initial_input_tokens,omitzero"`
	InputTokensByModality   []InteractionModalityTokens `json:"input_tokens_by_modality,omitzero"`
	OutputTokensByModality  []InteractionModalityTokens `json:"output_tokens_by_modality,omitzero"`
	CachedTokensByModality  []InteractionModalityTokens `json:"cached_tokens_by_modality,omitzero"`
	ToolUseTokensByModality []InteractionModalityTokens `json:"tool_use_tokens_by_modality,omitzero"`
}

// InteractionModality is the Interactions API's lowercase modality vocabulary,
// distinct from generateContent's uppercase Modality values.
type InteractionModality string

// Speech modality wire values.
const (
	InteractionModalityAudio InteractionModality = "audio"
	InteractionModalityText  InteractionModality = "text"
)

// InteractionModalityTokens is a native modality/token-count pair.
type InteractionModalityTokens struct {
	Modality InteractionModality `json:"modality"`
	Tokens   int                 `json:"tokens"`
}

// InteractionEventType discriminates a native Interactions SSE event.
type InteractionEventType string

// Native wire values.
const (
	InteractionEventCompleted InteractionEventType = "interaction.completed"
	InteractionEventCreated   InteractionEventType = "interaction.created"
	InteractionEventDelta     InteractionEventType = "step.delta"
	InteractionEventError     InteractionEventType = "error"
	InteractionEventStart     InteractionEventType = "step.start"
	InteractionEventStatus    InteractionEventType = "interaction.status_update"
	InteractionEventStop      InteractionEventType = "step.stop"
)

// InteractionEvent is a speech-scoped SSE event. Fields are selected by EventType:
// lifecycle uses Interaction; step.start uses Step; step.delta uses Delta; step.stop
// uses StepUsage; status updates use Status, InteractionID and Usage; errors use Error.
// Index preserves the step index and EventID the provider's stream resume token.
type InteractionEvent struct {
	EventType         InteractionEventType      `json:"event_type"`
	EventID           string                    `json:"event_id,omitzero"`
	InteractionID     string                    `json:"interaction_id,omitzero"`
	Index             *int                      `json:"index,omitzero"`
	Delta             InteractionContent        `json:"delta,omitzero"`
	Step              InteractionStep           `json:"step,omitzero"`
	Interaction       InteractionResponse       `json:"interaction,omitzero"`
	Error             *InteractionError         `json:"error,omitzero"`
	Status            InteractionState          `json:"status,omitzero"`
	Usage             *InteractionUsage         `json:"usage,omitzero"`
	StepUsage         *InteractionUsage         `json:"step_usage,omitzero"`
	Metadata          *InteractionEventMetadata `json:"metadata,omitzero"`
	ContinuationToken string                    `json:"continuation_token,omitzero"`
}

// InteractionEventMetadata accompanies deltas with cumulative token accounting.
type InteractionEventMetadata struct {
	TotalUsage *InteractionUsage `json:"total_usage,omitzero"`
}

// InteractionError is an error reported within an Interactions speech response.
// HTTP failures retain the provider's ErrorResponse type.
type InteractionError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Error implements error.
func (e *InteractionError) Error() string { return "gemini interaction " + e.Code + ": " + e.Message }
