// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Speech generation and server property wire types.
// Sourced from ggml-org/llama.cpp v1.0.0, commit
// b1b770d90d94cad9e9ea5b993bdbc383a522fb28, tools/server/server-chat.cpp
// (convert_speech_to_tts), tools/server/server-context.cpp and
// tools/server/server-schema.cpp.

package llamacpp

import (
	"encoding/json"
	"errors"
	"math"
)

// SpeechFormat selects the audio container returned by /v1/audio/speech.
type SpeechFormat string

// Supported speech output formats.
const (
	// SpeechPCM is headerless little-endian signed 16-bit mono PCM.
	SpeechPCM SpeechFormat = "pcm"
	SpeechWAV SpeechFormat = "wav"
)

// SpeechRequest generates speech with the server's loaded audio generation
// model through the OpenAI-compatible /v1/audio/speech endpoint, which forwards
// the llama.cpp /tts options. Output is at the model's native sample rate, not
// resampled to 24 kHz. WAV is the default. Stream requests deliver audio as it
// becomes available.
type SpeechRequest struct {
	Input string `json:"input"`
	// Voice is one of the voices listed by GET /props. OpenAI voice names are
	// not mapped.
	Voice          string       `json:"voice,omitzero"`
	Lang           string       `json:"lang,omitzero"`
	ResponseFormat SpeechFormat `json:"response_format,omitzero"`
	Stream         bool         `json:"stream,omitzero"`
	// SpeakerRef contains encoded reference audio (for example WAV), serialized as base64.
	// It requires a model with a voice encoder and cannot be combined with Voice.
	SpeakerRef []byte `json:"speaker_ref_b64,omitzero"`
	// TopK and TopP apply to the backbone sampler and the acoustic code
	// predictor. Zero keeps the server and model defaults.
	TopK int32   `json:"top_k,omitzero"`
	TopP float64 `json:"top_p,omitzero"`
	// Temperature applies to the backbone sampler and the audio frames, for
	// models that support it. Nil keeps the server and model defaults; 0 is greedy.
	Temperature *float64 `json:"temperature,omitzero"`
	// Seed selects the random seed; nil uses the server default and -1 requests randomness.
	Seed *int64 `json:"seed,omitzero"`
	// NPredict limits audio frames. Zero or -1 inherits the server limit;
	// a nonpositive server limit falls back to 512 frames.
	NPredict int32 `json:"n_predict,omitzero"`
	// RepeatPenalty overrides the server default of 1.05; 1 disables the penalty.
	// It must be positive.
	RepeatPenalty *float64 `json:"repeat_penalty,omitzero"`
}

// Validate implements genai.Validatable.
func (s *SpeechRequest) Validate() error {
	if s.Input == "" {
		return errors.New("input is required")
	}
	if s.ResponseFormat != "" && s.ResponseFormat != SpeechPCM && s.ResponseFormat != SpeechWAV {
		return errors.New("response_format must be wav or pcm")
	}
	if s.Voice != "" && len(s.SpeakerRef) != 0 {
		return errors.New("voice and speaker reference cannot be combined")
	}
	if s.TopK < 0 || s.TopP < 0 || s.TopP > 1 || math.IsNaN(s.TopP) {
		return errors.New("top_k must be non-negative and top_p must be between 0 and 1")
	}
	if s.Temperature != nil && (*s.Temperature < 0 || math.IsNaN(*s.Temperature)) {
		return errors.New("temperature must be non-negative")
	}
	if s.NPredict < -1 {
		return errors.New("n_predict must be at least -1")
	}
	if s.RepeatPenalty != nil && (*s.RepeatPenalty <= 0 || math.IsNaN(*s.RepeatPenalty) || math.IsInf(*s.RepeatPenalty, 0)) {
		return errors.New("repeat_penalty must be positive and finite")
	}
	return nil
}

// PropsResponse describes the currently loaded server and its available voices.
// Voices is absent for models without predefined voices.
type PropsResponse struct {
	DefaultGenerationSettings DefaultGenerationSettings `json:"default_generation_settings"`
	TotalSlots                int64                     `json:"total_slots"`
	ModelAlias                string                    `json:"model_alias"`
	ModelFtype                string                    `json:"model_ftype"`
	ModelPath                 string                    `json:"model_path"`
	Modalities                ModelModalities           `json:"modalities"`
	MediaMarker               string                    `json:"media_marker"`
	EndpointSlots             bool                      `json:"endpoint_slots"`
	EndpointProps             bool                      `json:"endpoint_props"`
	EndpointMetrics           bool                      `json:"endpoint_metrics"`
	UI                        bool                      `json:"ui"`
	// UISettings is arbitrary server UI configuration.
	UISettings          json.RawMessage  `json:"ui_settings"`
	ChatTemplate        string           `json:"chat_template"`
	ChatTemplateToolUse string           `json:"chat_template_tool_use"`
	ChatTemplateCaps    ChatTemplateCaps `json:"chat_template_caps"`
	BOSToken            string           `json:"bos_token"`
	EOSToken            string           `json:"eos_token"`
	BuildInfo           string           `json:"build_info"`
	IsSleeping          bool             `json:"is_sleeping"`
	CORSProxyEnabled    bool             `json:"cors_proxy_enabled"`
	ConnectCode         string           `json:"connect_code"`
	Voices              []string         `json:"voices"`
}

// DefaultGenerationSettings contains the server's default sampling settings and context size.
type DefaultGenerationSettings struct {
	Params GenerationSettings `json:"params"`
	NCtx   int64              `json:"n_ctx"`
}

// ModelModalities lists supported media inputs.
type ModelModalities struct {
	Vision bool `json:"vision"`
	Video  bool `json:"video"`
	Audio  bool `json:"audio"`
}

// ChatTemplateCaps describes the capabilities detected in the model's chat template.
type ChatTemplateCaps struct {
	SupportsStringContent     bool     `json:"supports_string_content"`
	SupportsTypedContent      bool     `json:"supports_typed_content"`
	SupportsTools             bool     `json:"supports_tools"`
	SupportsToolCalls         bool     `json:"supports_tool_calls"`
	SupportsParallelToolCalls bool     `json:"supports_parallel_tool_calls"`
	SupportsSystemRole        bool     `json:"supports_system_role"`
	SupportsPreserveReasoning bool     `json:"supports_preserve_reasoning"`
	SupportsReasoningEffort   bool     `json:"supports_reasoning_effort"`
	SupportsObjectArguments   bool     `json:"supports_object_arguments"`
	ReasoningEffortLevels     []string `json:"reasoning_effort_levels"`
}
