// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for Gemini voice catalog queries and metadata.
//
// Source: https://github.com/googleapis/go-genai, commit
// 6e70469b445c782e293683af9bbe56ae20c80210, interactions/models/voices
// and interactions/models/operations/listvoices.go.

package gemini

import "time"

// ListVoicesResponse contains custom stored voices, newest first, followed by
// matching prebuilt catalog voices. Listing omits SampleAudio.
type ListVoicesResponse struct {
	NextPageToken string  `json:"next_page_token,omitzero"`
	Voices        []Voice `json:"voices,omitzero"`
}

// Voice is a custom voice or a prebuilt voice from Google's catalog. This
// catalog describes speech synthesis voices; it does not establish that a
// particular realtime model accepts every returned voice.
type Voice struct {
	Accent      string      `json:"accent,omitzero"`
	Context     string      `json:"context,omitzero"`
	Description string      `json:"description,omitzero"`
	DisplayName string      `json:"display_name,omitzero"`
	ExpireTime  time.Time   `json:"expire_time,omitzero"`
	Gender      VoiceGender `json:"gender,omitzero"`
	// ID is a prebuilt speaker name (such as Puck) or a stored custom
	// voice identifier with the voice_ prefix.
	ID string `json:"id,omitzero"`
	// Key identifies a stateless replicated voice; it has the voicekey_ prefix.
	Key          string         `json:"key,omitzero"`
	LanguageCode string         `json:"language_code,omitzero"`
	Model        string         `json:"model,omitzero"`
	Persona      string         `json:"persona,omitzero"`
	Pitch        VoicePitch     `json:"pitch,omitzero"`
	Prompted     *PromptedVoice `json:"prompted,omitzero"`
	RegionCode   string         `json:"region_code,omitzero"`
	SampleAudio  *VoiceAudio    `json:"sample_audio,omitzero"`
	Type         VoiceType      `json:"type"`
	Usage        *VoiceUsage    `json:"usage,omitzero"`
}

// VoiceType identifies how a voice was created.
type VoiceType string

// VoiceType values.
const (
	VoiceTypePrebuilt   VoiceType = "prebuilt"
	VoiceTypePrompted   VoiceType = "prompted"
	VoiceTypeReplicated VoiceType = "replicated"
)

// VoiceGender describes perceived gender presentation in the voice catalog.
type VoiceGender string

// VoiceGender values.
const (
	VoiceGenderFemale  VoiceGender = "female"
	VoiceGenderMale    VoiceGender = "male"
	VoiceGenderNeutral VoiceGender = "neutral"
)

// VoicePitch is the catalog's vocal pitch classification.
type VoicePitch string

// VoicePitch values.
const (
	VoicePitchHigh   VoicePitch = "high"
	VoicePitchLow    VoicePitch = "low"
	VoicePitchMedium VoicePitch = "medium"
)

// PromptedVoice describes a designed or remixed voice.
type PromptedVoice struct {
	BaseVoice string `json:"base_voice,omitzero"`
	Input     string `json:"input"`
}

// VoiceAudio is the audio payload of a voice preview.
type VoiceAudio struct {
	Data     []byte `json:"data"`
	MIMEType string `json:"mime_type"`
}

// VoiceUsage is the token usage returned for voice creation.
type VoiceUsage struct {
	CachedTokensByModality  []VoiceModalityTokens `json:"cached_tokens_by_modality,omitzero"`
	GroundingToolCount      []VoiceGroundingCount `json:"grounding_tool_count,omitzero"`
	InitialInputTokens      int64                 `json:"initial_input_tokens,omitzero"`
	InputTokensByModality   []VoiceModalityTokens `json:"input_tokens_by_modality,omitzero"`
	OutputTokensByModality  []VoiceModalityTokens `json:"output_tokens_by_modality,omitzero"`
	ToolUseTokensByModality []VoiceModalityTokens `json:"tool_use_tokens_by_modality,omitzero"`
	TotalCachedTokens       int64                 `json:"total_cached_tokens,omitzero"`
	TotalInputTokens        int64                 `json:"total_input_tokens,omitzero"`
	TotalOutputTokens       int64                 `json:"total_output_tokens,omitzero"`
	TotalThoughtTokens      int64                 `json:"total_thought_tokens,omitzero"`
	TotalTokens             int64                 `json:"total_tokens,omitzero"`
	TotalToolUseTokens      int64                 `json:"total_tool_use_tokens,omitzero"`
}

// VoiceModalityTokens breaks down token usage by response modality.
type VoiceModalityTokens struct {
	Modality ResponseModality `json:"modality,omitzero"`
	Tokens   int64            `json:"tokens,omitzero"`
}

// ResponseModality identifies the lowercase response modalities used by the
// Interactions API, distinct from generateContent's uppercase Modality values.
type ResponseModality string

// ResponseModality values.
const (
	ResponseModalityAudio    ResponseModality = "audio"
	ResponseModalityDocument ResponseModality = "document"
	ResponseModalityImage    ResponseModality = "image"
	ResponseModalityText     ResponseModality = "text"
	ResponseModalityVideo    ResponseModality = "video"
)

// VoiceGroundingCount counts calls to a grounding tool.
type VoiceGroundingCount struct {
	Count int64                  `json:"count,omitzero"`
	Type  GroundingToolCountType `json:"type,omitzero"`
}

// GroundingToolCountType identifies the grounding tool counted in voice usage.
type GroundingToolCountType string

// GroundingToolCountType values.
const (
	GroundingToolCountTypeGoogleMaps   GroundingToolCountType = "google_maps"
	GroundingToolCountTypeGoogleSearch GroundingToolCountType = "google_search"
	GroundingToolCountTypeRetrieval    GroundingToolCountType = "retrieval"
)
