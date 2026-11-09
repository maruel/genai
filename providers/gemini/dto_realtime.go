// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Google Gemini Live API.
//
// API reference: https://ai.google.dev/api/live
//
// Source: https://github.com/googleapis/go-genai, commit
// 9cf04deac32f2418781170b3cd41c09abf060564, files live.go, types.go and
// live_converters.go (Gemini Developer API conversion).
//
// This file models the BidiGenerateContent protocol as documented for the Gemini
// Developer API, including inbound fields the session implementation in
// realtime.go does not translate yet, so strict decoding accepts them. Vertex
// AI-only configuration is omitted. The runtime support is a subset of what is
// described here.

package gemini

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/maruel/genai"
)

// LiveClientMessage is documented at https://ai.google.dev/api/live#BidiGenerateContentClientMessage
type LiveClientMessage struct {
	Setup         *LiveSetup         `json:"setup,omitempty"`
	ClientContent *LiveClientContent `json:"clientContent,omitempty"`
	RealtimeInput *LiveRealtimeInput `json:"realtimeInput,omitempty"`
	ToolResponse  *LiveToolResponse  `json:"toolResponse,omitempty"`
	ContextUpdate LiveContextUpdate  `json:"contextUpdate,omitzero"`
}

// LiveContextUpdate updates the model context of an open session.
//
// Preview: the SDK exposes it through SendContextUpdate. Updates are processed
// in order with other client messages. Both fields are part of the model
// preamble, so changing either invalidates the prefix cache.
type LiveContextUpdate struct {
	SystemInstruction LiveContent             `json:"systemInstruction,omitzero"`
	Tools             *LiveContextUpdateTools `json:"tools,omitempty"`
}

// LiveContextUpdateTools replaces the current tool declarations. The wrapper
// distinguishes "no update" from "clear all tools": an empty list clears them.
type LiveContextUpdateTools struct {
	Tools LiveTools `json:"tools,omitempty"`
}

// LiveSetup is documented at https://ai.google.dev/api/live#BidiGenerateContentSetup
type LiveSetup struct {
	Model                    string                   `json:"model"`
	GenerationConfig         LiveGenerationConfig     `json:"generationConfig"`
	SystemInstruction        *LiveContent             `json:"systemInstruction,omitempty"`
	Tools                    LiveTools                `json:"tools,omitempty"`
	SafetySettings           []SafetySetting          `json:"safetySettings,omitempty"`
	RealtimeInputConfig      LiveRealtimeInputConfig  `json:"realtimeInputConfig"`
	SessionResumption        *LiveSessionResumption   `json:"sessionResumption,omitempty"`
	ContextWindowCompression *LiveCompression         `json:"contextWindowCompression,omitempty"`
	InputAudioTranscription  *LiveTranscriptionConfig `json:"inputAudioTranscription,omitempty"`
	OutputAudioTranscription *LiveTranscriptionConfig `json:"outputAudioTranscription,omitempty"`
	Proactivity              *LiveProactivityConfig   `json:"proactivity,omitempty"`
	AvatarConfig             *LiveAvatarConfig        `json:"avatarConfig,omitempty"`
	HistoryConfig            *LiveHistoryConfig       `json:"historyConfig,omitempty"`
	Labels                   map[string]string        `json:"labels,omitempty"`
}

// LiveAvatarConfig configures an avatar model. It is modeled from go-genai
// types.go; the Developer API conversion forwards it unchanged.
type LiveAvatarConfig struct {
	AvatarName       string                `json:"avatarName,omitempty"`
	CustomizedAvatar *LiveCustomizedAvatar `json:"customizedAvatar,omitempty"`
	AudioBitrateBPS  *int32                `json:"audioBitrateBps,omitempty"`
	VideoBitrateBPS  *int32                `json:"videoBitrateBps,omitempty"`
}

// LiveCustomizedAvatar is a reference image for a custom avatar.
type LiveCustomizedAvatar struct {
	ImageMIMEType string `json:"imageMimeType,omitempty"`
	ImageData     []byte `json:"imageData,omitempty"`
}

// LiveGenerationConfig is documented at https://ai.google.dev/api/generate-content#GenerationConfig
//
// The Live endpoint rejects responseLogprobs, responseMimeType, logprobs,
// responseSchema, responseJsonSchema, stopSequence, skipResponseCache,
// routingConfig and audioTimestamp, so they are not modeled.
type LiveGenerationConfig struct {
	CandidateCount     int64                  `json:"candidateCount,omitempty"`
	MaxOutputTokens    int64                  `json:"maxOutputTokens,omitempty"`
	Temperature        float64                `json:"temperature,omitempty"`
	TopP               float64                `json:"topP,omitempty"`
	TopK               int64                  `json:"topK,omitempty"`
	PresencePenalty    float64                `json:"presencePenalty,omitempty"`
	FrequencyPenalty   float64                `json:"frequencyPenalty,omitempty"`
	ResponseModalities []Modality             `json:"responseModalities,omitempty"`
	SpeechConfig       *LiveSpeechConfig      `json:"speechConfig,omitempty"`
	MediaResolution    MediaResolution        `json:"mediaResolution,omitempty"`
	ThinkingConfig     *ThinkingConfig        `json:"thinkingConfig,omitempty"`
	TranslationConfig  *LiveTranslationConfig `json:"translationConfig,omitempty"`
}

// LiveTranslationConfig configures a Live translation session.
//
// Documented at https://ai.google.dev/api/generate-content#TranslationConfig
// and modeled from go-genai types.go (see the file header).
type LiveTranslationConfig struct {
	// TargetLanguageCode is a BCP-47 language code, e.g. "fr".
	TargetLanguageCode string `json:"targetLanguageCode,omitempty"`
	// EchoTargetLanguage also generates audio when the input already is the target language.
	EchoTargetLanguage *bool `json:"echoTargetLanguage,omitempty"`
}

// LiveSpeechConfig is documented at https://ai.google.dev/api/generate-content#SpeechConfig
type LiveSpeechConfig struct {
	VoiceConfig  LiveVoiceConfig `json:"voiceConfig,omitzero"`
	LanguageCode string          `json:"languageCode,omitempty"`
	// MultiSpeakerVoiceConfig is mutually exclusive with VoiceConfig.
	MultiSpeakerVoiceConfig *LiveMultiSpeakerVoiceConfig `json:"multiSpeakerVoiceConfig,omitempty"`
}

// LiveMultiSpeakerVoiceConfig configures a two-speaker session.
//
// Documented at https://ai.google.dev/api/generate-content#MultiSpeakerVoiceConfig
type LiveMultiSpeakerVoiceConfig struct {
	// SpeakerVoiceConfigs holds exactly two entries, one per speaker.
	SpeakerVoiceConfigs []LiveSpeakerVoiceConfig `json:"speakerVoiceConfigs,omitempty"`
}

// LiveSpeakerVoiceConfig configures one speaker of a multi-speaker session.
//
// Documented at https://ai.google.dev/api/generate-content#SpeakerVoiceConfig
type LiveSpeakerVoiceConfig struct {
	// Speaker is the name the prompt uses to identify the speaker.
	Speaker     string          `json:"speaker,omitempty"`
	VoiceConfig LiveVoiceConfig `json:"voiceConfig,omitzero"`
}

// LiveVoiceConfig is documented at https://ai.google.dev/api/generate-content#VoiceConfig
//
// Voice, PrebuiltVoiceConfig and ReplicatedVoiceConfig are a union: set at most
// one of them.
type LiveVoiceConfig struct {
	// Voice is the speaker identifier for synthesis.
	Voice                 string                     `json:"voice,omitempty"`
	PrebuiltVoiceConfig   LivePrebuiltVoiceConfig    `json:"prebuiltVoiceConfig,omitzero"`
	ReplicatedVoiceConfig *LiveReplicatedVoiceConfig `json:"replicatedVoiceConfig,omitempty"`
}

// LivePrebuiltVoiceConfig is documented at https://ai.google.dev/api/generate-content#PrebuiltVoiceConfig
type LivePrebuiltVoiceConfig struct {
	VoiceName string `json:"voiceName"`
}

// LiveReplicatedVoiceConfig is a clone of a user's voice.
//
// Source: go-genai types.go (see the file header), absent from the Gemini API reference.
type LiveReplicatedVoiceConfig struct {
	// MIMEType is "audio/wav": 16-bit signed little-endian wav data at 24kHz.
	MIMEType string `json:"mimeType,omitempty"`
	// VoiceSampleAudio is the audio sample of the cloned voice.
	VoiceSampleAudio []byte `json:"voiceSampleAudio,omitempty"`
	// ConsentAudio is recorded consent, in the same format as VoiceSampleAudio.
	ConsentAudio []byte `json:"consentAudio,omitempty"`
	// VoiceConsentSignature is a server signature for an earlier ConsentAudio.
	// When set, the server verifies it instead of ConsentAudio.
	VoiceConsentSignature *LiveVoiceConsentSignature `json:"voiceConsentSignature,omitempty"`
}

// LiveContent is documented at https://ai.google.dev/api/generate-content#v1beta.Content
type LiveContent struct {
	Parts []LivePart `json:"parts,omitempty"`
	Role  string     `json:"role,omitempty"`
}

// LivePart is documented at https://ai.google.dev/api/generate-content#Part
//
// Only the members the Live endpoint produces are modeled: model turns carry
// text and inline audio, optionally tagged as thoughts. Server-side tool calls
// arrive in LiveToolCall instead of a functionCall part.
type LivePart struct {
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature []byte            `json:"thoughtSignature,omitempty"`
	Text             string            `json:"text,omitempty"`
	InlineData       *LiveBlob         `json:"inlineData,omitempty"`
	FunctionCall     *LiveFunctionCall `json:"functionCall,omitempty"`
}

// LiveBlob is documented at https://ai.google.dev/api/generate-content#Blob
type LiveBlob struct {
	MIMEType string `json:"mimeType"`
	Data     []byte `json:"data"`
}

// LiveTool is documented at https://ai.google.dev/api/generate-content#v1beta.Tool
//
// Only function declarations are modeled. The server-side tools are not modeled
// because LivePart and GroundingMetadata cannot decode their results.
type LiveTool struct {
	FunctionDeclarations []LiveFunctionDeclaration `json:"functionDeclarations,omitempty"`
}

// LiveTools is the Gemini Live tool list.
type LiveTools []LiveTool

// Init converts portable tool declarations for model.
func (t *LiveTools) Init(tools []genai.ToolDef, model string) error {
	behavior := LiveFunctionBlocking
	if strings.Contains(model, "extended-thinking") {
		// Gemini extended-thinking models reject blocking calls.
		behavior = LiveFunctionNonBlocking
	}
	return t.From(tools, behavior)
}

// From converts portable tool declarations to Gemini Live declarations.
func (t *LiveTools) From(tools []genai.ToolDef, behavior LiveFunctionBehavior) error {
	if len(tools) == 0 {
		*t = nil
		return nil
	}
	decls := make([]LiveFunctionDeclaration, len(tools))
	for i := range tools {
		v := &tools[i]
		schema, err := v.GetInputSchema()
		if err != nil {
			return err
		}
		decls[i] = LiveFunctionDeclaration{Name: v.Name, Description: v.Description, ParametersJSONSchema: schema, Behavior: behavior}
	}
	*t = LiveTools{{FunctionDeclarations: decls}}
	return nil
}

// LiveFunctionDeclaration is documented at https://ai.google.dev/api/generate-content#FunctionDeclaration
type LiveFunctionDeclaration struct {
	Name                 string               `json:"name"`
	Description          string               `json:"description"`
	ParametersJSONSchema json.RawMessage      `json:"parametersJsonSchema"`
	Behavior             LiveFunctionBehavior `json:"behavior"`
}

// LiveFunctionBehavior is documented at https://ai.google.dev/api/generate-content#FunctionDeclaration
type LiveFunctionBehavior string

// LiveFunctionBehavior values.
const (
	LiveFunctionBlocking    LiveFunctionBehavior = "BLOCKING"
	LiveFunctionNonBlocking LiveFunctionBehavior = "NON_BLOCKING"
)

// LiveRealtimeInputConfig is documented at https://ai.google.dev/api/live#RealtimeInputConfig
type LiveRealtimeInputConfig struct {
	AutomaticActivityDetection        *LiveAutomaticActivityDetection `json:"automaticActivityDetection,omitempty"`
	ActivityHandling                  LiveActivityHandling            `json:"activityHandling"`
	TurnCoverage                      LiveTurnCoverage                `json:"turnCoverage"`
	InterimTranscriptTimestampEnabled bool                            `json:"interimTranscriptTimestampEnabled,omitempty"`
}

// LiveActivityHandling is documented at https://ai.google.dev/api/live#RealtimeInputConfig.ActivityHandling
type LiveActivityHandling string

// LiveActivityHandling values.
const (
	LiveActivityHandlingUnspecified LiveActivityHandling = "ACTIVITY_HANDLING_UNSPECIFIED"
	LiveActivityInterrupts          LiveActivityHandling = "START_OF_ACTIVITY_INTERRUPTS"
	LiveActivityNoInterruption      LiveActivityHandling = "NO_INTERRUPTION"
)

// LiveTurnCoverage is documented at https://ai.google.dev/api/live#RealtimeInputConfig.TurnCoverage
type LiveTurnCoverage string

// LiveTurnCoverage values.
const (
	LiveTurnCoverageUnspecified   LiveTurnCoverage = "TURN_COVERAGE_UNSPECIFIED"
	LiveTurnOnlyActivity          LiveTurnCoverage = "TURN_INCLUDES_ONLY_ACTIVITY"
	LiveTurnAllInput              LiveTurnCoverage = "TURN_INCLUDES_ALL_INPUT"
	LiveTurnAudioActivityAllVideo LiveTurnCoverage = "TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO"
)

// LiveAutomaticActivityDetection is documented at https://ai.google.dev/api/live#RealtimeInputConfig.AutomaticActivityDetection
type LiveAutomaticActivityDetection struct {
	Disabled                 bool                 `json:"disabled"`
	StartOfSpeechSensitivity LiveStartSensitivity `json:"startOfSpeechSensitivity,omitempty"`
	PrefixPaddingMs          int64                `json:"prefixPaddingMs,omitempty"`
	EndOfSpeechSensitivity   LiveEndSensitivity   `json:"endOfSpeechSensitivity,omitempty"`
	SilenceDurationMs        int64                `json:"silenceDurationMs,omitempty"`
}

// LiveStartSensitivity is documented at https://ai.google.dev/api/live#StartSensitivity
type LiveStartSensitivity string

// LiveStartSensitivity values.
const (
	LiveStartSensitivityUnspecified LiveStartSensitivity = "START_SENSITIVITY_UNSPECIFIED"
	LiveStartSensitivityHigh        LiveStartSensitivity = "START_SENSITIVITY_HIGH"
	LiveStartSensitivityLow         LiveStartSensitivity = "START_SENSITIVITY_LOW"
)

// LiveEndSensitivity is documented at https://ai.google.dev/api/live#EndSensitivity
type LiveEndSensitivity string

// LiveEndSensitivity values.
const (
	LiveEndSensitivityUnspecified LiveEndSensitivity = "END_SENSITIVITY_UNSPECIFIED"
	LiveEndSensitivityHigh        LiveEndSensitivity = "END_SENSITIVITY_HIGH"
	LiveEndSensitivityLow         LiveEndSensitivity = "END_SENSITIVITY_LOW"
)

// LiveSessionResumption is documented at https://ai.google.dev/api/live#SessionResumptionConfig
type LiveSessionResumption struct {
	Handle string `json:"handle,omitempty"`
}

// LiveCompression is documented at https://ai.google.dev/api/live#ContextWindowCompressionConfig
type LiveCompression struct {
	TriggerTokens int64             `json:"triggerTokens,string"`
	SlidingWindow LiveSlidingWindow `json:"slidingWindow"`
}

// LiveSlidingWindow is documented at https://ai.google.dev/api/live#ContextWindowCompressionConfig.SlidingWindow
type LiveSlidingWindow struct {
	TargetTokens int64 `json:"targetTokens,string"`
}

// LiveTranscriptionConfig is documented at https://ai.google.dev/api/live#AudioTranscriptionConfig
type LiveTranscriptionConfig struct {
	LanguageCodes    []string              `json:"languageCodes,omitempty"`
	CustomVocabulary []string              `json:"customVocabulary,omitempty"`
	WordTimestamp    bool                  `json:"wordTimestamp,omitempty"`
	Diarization      bool                  `json:"diarization,omitempty"`
	Mode             LiveTranscriptionMode `json:"mode,omitempty"`
}

// LiveTranscriptionMode is documented at https://ai.google.dev/api/live#AudioTranscriptionConfig.Mode
type LiveTranscriptionMode string

// LiveTranscriptionMode values.
const (
	LiveTranscriptionModeUnspecified LiveTranscriptionMode = "MODE_UNSPECIFIED"
	LiveTranscriptionModeVerbatim    LiveTranscriptionMode = "VERBATIM"
	LiveTranscriptionModeSmart       LiveTranscriptionMode = "SMART"
)

// LiveProactivityConfig is documented at https://ai.google.dev/api/live#ProactivityConfig
type LiveProactivityConfig struct {
	ProactiveAudio bool `json:"proactiveAudio,omitempty"`
}

// LiveHistoryConfig is documented at https://ai.google.dev/api/live#HistoryConfig
type LiveHistoryConfig struct {
	InitialHistoryInClientContent bool `json:"initialHistoryInClientContent,omitempty"`
}

// LiveClientContent is documented at https://ai.google.dev/api/live#BidiGenerateContentClientContent
type LiveClientContent struct {
	Turns        []LiveContent `json:"turns"`
	TurnComplete bool          `json:"turnComplete"`
}

// LiveRealtimeInput is documented at https://ai.google.dev/api/live#BidiGenerateContentRealtimeInput
type LiveRealtimeInput struct {
	// MediaChunks is deprecated by the API in favor of Audio, Video and Text.
	MediaChunks     []LiveBlob      `json:"mediaChunks,omitempty"`
	Audio           *LiveBlob       `json:"audio,omitempty"`
	Video           *LiveBlob       `json:"video,omitempty"`
	Text            string          `json:"text,omitempty"`
	AudioStreamEnd  bool            `json:"audioStreamEnd,omitempty"`
	MediaResolution MediaResolution `json:"mediaResolution,omitempty"`
	ActivityStart   *LiveActivity   `json:"activityStart,omitempty"`
	ActivityEnd     *LiveActivity   `json:"activityEnd,omitempty"`
}

// LiveActivity is the empty payload of activityStart and activityEnd.
//
// Documented at https://ai.google.dev/api/live#BidiGenerateContentRealtimeInput.ActivityStart and
// https://ai.google.dev/api/live#BidiGenerateContentRealtimeInput.ActivityEnd
type LiveActivity struct{}

// LiveToolResponse is documented at https://ai.google.dev/api/live#BidiGenerateContentToolResponse
type LiveToolResponse struct {
	FunctionResponses []LiveFunctionResponse `json:"functionResponses"`
}

// LiveFunctionResponse is documented at https://ai.google.dev/api/generate-content#FunctionResponse
type LiveFunctionResponse struct {
	ID           string                         `json:"id,omitempty"`
	Name         string                         `json:"name"`
	Response     json.RawMessage                `json:"response,omitempty"`
	WillContinue bool                           `json:"willContinue,omitempty"`
	Scheduling   LiveFunctionResponseScheduling `json:"scheduling,omitempty"`
	// Parts carries a media function response instead of Response.
	Parts []LiveFunctionResponsePart `json:"parts,omitempty"`
}

// LiveFunctionResponsePart is one media part of a function response.
//
// Documented at https://ai.google.dev/api/generate-content#FunctionResponsePart
type LiveFunctionResponsePart struct {
	InlineData *LiveBlob                     `json:"inlineData,omitempty"`
	FileData   *LiveFunctionResponseFileData `json:"fileData,omitempty"`
}

// LiveFunctionResponseFileData is URI based function response media.
//
// Documented at https://ai.google.dev/api/generate-content#FunctionResponseFileData
type LiveFunctionResponseFileData struct {
	FileURI  string `json:"fileUri,omitempty"`
	MIMEType string `json:"mimeType,omitempty"`
}

// LiveFunctionResponseScheduling is documented at https://ai.google.dev/api/generate-content#FunctionResponseScheduling
type LiveFunctionResponseScheduling string

// LiveFunctionResponseScheduling values.
const (
	LiveSchedulingUnspecified LiveFunctionResponseScheduling = "SCHEDULING_UNSPECIFIED"
	LiveSchedulingSilent      LiveFunctionResponseScheduling = "SILENT"
	LiveSchedulingWhenIdle    LiveFunctionResponseScheduling = "WHEN_IDLE"
	LiveSchedulingInterrupt   LiveFunctionResponseScheduling = "INTERRUPT"
)

// LiveTextResult wraps non-object tool text in an object response.
//
// Documented at https://ai.google.dev/api/generate-content#FunctionResponse
type LiveTextResult struct {
	Response string `json:"response"`
}

// LiveServerMessage is documented at https://ai.google.dev/api/live#BidiGenerateContentServerMessage
//
// Error is not part of the declared message but the Live endpoint sends it as a
// terminal frame; the SDK handles it the same way in live.go Receive.
type LiveServerMessage struct {
	SetupComplete                *LiveSetupComplete        `json:"setupComplete,omitempty"`
	ServerContent                *LiveServerContent        `json:"serverContent,omitempty"`
	ToolCall                     *LiveToolCall             `json:"toolCall,omitempty"`
	ToolCallCancellation         *LiveToolCallCancellation `json:"toolCallCancellation,omitempty"`
	UsageMetadata                *LiveUsage                `json:"usageMetadata,omitempty"`
	GoAway                       *LiveGoAway               `json:"goAway,omitempty"`
	SessionResumptionUpdate      *LiveResumptionUpdate     `json:"sessionResumptionUpdate,omitempty"`
	VoiceActivityDetectionSignal *LiveVADSignal            `json:"voiceActivityDetectionSignal,omitempty"`
	VoiceActivity                *LiveVoiceActivity        `json:"voiceActivity,omitempty"`
	Error                        *LiveError                `json:"error,omitempty"`
}

// GoAway warns that Gemini Live will close the connection. TimeLeft is zero when
// the server did not supply a duration.
//
// Documented at https://ai.google.dev/api/live#GoAway
type GoAway struct {
	TimeLeft time.Duration
}

// RealtimeEventName implements genai.RealtimeProviderEvent.
func (*GoAway) RealtimeEventName() string { return "goAway" }

// Resumption is a resumable-state update. Handle is empty when the current state
// has no new resumable token.
//
// Documented at https://ai.google.dev/api/live#SessionResumptionUpdate
type Resumption struct {
	Handle string
}

// RealtimeEventName implements genai.RealtimeProviderEvent.
func (*Resumption) RealtimeEventName() string { return "sessionResumptionUpdate" }

// InteractionStatus reports the Live server's activity state. Status is
// IN_PROGRESS while background reasoning or tool execution continues, and IDLE
// when all processing completes.
//
// Documented at https://ai.google.dev/api/live#BidiGenerateContentServerContent.InteractionStatus
type InteractionStatus struct {
	Status InteractionStatusKind
}

// RealtimeEventName implements genai.RealtimeProviderEvent.
func (*InteractionStatus) RealtimeEventName() string { return "interactionStatus" }

// InteractionStatusKind is the Live server's activity state.
//
// Documented at https://ai.google.dev/api/live#InteractionStatus
type InteractionStatusKind string

// InteractionStatusKind values.
const (
	InteractionStatusUnspecified InteractionStatusKind = "INTERACTION_STATUS_UNSPECIFIED"
	InteractionStatusInProgress  InteractionStatusKind = "IN_PROGRESS"
	// InteractionStatusRequiresAction is deprecated by the API in favor of IDLE.
	InteractionStatusRequiresAction InteractionStatusKind = "REQUIRES_ACTION"
	InteractionStatusIdle           InteractionStatusKind = "IDLE"
)

// LiveTurnCompleteReason is the reason a turn finished.
//
// Source: go-genai types.go (see the file header).
type LiveTurnCompleteReason string

// LiveTurnCompleteReason values.
const (
	LiveTurnCompleteUnspecified      LiveTurnCompleteReason = "TURN_COMPLETE_REASON_UNSPECIFIED"
	LiveTurnCompleteMalformedCall    LiveTurnCompleteReason = "MALFORMED_FUNCTION_CALL"
	LiveTurnCompleteResponseRejected LiveTurnCompleteReason = "RESPONSE_REJECTED"
	LiveTurnCompleteNeedMoreInput    LiveTurnCompleteReason = "NEED_MORE_INPUT"
	LiveTurnCompleteProhibitedInput  LiveTurnCompleteReason = "PROHIBITED_INPUT_CONTENT"
)

// LiveSetupComplete is documented at https://ai.google.dev/api/live#BidiGenerateContentSetupComplete
type LiveSetupComplete struct {
	SessionID             string                     `json:"sessionId,omitempty"`
	VoiceConsentSignature *LiveVoiceConsentSignature `json:"voiceConsentSignature,omitempty"`
}

// LiveVoiceConsentSignature is the signature of a verified consent audio.
//
// Source: go-genai types.go (see the file header), absent from the Gemini API reference.
type LiveVoiceConsentSignature struct {
	Signature string `json:"signature,omitempty"`
}

// LiveServerContent is documented at https://ai.google.dev/api/live#BidiGenerateContentServerContent
type LiveServerContent struct {
	ModelTurn          LiveContent            `json:"modelTurn,omitempty"` //nolint:modernize // Preserve the existing JSON wire shape.
	TurnComplete       bool                   `json:"turnComplete,omitempty"`
	Interrupted        bool                   `json:"interrupted,omitempty"`
	GenerationComplete bool                   `json:"generationComplete,omitempty"`
	WaitingForInput    bool                   `json:"waitingForInput,omitempty"`
	InteractionStatus  InteractionStatusKind  `json:"interactionStatus,omitempty"`
	TurnCompleteReason LiveTurnCompleteReason `json:"turnCompleteReason,omitempty"`
	InputTranscription *LiveTranscription     `json:"inputTranscription,omitempty"`
	// InterimInputTranscription is updated while the user is still speaking.
	InterimInputTranscription *LiveTranscription `json:"interimInputTranscription,omitempty"`
	OutputTranscription       *LiveTranscription `json:"outputTranscription,omitempty"`
	// GroundingMetadata is populated when a server-side grounding tool is enabled.
	GroundingMetadata  *GroundingMetadata  `json:"groundingMetadata,omitempty"`
	UrlContextMetadata *UrlContextMetadata `json:"urlContextMetadata,omitempty"`
}

// LiveTranscription is documented at https://ai.google.dev/api/live#BidiGenerateContentTranscription
//
// StartOffset and EndOffset are proto durations encoded as JSON strings, e.g.
// "1.5s". Finished is sent by the endpoint and modeled by the SDK even though the
// reference page only lists the text and the offsets.
type LiveTranscription struct {
	Text         string         `json:"text,omitempty"`
	Finished     bool           `json:"finished,omitempty"`
	LanguageCode string         `json:"languageCode,omitempty"`
	SpeakerLabel string         `json:"speakerLabel,omitempty"`
	StartOffset  string         `json:"startOffset,omitempty"`
	EndOffset    string         `json:"endOffset,omitempty"`
	Words        []LiveWordInfo `json:"words,omitempty"`
}

// LiveWordInfo is word-level transcription timing.
//
// Documented at https://ai.google.dev/api/generate-content#WordInfo
type LiveWordInfo struct {
	Word        string `json:"word,omitempty"`
	StartOffset string `json:"startOffset,omitempty"`
	EndOffset   string `json:"endOffset,omitempty"`
}

// LiveToolCall is documented at https://ai.google.dev/api/live#BidiGenerateContentToolCall
type LiveToolCall struct {
	FunctionCalls []LiveFunctionCall `json:"functionCalls"`
}

// LiveFunctionCall is documented at https://ai.google.dev/api/generate-content#FunctionCall
type LiveFunctionCall struct {
	ID               string          `json:"id,omitempty"`
	Name             string          `json:"name"`
	Args             json.RawMessage `json:"args,omitempty"`
	ThoughtSignature []byte          `json:"thoughtSignature,omitempty"`
}

// LiveToolCallCancellation is documented at https://ai.google.dev/api/live#BidiGenerateContentToolCallCancellation
type LiveToolCallCancellation struct {
	IDs []string `json:"ids"`
}

// ToolCallCancellation reports calls that Gemini Live cancelled. The client
// must not execute them. If a call already ran, undo its side effects when
// possible.
//
// Documented at https://ai.google.dev/api/live#BidiGenerateContentToolCallCancellation
type ToolCallCancellation struct {
	IDs []string
}

// RealtimeEventName implements genai.RealtimeProviderEvent.
func (*ToolCallCancellation) RealtimeEventName() string { return "toolCallCancellation" }

// LiveUsage is documented at https://ai.google.dev/api/live#UsageMetadata
type LiveUsage struct {
	PromptTokenCount           int64                    `json:"promptTokenCount,omitempty"`
	CachedContentTokenCount    int64                    `json:"cachedContentTokenCount,omitempty"`
	ResponseTokenCount         int64                    `json:"responseTokenCount,omitempty"`
	ThoughtsTokenCount         int64                    `json:"thoughtsTokenCount,omitempty"`
	TotalTokenCount            int64                    `json:"totalTokenCount,omitempty"`
	ServiceTier                LiveServiceTier          `json:"serviceTier,omitempty"`
	PromptTokensDetails        []LiveModalityTokenCount `json:"promptTokensDetails,omitempty"`
	CacheTokensDetails         []LiveModalityTokenCount `json:"cacheTokensDetails,omitempty"`
	ResponseTokensDetails      []LiveModalityTokenCount `json:"responseTokensDetails,omitempty"`
	ToolUsePromptTokenCount    int64                    `json:"toolUsePromptTokenCount,omitempty"`
	ToolUsePromptTokensDetails []LiveModalityTokenCount `json:"toolUsePromptTokensDetails,omitempty"`
	TrafficType                LiveTrafficType          `json:"trafficType,omitempty"`
}

// LiveServiceTier is the quality of service that processed the session.
//
// Source: go-genai types.go (see the file header).
type LiveServiceTier string

// LiveServiceTier values.
const (
	LiveServiceTierUnspecified LiveServiceTier = "unspecified"
	LiveServiceTierFlex        LiveServiceTier = "flex"
	LiveServiceTierStandard    LiveServiceTier = "standard"
	LiveServiceTierPriority    LiveServiceTier = "priority"
)

// LiveTrafficType is the quota the session was billed against.
//
// Source: go-genai types.go (see the file header).
type LiveTrafficType string

// LiveTrafficType values.
const (
	LiveTrafficTypeUnspecified      LiveTrafficType = "TRAFFIC_TYPE_UNSPECIFIED"
	LiveTrafficTypeOnDemand         LiveTrafficType = "ON_DEMAND"
	LiveTrafficTypeOnDemandPriority LiveTrafficType = "ON_DEMAND_PRIORITY"
	LiveTrafficTypeOnDemandFlex     LiveTrafficType = "ON_DEMAND_FLEX"
	LiveTrafficTypeOnDemandOffpeak  LiveTrafficType = "ON_DEMAND_OFFPEAK"
)

// LiveModalityTokenCount is token accounting for one modality.
type LiveModalityTokenCount struct {
	Modality   Modality `json:"modality,omitempty"`
	TokenCount int64    `json:"tokenCount,omitempty"`
}

// LiveGoAway is documented at https://ai.google.dev/api/live#GoAway
//
// TimeLeft is a proto duration encoded as a JSON string, e.g. "5.5s".
type LiveGoAway struct {
	TimeLeft string `json:"timeLeft"`
}

// LiveResumptionUpdate is documented at https://ai.google.dev/api/live#SessionResumptionUpdate
type LiveResumptionUpdate struct {
	NewHandle string `json:"newHandle,omitempty"`
	Resumable bool   `json:"resumable,omitempty"`
	// LastConsumedClientMessageIndex is only sent when the session was resumed transparently.
	LastConsumedClientMessageIndex int64 `json:"lastConsumedClientMessageIndex,omitempty,string"`
}

// LiveVADSignal is documented at https://ai.google.dev/api/live#BidiGenerateContentServerMessage
type LiveVADSignal struct {
	VADSignalType LiveVADSignalType `json:"vadSignalType"`
}

// LiveVADSignalType is a voice activity detection signal.
type LiveVADSignalType string

// LiveVADSignalType values.
const (
	LiveVADSignalUnspecified LiveVADSignalType = "VAD_SIGNAL_TYPE_UNSPECIFIED"
	LiveVADSignalEOS         LiveVADSignalType = "VAD_SIGNAL_TYPE_EOS"
	LiveVADSignalSOS         LiveVADSignalType = "VAD_SIGNAL_TYPE_SOS"
)

// LiveVoiceActivity is documented at https://ai.google.dev/api/live#VoiceActivity
//
// AudioOffset is a proto duration encoded as a JSON string, e.g. "1.5s".
type LiveVoiceActivity struct {
	Type        LiveVoiceActivityType `json:"type"`
	AudioOffset string                `json:"audioOffset,omitempty"`
}

// LiveVoiceActivityType is a voice activity event.
type LiveVoiceActivityType string

// LiveVoiceActivityType values.
const (
	LiveVoiceActivityEnd         LiveVoiceActivityType = "ACTIVITY_END"
	LiveVoiceActivityStart       LiveVoiceActivityType = "ACTIVITY_START"
	LiveVoiceActivityUnspecified LiveVoiceActivityType = "TYPE_UNSPECIFIED"
)

// LiveError is the terminal error frame of the Live endpoint.
//
// Documented at https://github.com/googleapis/googleapis/blob/master/google/rpc/status.proto
type LiveError struct {
	Code    int               `json:"code,omitempty"`
	Message string            `json:"message,omitempty"`
	Status  string            `json:"status,omitempty"`
	Details []json.RawMessage `json:"details,omitempty"`
}
