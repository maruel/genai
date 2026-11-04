// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package scoreboard declares the structures to define a scoreboard.
//
// It is in a separate package from genai to reduce noise.
package scoreboard

import (
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// Model specifies a model to test and whether it should run in reasoning mode.
//
// Most models only support one or the other, but some support both.
// Functionality often differs depending on whether reasoning is enabled.
type Model struct {
	Model  string
	Reason bool
}

func (m *Model) String() string {
	normalized := strings.ReplaceAll(m.Model, ":", "-")
	if m.Reason {
		normalized += "_thinking"
	}
	return normalized
}

// Wire identifies how a provider exchanges bytes with a model.
type Wire string

// Supported wires.
//
// A row records the wire it was measured over, because the same model can answer the
// same suite over two of them: the Responses API travels over HTTP or over WebSocket,
// and a command line provider answers the chat suite through a process.
const (
	// WireHTTP sends JSON over HTTPS. The recorder wraps the transport, one cassette
	// per scenario name.
	WireHTTP Wire = "http"
	// WireWebSocket sends JSON frames over a WebSocket for the Responses API. The
	// recorder proxies the frames, one fixture per scenario name.
	WireWebSocket Wire = "ws"
	// WireSession runs a realtime session over a WebSocket. The recorder wraps
	// ConnectRealtime, one fixture per session call.
	WireSession Wire = "session"
	// WireSubprocess runs a command line provider. The recorder wraps the process,
	// one trace per scenario name.
	WireSubprocess Wire = "subprocess"
)

// Key identifies one row within its section: a model, its reasoning mode and the
// wire it was measured over.
type Key struct {
	Model  string
	Reason bool
	Wire   Wire
}

// String returns the model with its reasoning and wire suffixes. HTTP carries no
// suffix: it is the norm, and only a wire that shares a model with another needs
// a name of its own. The suite is not part of the name: cassette directories come
// from the test name and the model.
func (k Key) String() string {
	normalized := strings.ReplaceAll(k.Model, ":", "-")
	if k.Reason {
		normalized += "_thinking"
	}
	// A session is implied by its suite, so the wire only names itself when a suite
	// travels over several of them.
	switch k.Wire {
	case WireHTTP, WireSession:
	case WireWebSocket:
		normalized += "_ws"
	case WireSubprocess:
		normalized += "_cli"
	}
	return normalized
}

// Modality is one of the supported modalities.
type Modality string

// Validate returns an error if the Modality is not a known value.
func (m Modality) Validate() error {
	switch m {
	case ModalityAudio, ModalityDecision, ModalityDocument, ModalityEmbedding, ModalityImage, ModalityText, ModalityVideo:
		return nil
	default:
		return fmt.Errorf("invalid Modality: %q", m)
	}
}

const (
	// ModalityAudio is support for audio formats like MP3, WAV, Opus, Flac, etc.
	ModalityAudio Modality = "audio"
	// ModalityDecision is typed answers to questions about a state.
	ModalityDecision Modality = "decision"
	// ModalityDocument is support for PDF with multi-modal comprehension, both images and text. This includes
	// code blocks.
	ModalityDocument Modality = "document"
	// ModalityEmbedding is a vector representing input content.
	ModalityEmbedding Modality = "embedding"
	// ModalityImage is support for image formats like PNG, JPEG, often single frame GIF, and WEBP.
	ModalityImage Modality = "image"
	// ModalityText is for raw text.
	ModalityText Modality = "text"
	// ModalityVideo is support for video formats like MP4 or MKV.
	ModalityVideo Modality = "video"
)

// Functionality defines which functionalites are supported in a scenario.
//
// The first group is for all models. The remainder is for text models.
//
// The second group is about tool use, is as of 2025-08 is only supported for text models.
//
// The third group is about text specific features.
type Functionality struct {
	// ReportRateLimits means that the provider reports rate limits in its Usage.
	ReportRateLimits bool `json:"reportRateLimits,omitzero"`
	// ReportTokenUsage means that the token usage is correctly reported in all cases. It is flaky if it is not
	// reported in some specific cases. A frequent example is tokens not being reported in JSON output mode.
	ReportTokenUsage TriState `json:"reportTokenUsage,omitzero"`
	// ReportFinishReason means that the finish reason (FinishStop, FinishLength, etc) is not correctly reported.
	ReportFinishReason TriState `json:"reportFinishReason,omitzero"`
	// Seed is set when the provider and model combination supports seed for reproducibility.
	Seed bool `json:"seed,omitzero"`

	// Text related fields.

	// OutOfOrder measures a spoken conversation with consecutive user messages
	// and intentionally deferred tool results. It preserves actual assistant
	// responses and call metadata, then delivers results in reverse call order.
	// Repeated task checks are allowed; multiple calls are not required.
	//
	// The probe requires Tools == True. Nil means unmeasured, not unsupported,
	// including when the basic tool check does not pass. Each mode's score applies
	// only to the measured model and options. It does not establish cancellation,
	// concurrent mutation, gateway scheduling, or statistical reliability.
	OutOfOrder *TriState `json:"outOfOrder,omitzero"`

	// Tools means that tool call is supported. This is a requirement for MCP. Some provider support tool
	// calling but the model is very flaky at actually requesting the calls. This is more frequent on highly
	// quantized models, small models or MoE models.
	Tools TriState `json:"tools,omitzero"`
	// ToolsBiased is true when we ask the LLM to use a tool in an ambiguous biased question, it will always
	// reply with the first readily available answer.
	//
	// This means that when using enum, it is important to understand that the LLM will put heavy weight on the
	// first option.
	//
	// This is affected by two factors: model size and quantization. Quantization affects this dramatically.
	ToolsBiased TriState `json:"toolsBiased,omitzero"`
	// ToolsIndecisive is True when we ask the LLM to use a tool in an ambiguous biased question, it'll call both
	// options. It is Flaky when both can happen.
	//
	// This is actually fine, it means that the LLM will be less opinionated in some cases. The test into which
	// a LLM is indecisive is likely model-specific too.
	ToolsIndecisive TriState `json:"toolsIndecisive,omitzero"`
	// ToolCallRequired is true when the value genai.ToolCallRequired works. Not supporting it significantly
	// increases the risk of flakiness.
	ToolCallRequired bool `json:"toolCallRequired,omitzero"`
	// WebSearch is true if the provider supports web search via its own backend.
	WebSearch bool `json:"webSearch,omitzero"`
	// WebFetch is true if the provider supports fetching content from URLs via its own backend.
	WebFetch bool `json:"webFetch,omitzero"`

	// JSON means that the model supports enforcing that the response is valid JSON but not necessarily with a
	// schema.
	JSON bool `json:"json,omitzero"`
	// JSONSchema describes the reply structure the model supports enforcing.
	JSONSchema JSONSchema `json:"jsonSchema,omitzero"`
	// Citations is set when the provider and model combination supports citations in the response.
	Citations bool `json:"citations,omitzero"`
	// TopLogprobs is set when the provider and model combination supports top_logprobs.
	TopLogprobs bool `json:"topLogprobs,omitzero"`
	// MaxTokens means that the provider supports limiting text output to a specific number of tokens.
	//
	// Tokens are characters nor words. The tokens are embedding specific, and each model family uses a
	// different vocabulary. Thus the number of characters generated varies wildly.
	//
	// It fails more often with model with implicit reasoning.
	MaxTokens bool `json:"maxTokens,omitzero"`
	// StopSequence means that the provider supports stop words. The number of stop words is generally limited,
	// frequently to 5 words. The sequence should be a valid token in the model's vocabulary.
	StopSequence bool `json:"stopSequence,omitzero"`

	_ struct{}
}

// Less returns true if the functionality is less that the other.
func (f *Functionality) Less(rhs *Functionality) bool {
	if !f.ReportRateLimits && rhs.ReportRateLimits {
		return true
	}
	if f.ReportTokenUsage == False && rhs.ReportTokenUsage != False {
		return true
	}
	if f.ReportFinishReason == False && rhs.ReportFinishReason != False {
		return true
	}
	if !f.Seed && rhs.Seed {
		return true
	}
	if f.Tools == False && rhs.Tools != False {
		return true
	}
	if f.OutOfOrder != nil && rhs.OutOfOrder != nil {
		if *f.OutOfOrder == False && *rhs.OutOfOrder != False || *f.OutOfOrder == Flaky && *rhs.OutOfOrder == True {
			return true
		}
	}
	// Ignore ToolsBiased and ToolsIndecisive.
	if !f.ToolCallRequired && rhs.ToolCallRequired {
		return true
	}
	if !f.JSON && rhs.JSON {
		return true
	}
	if f.JSONSchema.Less(rhs.JSONSchema) {
		return true
	}
	if !f.Citations && rhs.Citations {
		return true
	}
	// Ignore TopLogprobs, it's not important enough.
	if !f.MaxTokens && rhs.MaxTokens {
		return true
	}
	if !f.StopSequence && rhs.StopSequence {
		return true
	}
	return false
}

// Validate returns an error if the Functionality contains invalid values.
func (f *Functionality) Validate() error {
	if err := f.ReportTokenUsage.Validate(); err != nil {
		return fmt.Errorf("invalid ReportTokenUsage: %w", err)
	}
	if err := f.ReportFinishReason.Validate(); err != nil {
		return fmt.Errorf("invalid ReportFinishReason: %w", err)
	}
	if f.OutOfOrder != nil {
		if err := f.OutOfOrder.Validate(); err != nil {
			return fmt.Errorf("invalid OutOfOrder: %w", err)
		}
	}
	if err := f.Tools.Validate(); err != nil {
		return fmt.Errorf("invalid Tools: %w", err)
	}
	if err := f.ToolsBiased.Validate(); err != nil {
		return fmt.Errorf("invalid ToolsBiased: %w", err)
	}
	if err := f.ToolsIndecisive.Validate(); err != nil {
		return fmt.Errorf("invalid ToolsIndecisive: %w", err)
	}
	if err := f.JSONSchema.Validate(); err != nil {
		return fmt.Errorf("invalid JSONSchema: %w", err)
	}
	if f.Tools == False {
		if f.ToolsBiased != False {
			return fmt.Errorf("invalid ToolsBiased %s when Tools is false", f.ToolsBiased.String())
		}
		if f.ToolsIndecisive != False {
			return fmt.Errorf("invalid ToolsIndecisive %s when Tools is false", f.ToolsIndecisive.String())
		}
		if f.ToolCallRequired {
			return fmt.Errorf("invalid ToolCallRequired %t when Tools is false", f.ToolCallRequired)
		}
	}
	return nil
}

// JSONSchema is the reply structure the model supports enforcing with genai.GenOptionText.DecodeAs.
//
// The zero value means no structure can be enforced.
type JSONSchema struct {
	// Object means the reply root can be enforced to be an object, i.e. a struct or a map.
	Object TriState `json:"object,omitzero"`
	// Array means the reply root can be enforced to be an array, i.e. a slice or an array. Most providers
	// reject it, OpenAI and Anthropic require an object at the root, Gemini accepts an array.
	Array TriState `json:"array,omitzero"`
}

// Less returns true if the JSONSchema is less capable than the other.
func (j JSONSchema) Less(rhs JSONSchema) bool {
	if j.Object == False && rhs.Object != False {
		return true
	}
	if j.Array == False && rhs.Array != False {
		return true
	}
	return false
}

// Validate returns an error if the JSONSchema contains invalid values.
func (j JSONSchema) Validate() error {
	if err := j.Object.Validate(); err != nil {
		return fmt.Errorf("invalid Object: %w", err)
	}
	if err := j.Array.Validate(); err != nil {
		return fmt.Errorf("invalid Array: %w", err)
	}
	return nil
}

// TriState helps describing support when a feature "kinda work", which is frequent with LLM's inherent
// non-determinism.
type TriState int8

func (t TriState) String() string {
	t -= -1
	if t < 0 || t >= TriState(len(triStateIndex)-1) {
		return "TriState(" + strconv.FormatInt(int64(t+-1), 10) + ")"
	}
	return triStateName[triStateIndex[t]:triStateIndex[t+1]]
}

// GoString implements fmt.GoStringer.
func (t TriState) GoString() string {
	return t.String()
}

// Validate returns an error if the TriState is not a known value.
func (t TriState) Validate() error {
	switch t {
	case False, True, Flaky:
		return nil
	default:
		return fmt.Errorf("invalid TriState: %q", t)
	}
}

// MarshalJSON implements json.Marshaler.
func (t TriState) MarshalJSON() ([]byte, error) {
	switch t {
	case False:
		// TODO: Not present.
		return []byte(`"false"`), nil
	case True:
		return []byte(`"true"`), nil
	case Flaky:
		return []byte(`"flaky"`), nil
	default:
		return nil, fmt.Errorf("invalid TriState: %q", t)
	}
}

// UnmarshalJSON implements json.Unmarshaler.
func (t *TriState) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch s {
	case "false":
		*t = False
	case "true":
		*t = True
	case "flaky":
		*t = Flaky
	default:
		return fmt.Errorf("invalid TriState: %q", s)
	}
	return nil
}

// TriState values for feature support.
const (
	// False means the feature is not supported.
	False TriState = 0
	// True means the feature is supported.
	True TriState = 1
	// Flaky means the feature works intermittently.
	Flaky TriState = -1
)

const triStateName = "flakyfalsetrue"

var triStateIndex = [...]uint8{0, 5, 10, 14}

// DecisionFunctionality describes typed decision inference through Provider.SystemOne.
type DecisionFunctionality struct {
	// Noul, Choice and Score indicate supported question types.
	Noul   bool `json:"noul,omitzero"`
	Choice bool `json:"choice,omitzero"`
	Score  bool `json:"score,omitzero"`
	// Object and Array indicate supported JSON state shapes.
	Object bool `json:"object,omitzero"`
	Array  bool `json:"array,omitzero"`
	// ReportTokenUsage describes input token reporting across successful requests.
	ReportTokenUsage TriState `json:"reportTokenUsage,omitzero"`
}

// Validate checks decision reporting values.
func (f *DecisionFunctionality) Validate() error {
	return f.ReportTokenUsage.Validate()
}

// EmbeddingFunctionality records measured text batch embedding behavior.
// Successful qualification checks vectors, input order and semantic retrieval.
type EmbeddingFunctionality struct {
	// Dimensions is the default vector size measured with no requested size.
	Dimensions int `json:"dimensions,omitzero"`
	// RequestedDimensions is nil when unmeasured, otherwise whether the requested
	// qualification size was honored. It doesn't imply arbitrary size support.
	RequestedDimensions *bool `json:"requestedDimensions,omitzero"`
	// ReportTokenUsage describes input token reporting across successful requests.
	ReportTokenUsage TriState `json:"reportTokenUsage,omitzero"`
}

// Validate checks embedding dimensions and token reporting.
func (f *EmbeddingFunctionality) Validate() error {
	if f.Dimensions < 0 {
		return errors.New("embedding dimensions must not be negative")
	}
	return f.ReportTokenUsage.Validate()
}

// RealtimeFunctionality records measured bidirectional realtime session behavior.
//
// The sample rates are measured from the qualified session. Boolean fields
// report capabilities whose probe passed; false means the probe was skipped,
// failed, or did not run. Tools is tri-state because models can call tools only
// intermittently with audio output.
type RealtimeFunctionality struct {
	// InputSampleRate is the native mono PCM16LE input rate in Hz.
	InputSampleRate int `json:"inputSampleRate,omitzero"`
	// OutputSampleRate is the native mono PCM16LE output rate in Hz.
	OutputSampleRate int `json:"outputSampleRate,omitzero"`
	// CancelResponse reports that the cancellation probe passed.
	CancelResponse bool `json:"cancelResponse,omitzero"`
	// Tools reports whether a tool-call roundtrip passed. Flaky means the audio
	// roundtrip passed only in some runs.
	Tools TriState `json:"tools,omitzero"`
	// ReportRateLimits reports that the session supplies rate limits in usage.
	ReportRateLimits bool `json:"reportRateLimits,omitzero"`
	// ReportTokenUsage reports that session usage events include token counts.
	ReportTokenUsage TriState `json:"reportTokenUsage,omitzero"`
}

// Validate returns an error if the sample rates are negative.
func (f *RealtimeFunctionality) Validate() error {
	if f.InputSampleRate < 0 {
		return errors.New("input sample rate must not be negative")
	}
	if f.OutputSampleRate < 0 {
		return errors.New("output sample rate must not be negative")
	}
	if err := f.Tools.Validate(); err != nil {
		return fmt.Errorf("tools: %w", err)
	}
	if err := f.ReportTokenUsage.Validate(); err != nil {
		return fmt.Errorf("token usage: %w", err)
	}
	return nil
}

// ModalCapability describes how a modality is supported by a provider.
type ModalCapability struct {
	// Inline means content can be embedded directly (e.g., base64 encoded)
	Inline bool `json:"inline,omitzero"`
	// URL means content can be referenced by URL
	URL bool `json:"url,omitzero"`
	// MaxSize specifies the maximum size in bytes.
	MaxSize int64 `json:"maxSize,omitzero"`
	// SupportedFormats lists supported MIME types for this modality
	SupportedFormats []string `json:"supportedFormats,omitzero"`
}

// Reason specifies if a model Scenario supports reasoning (thinking).
type Reason int8

// Validate returns an error if the Reason is not a known value.
func (t Reason) Validate() error {
	switch t {
	case ReasonNone, ReasonInline, ReasonAuto:
		return nil
	default:
		return fmt.Errorf("invalid Reason: %d", t)
	}
}

// MarshalJSON implements json.Marshaler.
func (t Reason) MarshalJSON() ([]byte, error) {
	switch t {
	case ReasonNone:
		// TODO: Not present.
		return []byte(`"none"`), nil
	case ReasonInline:
		return []byte(`"inline"`), nil
	case ReasonAuto:
		return []byte(`"auto"`), nil
	default:
		return nil, fmt.Errorf("invalid Reason: %d", t)
	}
}

// UnmarshalJSON implements json.Unmarshaler.
func (t *Reason) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch s {
	case "none":
		*t = ReasonNone
	case "inline":
		*t = ReasonInline
	case "auto":
		*t = ReasonAuto
	default:
		return fmt.Errorf("invalid Reason: %q", s)
	}
	return nil
}

const (
	// ReasonNone means that no reasoning is supported.
	ReasonNone Reason = 0
	// ReasonInline means that the reasoning tokens are inline and must be explicitly parsed from Content.Text
	// with adapters.ProviderReasoning.
	ReasonInline Reason = 1
	// ReasonAuto means that the reasoning tokens are properly generated and handled by the provider and are
	// returned as Content.Reasoning.
	ReasonAuto Reason = -1
)

// TestdataFiles embeds the testdata/ directory for use in smoke tests.
//
// They are the canonical data to be used to declare the supported modalities.
//
//go:embed testdata/*
var TestdataFiles embed.FS
