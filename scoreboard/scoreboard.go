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

// Modality is one of the supported modalities.
type Modality string

const (
	// ModalityAudio is support for audio formats like MP3, WAV, Opus, Flac, etc.
	ModalityAudio Modality = "audio"
	// ModalityDocument is support for PDF with multi-modal comprehension, both images and text. This includes
	// code blocks.
	ModalityDocument Modality = "document"
	// ModalityImage is support for image formats like PNG, JPEG, often single frame GIF, and WEBP.
	ModalityImage Modality = "image"
	// ModalityText is for raw text.
	ModalityText Modality = "text"
	// ModalityVideo is support for video formats like MP4 or MKV.
	ModalityVideo Modality = "video"
)

func (m Modality) Validate() error {
	switch m {
	case ModalityAudio, ModalityDocument, ModalityImage, ModalityText, ModalityVideo:
		return nil
	default:
		return fmt.Errorf("invalid Modality: %q", m)
	}
}

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

	// JSON means that the model supports enforcing that the response is valid JSON but not necessarily with a
	// schema.
	JSON bool `json:"json,omitzero"`
	// JSONSchema means that the model supports enforcing that the response is a specific JSON schema.
	JSONSchema bool `json:"jsonSchema,omitzero"`
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
	// Ignore ToolsBiased and ToolsIndecisive.
	if !f.ToolCallRequired && rhs.ToolCallRequired {
		return true
	}
	if !f.JSON && rhs.JSON {
		return true
	}
	if !f.JSONSchema && rhs.JSONSchema {
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

func (f *Functionality) Validate() error {
	if err := f.ReportTokenUsage.Validate(); err != nil {
		return fmt.Errorf("invalid ReportTokenUsage: %w", err)
	}
	if err := f.ReportFinishReason.Validate(); err != nil {
		return fmt.Errorf("invalid ReportFinishReason: %w", err)
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

// TriState helps describing support when a feature "kinda work", which is frequent with LLM's inherent
// non-determinism.
type TriState int8

const (
	False TriState = 0
	True  TriState = 1
	Flaky TriState = -1
)

const triStateName = "flakyfalsetrue"

var triStateIndex = [...]uint8{0, 5, 10, 14}

func (t TriState) String() string {
	t -= -1
	if t < 0 || t >= TriState(len(triStateIndex)-1) {
		return "TriState(" + strconv.FormatInt(int64(t+-1), 10) + ")"
	}
	return triStateName[triStateIndex[t]:triStateIndex[t+1]]
}

func (t TriState) GoString() string {
	return t.String()
}

func (t TriState) Validate() error {
	switch t {
	case False, True, Flaky:
		return nil
	default:
		return fmt.Errorf("invalid TriState: %q", t)
	}
}

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

// Scenario defines one way to use the provider.
type Scenario struct {
	// Comments are notes about the scenario. For example, if a scenario is known to be bugged, deprecated,
	// expensive, etc.
	Comments string `json:"comments,omitzero"`
	// Models is a *non exhaustive* list of models that support this scenario. It can't be exhaustive since
	// providers continuouly release new models. It is still valuable to use the first value. Required.
	Models []string `json:"models"`

	// These mean that the model is automatically selected. There can be multiple SOTA models, one per
	// modalities (e.g. text output vs image output). They must be first. Models must be a list of one model in
	// this case.
	SOTA  bool `json:"sota,omitzero"`
	Good  bool `json:"good,omitzero"`
	Cheap bool `json:"cheap,omitzero"`

	// Reason means that the model does either explicit chain-of-thought or hidden reasoning. For some
	// providers, this is controlled via a OptionsText. For some models (like Qwen3), a token "/no_think" or
	// "/think" is used to control. ReasoningTokenStart and ReasoningTokenEnd must only be set on explicit inline
	// reasoning models. They often use <think> and </think>.
	Reason              bool   `json:"reason,omitzero"`
	ReasoningTokenStart string `json:"reasoningTokenStart,omitzero"`
	ReasoningTokenEnd   string `json:"reasoningTokenEnd,omitzero"`

	In  map[Modality]ModalCapability `json:"in,omitzero,omitempty"`
	Out map[Modality]ModalCapability `json:"out,omitzero,omitempty"`

	// GenSync declares features supported when using Provider.GenSync
	GenSync *Functionality `json:"GenSync,omitzero,omitempty"`
	// GenStream declares features supported when using Provider.GenStream
	GenStream *Functionality `json:"GenStream,omitzero,omitempty"`

	_ struct{}
}

// Untested returns true if the scenario has no test results.
func (s *Scenario) Untested() bool {
	return s.GenSync == nil && s.GenStream == nil && len(s.In) == 0 && len(s.Out) == 0
}

func (s *Scenario) Validate() error {
	if len(s.Models) == 0 {
		return errors.New("scenario must have at least one model")
	}
	for k := range s.In {
		if err := k.Validate(); err != nil {
			return err
		}
	}
	for k := range s.Out {
		if err := k.Validate(); err != nil {
			return err
		}
	}
	if (len(s.In) == 0) != (len(s.Out) == 0) {
		return errors.New("scenario must have either both or none of In or Out")
	}
	if len(s.In) == 0 && (s.GenSync != nil || s.GenStream != nil) {
		return errors.New("scenario must have be defined to have either GenSync or GenStream")
	}
	if s.GenSync != nil {
		if err := s.GenSync.Validate(); err != nil {
			return err
		}
	}
	if s.GenStream != nil {
		if err := s.GenStream.Validate(); err != nil {
			return err
		}
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

func (t Reason) Validate() error {
	switch t {
	case ReasonNone, ReasonInline, ReasonAuto:
		return nil
	default:
		return fmt.Errorf("invalid Reason: %q", t)
	}
}

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
		return nil, fmt.Errorf("invalid Reason: %q", t)
	}
}

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

// Score is a snapshot of the capabilities of the provider. These are smoke tested to confirm the
// accuracy.
type Score struct {
	// Warnings lists concerns the user should be aware of.
	Warnings []string `json:"warnings,omitzero,omitempty"`
	// Country where the provider is based, e.g. "US", "CN", "EU". Two exceptions: "Local" for local and "N/A"
	// for pure routers.
	Country string `json:"country"`
	// DashboardURL is the URL to the provider's dashboard, if available.
	DashboardURL string `json:"dashboardURL"`

	// Scenarios is the list of all known supported and tested scenarios.
	//
	// A single provider can provide various distinct use cases, like text-to-text, multi-modal-to-text,
	// text-to-audio, audio-to-text, etc.
	Scenarios []Scenario `json:"scenarios"`

	_ struct{}
}

func (s *Score) Validate() error {
	// Check for duplicate model/reason pairs
	seen := make(map[Model]struct{})
	for _, sc := range s.Scenarios {
		if err := sc.Validate(); err != nil {
			return err
		}
		for _, model := range sc.Models {
			key := Model{model, sc.Reason}
			if _, ok := seen[key]; ok {
				return fmt.Errorf("duplicate model in scoreboard: %q (reason=%v)", model, sc.Reason)
			}
			seen[key] = struct{}{}
		}
	}

	// Only validate ordering if there are multiple scenarios
	if len(s.Scenarios) <= 1 {
		return nil
	}

	// Find first and count occurrences of SOTA, Good, Cheap models
	countSOTA, countGood, countCheap := 0, 0, 0
	firstSOTA, firstGood, firstCheap := -1, -1, -1
	for i, sc := range s.Scenarios {
		if len(sc.Models) == 0 {
			continue
		}
		if sc.SOTA {
			countSOTA++
			if firstSOTA < 0 {
				firstSOTA = i
			}
		}
		if sc.Good {
			countGood++
			if firstGood < 0 {
				firstGood = i
			}
		}
		if sc.Cheap {
			countCheap++
			if firstCheap < 0 {
				firstCheap = i
			}
		}
	}

	if firstSOTA < 0 {
		return errors.New("no SOTA model marked")
	}
	if countSOTA > 1 {
		return fmt.Errorf("multiple SOTA models marked (count: %d)", countSOTA)
	}
	if firstGood < 0 {
		return errors.New("no Good model marked")
	}
	if countGood > 1 {
		return fmt.Errorf("multiple Good models marked (count: %d)", countGood)
	}
	if firstCheap < 0 {
		return errors.New("no Cheap model marked")
	}
	if countCheap > 1 {
		return fmt.Errorf("multiple Cheap models marked (count: %d)", countCheap)
	}

	// SOTA must be first
	if firstSOTA != 0 {
		return fmt.Errorf("SOTA model %q should be first (at position %d)", s.Scenarios[firstSOTA].Models[0], firstSOTA)
	}

	// Good should come before Cheap
	if firstGood > firstCheap {
		return fmt.Errorf("Good model comes after Cheap model (good at %d, cheap at %d)", firstGood, firstCheap)
	}

	return nil
}

// TestdataFiles embeds the testdata/ directory for use in smoke tests.
//
// They are the canonical data to be used to declare the supported modalities.
//
//go:embed testdata/*
var TestdataFiles embed.FS
