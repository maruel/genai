// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package scoreboard contains all dynamic scoreboard creation and gathering logic.
package scoreboard

import (
	"embed"
	"net/http"
	"strconv"

	"github.com/maruel/genai"
)

//

// ProviderScore describes the known state of the provider.
type ProviderScore interface {
	genai.Provider
	// Scoreboard returns what the provider supports.
	//
	// Some models have more features than others, e.g. some models may be text-only while others have vision or
	// audio support.
	//
	// The client code may be the limiting factor for some models, and not the provider itself.
	//
	// The values returned here are gone through a smoke test to make sure they are valid.
	Scoreboard() Score
}

// FunctionalityText defines which functionalites are supported in a scenario for models that support text
// output modality.
//
// The first group is for multi-modal models, either with non-text inputs (e.g. vision, STT) or outputs
// (combined text and image generation).
//
// The second group are supported functional features for agency.
//
// The third group is to identify bugged providers. A provider is considered to be bugged if any of the field
// is false.
type FunctionalityText struct {
	// Tools means that tool call is supported. This is a requirement for MCP. Some provider support tool
	// calling but the model is very flaky at actually requesting the calls. This is more frequent on highly
	// quantized models, small models or MoE models.
	Tools TriState
	// JSON means that the model supports enforcing that the response is valid JSON but not necessarily with a
	// schema.
	JSON bool
	// JSONSchema means that the model supports enforcing that the response is a specific JSON schema.
	JSONSchema bool
	// Citations is set when the provider and model combination supports citations in the response.
	Citations bool
	// Seed is set when the provider and model combination supports seed for reproducibility.
	Seed bool
	// TopLogprobs is set when the provider and model combination supports top_logprobs.
	TopLogprobs bool

	// ReportRateLimits means that the provider reports rate limits in its Usage.
	ReportRateLimits bool
	// BrokenTokenUsage means that the usage is not correctly reported.
	BrokenTokenUsage TriState
	// BrokenFinishReason means that the finish reason (FinishStop, FinishLength, etc) is not correctly reported.
	BrokenFinishReason bool
	// NoMaxTokens means that the provider doesn't support limiting text output. Only relevant on text output.
	NoMaxTokens bool
	// NoStopSequence means that the provider doesn't support stop words. Only relevant on text output.
	NoStopSequence bool
	// BiasedTool is true when we ask the LLM to use a tool in an ambiguous biased question, it will always
	// reply with the first readily available answer.
	//
	// This means that when using enum, it is important to understand that the LLM will put heavy weight on the
	// first option.
	//
	// This is affected by two factors: model size and quantization. Quantization affects this dramatically.
	BiasedTool TriState
	// IndecisiveTool is True when we ask the LLM to use a tool in an ambiguous biased question, it'll call both
	// options. It is Flaky when both can happen.
	//
	// This is actually fine, it means that the LLM will be less opinionated in some cases. The test into which
	// a LLM is indecisive is likely model-specific too.
	IndecisiveTool TriState

	_ struct{}
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

func (i TriState) String() string {
	i -= -1
	if i < 0 || i >= TriState(len(triStateIndex)-1) {
		return "TriState(" + strconv.FormatInt(int64(i+-1), 10) + ")"
	}
	return triStateName[triStateIndex[i]:triStateIndex[i+1]]
}

func (i TriState) GoString() string {
	return i.String()
}

// FunctionalityDoc defines which functionalites are supported in a scenario for non-text output modality.
type FunctionalityDoc struct {
	// Seed is set when the provider and model combination supports seed for reproducibility.
	Seed bool

	// ReportRateLimits means that the provider reports rate limits in its Usage.
	ReportRateLimits bool
	// BrokenTokenUsage means that the usage is not correctly reported.
	BrokenTokenUsage TriState
	// BrokenFinishReason means that the finish reason (FinishStop, FinishLength, etc) is not correctly reported.
	BrokenFinishReason bool

	_ struct{}
}

// Scenario defines one way to use the provider.
type Scenario struct {
	In  map[genai.Modality]ModalCapability
	Out map[genai.Modality]ModalCapability
	// Models is a *non exhaustive* list of models that support this scenario. It can't be exhaustive since
	// providers continuouly release new models. It is still valuable to use the first value
	Models []string

	// Thinking means that the model does either explicit chain-of-thought or hidden thinking. For some
	// providers, this is controlled via a OptionsText. For some models (like Qwen3), a token "/no_think" or
	// "/think" is used to control. ThinkingTokenStart and ThinkingTokenEnd must only be set on explicit inline
	// thinking models. They often use <think> and </think>.
	Thinking           bool
	ThinkingTokenStart string
	ThinkingTokenEnd   string

	// GenSync declares features supported when using ProviderGen.GenSync
	GenSync *FunctionalityText
	// GenStream declares features supported when using ProviderGen.GenStream
	GenStream *FunctionalityText
	// GenDoc declares features supported when using a ProviderGenDoc
	GenDoc *FunctionalityDoc

	_ struct{}
}

// ModalCapability describes how a modality is supported by a provider.
type ModalCapability struct {
	// Inline means content can be embedded directly (e.g., base64 encoded)
	Inline bool
	// URL means content can be referenced by URL
	URL bool
	// MaxSize specifies the maximum size in bytes.
	MaxSize int64
	// SupportedFormats lists supported MIME types for this modality
	SupportedFormats []string
}

// Thinking specifies if a model Scenario supports thinking.
type Thinking int8

const (
	// NoThinking means that no thinking is supported.
	NoThinking Thinking = 0
	// ThinkingInline means that the thinking tokens are inline and must be explicitly parsed from Content.Text
	// with adapters.ProviderGenThinking.
	ThinkingInline Thinking = 1
	// ThinkingAutomatic means that the thinking tokens are properly generated and handled by the provider and
	// are returned as Content.Thinking.
	ThinkingAutomatic Thinking = -1
)

// Score is a snapshot of the capabilities of the provider. These are smoke tested to confirm the
// accuracy.
type Score struct {
	// Scenarios is the list of all known supported and tested scenarios.
	//
	// A single provider can provide various distinct use cases, like text-to-text, multi-modal-to-text,
	// text-to-audio, audio-to-text, etc.
	Scenarios []Scenario

	// Country where the provider is based, e.g. "US", "CN", "EU". Two exceptions: "Local" for local and "N/A"
	// for pure routers.
	Country string
	// DashboardURL is the URL to the provider's dashboard, if available.
	DashboardURL string

	_ struct{}
}

//go:embed testdata/*
var testdataFiles embed.FS

// ProviderFactory is a function that returns a provider instance. The name represents the sub-test name.
//
// This may be used for HTTP recording and replays.
type ProviderFactory func(name string) (genai.Provider, http.RoundTripper)
