// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package scoreboard contains all dynamic scoreboard creation and gathering logic.
package scoreboard

import (
	"embed"
	"encoding/json"
	"fmt"
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
	Tools TriState `json:"tools,omitzero"`
	// JSON means that the model supports enforcing that the response is valid JSON but not necessarily with a
	// schema.
	JSON bool `json:"json,omitzero"`
	// JSONSchema means that the model supports enforcing that the response is a specific JSON schema.
	JSONSchema bool `json:"jsonSchema,omitzero"`
	// Citations is set when the provider and model combination supports citations in the response.
	Citations bool `json:"citations,omitzero"`
	// Seed is set when the provider and model combination supports seed for reproducibility.
	Seed bool `json:"seed,omitzero"`
	// TopLogprobs is set when the provider and model combination supports top_logprobs.
	TopLogprobs bool `json:"topLogprobs,omitzero"`
	// MaxTokens means that the provider supports limiting text output to a specific number of tokens.
	//
	// Tokens are characters nor words. The tokens are embedding specific, and each model family uses a
	// different vocabulary. Thus the number of characters generated varies wildly.
	//
	// It fails more often with model with implicit thinking.
	MaxTokens bool `json:"maxTokens,omitzero"`
	// StopSequence means that the provider supports stop words. The number of stop words is generally limited,
	// frequently to 5 words. The sequence should be a valid token in the model's vocabulary.
	StopSequence bool `json:"stopSequence,omitzero"`

	// ReportRateLimits means that the provider reports rate limits in its Usage.
	ReportRateLimits bool `json:"reportRateLimits,omitzero"`
	// ReportTokenUsage means that the token usage is correctly reported in all cases. It is flaky if it is not
	// reported in some specific cases. A frequent example is tokens not being reported in JSON output mode.
	ReportTokenUsage TriState `json:"reportTokenUsage,omitzero"`
	// ReportFinishReason means that the finish reason (FinishStop, FinishLength, etc) is not correctly reported.
	ReportFinishReason TriState `json:"reportFinishReason,omitzero"`
	// BiasedTool is true when we ask the LLM to use a tool in an ambiguous biased question, it will always
	// reply with the first readily available answer.
	//
	// This means that when using enum, it is important to understand that the LLM will put heavy weight on the
	// first option.
	//
	// This is affected by two factors: model size and quantization. Quantization affects this dramatically.
	BiasedTool TriState `json:"biasedTool,omitzero"`
	// IndecisiveTool is True when we ask the LLM to use a tool in an ambiguous biased question, it'll call both
	// options. It is Flaky when both can happen.
	//
	// This is actually fine, it means that the LLM will be less opinionated in some cases. The test into which
	// a LLM is indecisive is likely model-specific too.
	IndecisiveTool TriState `json:"indecisiveTool,omitzero"`

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

func (i TriState) MarshalJSON() ([]byte, error) {
	switch i {
	case False:
		// TODO: Not present.
		return []byte(`"false"`), nil
	case True:
		return []byte(`"true"`), nil
	case Flaky:
		return []byte(`"flaky"`), nil
	default:
		return nil, fmt.Errorf("invalid TriState: %q", i)
	}
}

func (i *TriState) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch s {
	case `false`:
		*i = False
	case `true`:
		*i = True
	case `flaky`:
		*i = Flaky
	default:
		return fmt.Errorf("invalid TriState: %q", s)
	}
	return nil
}

// FunctionalityDoc defines which functionalites are supported in a scenario for non-text output modality.
type FunctionalityDoc struct {
	// Seed is set when the provider and model combination supports seed for reproducibility.
	Seed bool `json:"seed,omitzero"`

	// ReportRateLimits means that the provider reports rate limits in its Usage.
	ReportRateLimits bool `json:"reportRateLimits,omitzero"`
	// ReportTokenUsage means that the token usage is correctly reported in all cases. It is flaky if it is not
	// reported in some specific cases. A frequent example is tokens not being reported in JSON output mode.
	ReportTokenUsage TriState `json:"reportTokenUsage,omitzero"`
	// BrokenFinishReason means that the finish reason (FinishStop, FinishLength, etc) is not correctly reported.
	ReportFinishReason TriState `json:"reportFinishReason,omitzero"`

	_ struct{}
}

// Scenario defines one way to use the provider.
type Scenario struct {
	// Comments are notes about the scenario. For example, if a scenario is known to be bugged, deprecated,
	// expensive, etc.
	Comments string `json:"comments,omitzero"`
	// Models is a *non exhaustive* list of models that support this scenario. It can't be exhaustive since
	// providers continuouly release new models. It is still valuable to use the first value
	Models []string `json:"models"`

	// Thinking means that the model does either explicit chain-of-thought or hidden thinking. For some
	// providers, this is controlled via a OptionsText. For some models (like Qwen3), a token "/no_think" or
	// "/think" is used to control. ThinkingTokenStart and ThinkingTokenEnd must only be set on explicit inline
	// thinking models. They often use <think> and </think>.
	Thinking           bool   `json:"thinking,omitzero"`
	ThinkingTokenStart string `json:"thinkingTokenStart,omitzero"`
	ThinkingTokenEnd   string `json:"thinkingTokenEnd,omitzero"`

	In  map[genai.Modality]ModalCapability `json:"in,omitzero"`
	Out map[genai.Modality]ModalCapability `json:"out,omitzero"`

	// GenSync declares features supported when using Provider.GenSync
	GenSync *FunctionalityText `json:"GenSync,omitzero,omitempty"`
	// GenStream declares features supported when using Provider.GenStream
	GenStream *FunctionalityText `json:"GenStream,omitzero,omitempty"`
	// GenDoc declares features supported when using a non-text output modality.
	GenDoc *FunctionalityDoc `json:"GenDoc,omitzero,omitempty"`

	_ struct{}
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

// Thinking specifies if a model Scenario supports thinking.
type Thinking int8

const (
	// ThinkingNone means that no thinking is supported.
	ThinkingNone Thinking = 0
	// ThinkingInline means that the thinking tokens are inline and must be explicitly parsed from Content.Text
	// with adapters.ProviderThinking.
	ThinkingInline Thinking = 1
	// ThinkingAuto means that the thinking tokens are properly generated and handled by the provider and
	// are returned as Content.Thinking.
	ThinkingAuto Thinking = -1
)

func (i Thinking) MarshalJSON() ([]byte, error) {
	switch i {
	case ThinkingNone:
		// TODO: Not present.
		return []byte(`"none"`), nil
	case ThinkingInline:
		return []byte(`"inline"`), nil
	case ThinkingAuto:
		return []byte(`"auto"`), nil
	default:
		return nil, fmt.Errorf("invalid Thinking: %q", i)
	}
}

func (i *Thinking) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	switch s {
	case `none`:
		*i = ThinkingNone
	case `inline`:
		*i = ThinkingInline
	case `auto`:
		*i = ThinkingAuto
	default:
		return fmt.Errorf("invalid Thinking: %q", s)
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
	type pair struct {
		name     string
		thinking bool
	}
	seen := map[pair]struct{}{}
	for _, sc := range s.Scenarios {
		for _, model := range sc.Models {
			k := pair{name: model, thinking: sc.Thinking}
			if _, ok := seen[k]; ok {
				return fmt.Errorf("duplicate model in scoreboard: %v", k)
			}
			seen[k] = struct{}{}
		}
	}
	return nil
}

//go:embed testdata/*
var testdataFiles embed.FS

// ProviderFactory is a function that returns a provider instance. The name represents the sub-test name.
//
// This may be used for HTTP recording and replays.
type ProviderFactory func(name string) (genai.Provider, http.RoundTripper)

var _ genai.Validatable = (*Score)(nil)
