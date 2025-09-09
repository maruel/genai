// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

// Options is options that can be provided to a Provider interface.
type Options interface {
	// Validate ensures the options object is valid.
	Validate() error
}

// Modality is a modality supported by the provider.
//
// It is aliased from scoreboard.Modality to avoid a circular dependency.
type Modality = scoreboard.Modality

const (
	// ModalityAudio is support for audio formats like MP3, WAV, Opus, Flac, etc.
	ModalityAudio = scoreboard.ModalityAudio
	// ModalityDocument is support for PDF with multi-modal comprehension, both images and text. This includes
	// code blocks.
	ModalityDocument = scoreboard.ModalityDocument
	// ModalityImage is support for image formats like PNG, JPEG, often single frame GIF, and WEBP.
	ModalityImage = scoreboard.ModalityImage
	// ModalityText is for raw text.
	ModalityText = scoreboard.ModalityText
	// ModalityVideo is support for video formats like MP4 or MKV.
	ModalityVideo = scoreboard.ModalityVideo
)

// Modalities represents the modality supported by the provider in a specific scenario. It can be multiple
// modalities in multi-modals scenarios.
type Modalities []scoreboard.Modality

func (m Modalities) String() string {
	switch len(m) {
	case 0:
		return ""
	case 1:
		return string(m[0])
	default:
		// Inline strings.Join()
		n := (len(m) - 1)
		for _, elem := range m {
			n += len(elem)
		}
		var b strings.Builder
		b.Grow(n)
		b.WriteString(string(m[0]))
		for _, s := range m[1:] {
			b.WriteString(",")
			b.WriteString(string(s))
		}
		return b.String()
	}
}

func (m Modalities) Validate() error {
	for _, mod := range m {
		if err := mod.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// OptionsText is a list of frequent options supported by most Provider with text output modality.
// Each provider is free to support more options through a specialized struct.
//
// The first group are options supported by (nearly) all providers.
//
// The second group are options supported only by some providers. Using them may cause the chat operation to
// return a base.ErrNotsupported.
//
// The third group are options supported by a few providers and a few models on each, that will slow down
// generation (increase latency) and will increase token use (cost).
type OptionsText struct {
	// Temperature adjust the creativity of the sampling. Generally between 0 and 2.
	Temperature float64
	// TopP adjusts correctness sampling between 0 and 1. The higher the more diverse the output.
	TopP float64
	// MaxTokens is the maximum number of tokens to generate. Used to limit it
	// lower than the default maximum, for budget reasons.
	MaxTokens int64
	// TopLogprobs requests to return the top logprobs in the reply.
	TopLogprobs int64
	// SystemPrompt is the prompt to use for the system role.
	SystemPrompt string

	// Seed for the random number generator. Default is 0 which means
	// non-deterministic.
	Seed int64
	// TopK adjusts sampling where only the N first candidates are considered.
	TopK int64
	// Stop is the list of tokens to stop generation.
	Stop []string

	// ReplyAsJSON enforces the output to be valid JSON, any JSON. It is
	// important to tell the model to reply in JSON in the prompt itself.
	ReplyAsJSON bool
	// DecodeAs enforces a reply with a specific JSON structure. It must be a pointer to a struct that can be
	// decoded by encoding/json and can have jsonschema tags.
	//
	// It is important to request the model to "reply in JSON" in the prompt itself.
	//
	// It is recommended to use jsonschema_description tags to describe each
	// field or argument.
	//
	// Use jsonschema:"enum=..." to enforce a specific value within a set.
	//
	// Use omitempty to make the field optional.
	//
	// See https://github.com/invopop/jsonschema#example for more examples.
	DecodeAs any

	_ struct{}
}

// Validate ensures the completion options are valid.
func (o *OptionsText) Validate() error {
	if o.Seed < 0 {
		return errors.New("field Seed: must be non-negative")
	}
	if o.Temperature < 0 || o.Temperature > 100 {
		return errors.New("field Temperature: must be [0, 100]")
	}
	if o.MaxTokens < 0 || o.MaxTokens > 1024*1024*1024 {
		return errors.New("field MaxTokens: must be [0, 1 GiB]")
	}
	if o.TopP < 0 || o.TopP > 1 {
		return errors.New("field TopP: must be [0, 1]")
	}
	if o.TopK < 0 || o.TopK > 1024 {
		return errors.New("field TopK: must be [0, 1024]")
	}
	if o.DecodeAs != nil {
		if err := validateReflectedToJSON(o.DecodeAs); err != nil {
			return fmt.Errorf("field DecodeAs: %w", err)
		}
	}
	return nil
}

// Tools

type OptionsTools struct {
	// Tools is the list of tools that the LLM can request to call.
	Tools []ToolDef
	// Force tells the LLM a tool call must be done, or not.
	Force ToolCallRequest
	// WebSearch specifies if websearch should be enabled. It is generally disabled by default except for
	// perplexity.
	//
	// # Warning
	//
	// This will become a structure to provide information about included and excluded domains, and the user's
	// location.
	WebSearch bool
}

// Validate ensures the completion options are valid.
func (o *OptionsTools) Validate() error {
	names := map[string]int{}
	for i, t := range o.Tools {
		if err := t.Validate(); err != nil {
			return fmt.Errorf("tool %d: %w", i, err)
		}
		if j, ok := names[t.Name]; ok {
			return fmt.Errorf("tool %d: has name %q which is the same as tool %d", i, t.Name, j)
		}
		names[t.Name] = i
	}
	if len(o.Tools) == 0 && !o.WebSearch && o.Force == ToolCallRequired {
		return fmt.Errorf("field Force is ToolCallRequired: Tools are required")
	}
	return nil
}

// ToolDef describes a tool that the LLM can request to use.
type ToolDef struct {
	// Name must be unique among all tools.
	Name string
	// Description must be a LLM-friendly short description of the tool.
	Description string
	// Callback is the function to call with the inputs.
	// It must accept a context.Context one struct pointer as input: (ctx context.Context, input *struct{}). The
	// struct must use json_schema to be serializable as JSON.
	// It must return the result and an error: (string, error).
	Callback any
	// InputSchemaOverride overrides the schema deduced from the Callback's second argument. It's meant to be
	// used when an enum or a description is set dynamically, or with complex if/then/else that would be tedious
	// to describe as struct tags.
	//
	// It is okay to initialize Callback, then take the return value of GetInputSchema() to initialize InputSchemaOverride, then mutate it.
	InputSchemaOverride *jsonschema.Schema

	_ struct{}
}

var reToolName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// Validate ensures the tool definition is valid.
//
// For the Name field, it uses the rule according to
// https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implement-tool-use#example-simple-tool-definition
func (t *ToolDef) Validate() error {
	if !reToolName.MatchString(t.Name) {
		return errors.New("field Name: must be a valid tool name between 1 and 64 characters and contain only [a-zA-Z0-9_-]")
	}
	if t.Description == "" {
		return errors.New("field Description: required")
	}
	if t.Callback != nil {
		cbType := reflect.TypeOf(t.Callback)
		if cbType.Kind() != reflect.Func {
			return errors.New("field Callback: must be a function")
		}
		if cbType.NumIn() != 2 {
			return errors.New("field Callback: must accept exactly two parameters: (context.Context, input *struct{})")
		}
		paramType := cbType.In(0)
		if paramType != reflect.TypeFor[context.Context]() {
			return fmt.Errorf("field Callback: must accept exactly two parameters, first that is a context.Context, not a %q", paramType.Name())
		}
		paramType = cbType.In(1)
		if paramType.Kind() != reflect.Pointer {
			return fmt.Errorf("field Callback: must accept exactly two parameters, second that is a pointer to a struct, not a %q", paramType.Name())
		}
		paramType = paramType.Elem()
		if paramType.Kind() != reflect.Struct {
			return fmt.Errorf("field Callback: must accept exactly two parameters, second that is a pointer to a struct, not a %q", paramType.Name())
		}
		if err := validateReflectedToJSON(paramType); err != nil {
			return fmt.Errorf("field Callback: must accept exactly two parameters, second that is a pointer to a struct that has valid json schema: %w", err)
		}
		if cbType.NumOut() != 2 {
			return errors.New("field Callback: must return exactly two values: (string, error)")
		}
		if cbType.Out(0).Kind() != reflect.String {
			return fmt.Errorf("field Callback: must return a string first, not %q", cbType.Out(0).Name())
		}
		if !isErrorType(cbType.Out(1)) {
			return fmt.Errorf("field Callback: must return an error second, not %q", cbType.Out(1).Name())
		}
	}
	return nil
}

// GetInputSchema returns the json schema for the input argument of the callback.
func (t *ToolDef) GetInputSchema() *jsonschema.Schema {
	// This function assumes Validate() was called.
	return internal.JSONSchemaFor(reflect.TypeOf(t.Callback).In(1))
}

// ToolCallRequest determines if we want the LLM to request a tool call.
type ToolCallRequest int

const (
	// ToolCallAny is the default, the model is free to choose if a tool is called or not. For some models (like
	// llama family), it may be a bit too "tool call happy".
	ToolCallAny ToolCallRequest = iota
	// ToolCallRequired means a tool call is required. Don't forget to change the value after sending the
	// response!
	ToolCallRequired
	// ToolCallNone means that while tools are described, they should not be called. It is useful when a LLM did
	// tool calls, got the response and now it's time to generate some text to present to the end user.
	ToolCallNone
)

// Other modalities

type OptionsAudio struct {
	// Seed for the random number generator. Default is 0 which means
	// non-deterministic.
	Seed int64

	_ struct{}
}

func (o *OptionsAudio) Validate() error {
	if o.Seed < 0 {
		return errors.New("field Seed: must be non-negative")
	}
	return nil
}

// OptionsImage is a list of frequent options supported by most ProviderDoc.
// Each provider is free to support more options through a specialized struct.
type OptionsImage struct {
	// Seed for the random number generator. Default is 0 which means
	// non-deterministic.
	Seed   int64
	Width  int
	Height int

	// PollInterval is the time interval to poll the image generation progress when using GenSync.
	PollInterval time.Duration

	_ struct{}
}

// Validate ensures the completion options are valid.
func (o *OptionsImage) Validate() error {
	if o.Seed < 0 {
		return errors.New("field Seed: must be non-negative")
	}
	if o.Height < 0 {
		return errors.New("field Height: must be non-negative")
	}
	if o.Width < 0 {
		return errors.New("field Width: must be non-negative")
	}
	return nil
}

type OptionsVideo struct {
	// Duration of the video to generate, if supported.
	//
	// Veo 2 supports only between 5 and 8 seconds and Veo 3 only supports 8 seconds.
	Duration time.Duration

	// PollInterval is the time interval to poll the image generation progress when using GenSync.
	PollInterval time.Duration

	_ struct{}
}

func (o *OptionsVideo) Validate() error {
	return nil
}

// Private

func validateReflectedToJSON(r any) error {
	tp := reflect.TypeOf(r)
	if tp.Kind() == reflect.Pointer {
		tp = tp.Elem()
		if _, ok := r.(*jsonschema.Schema); ok {
			return errors.New("must be an actual struct serializable as JSON, not a *jsonschema.Schema")
		}
	}
	if tp.Kind() != reflect.Struct {
		return fmt.Errorf("must be a struct, not %T", r)
	}
	return nil
}

// isErrorType returns true if the type is of error type.
func isErrorType(t reflect.Type) bool {
	return t == reflect.TypeOf((*error)(nil)).Elem()
}

var (
	_ Options              = (*OptionsAudio)(nil)
	_ Options              = (*OptionsImage)(nil)
	_ Options              = (*OptionsVideo)(nil)
	_ Options              = (*OptionsText)(nil)
	_ Options              = (*OptionsTools)(nil)
	_ internal.Validatable = (*Modality)(nil)
	_ internal.Validatable = (*Modalities)(nil)
	_ internal.Validatable = (*ToolDef)(nil)
)
