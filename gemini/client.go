// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package gemini implements a client for Google's Gemini API.
//
// Not to be confused with Google's Vertex AI.
//
// It is described at https://ai.google.dev/api/?lang=rest but the doc is weirdly organized.
package gemini

// See official client at https://github.com/google/generative-ai-go

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"mime"
	"net/http"
	"os"
	"path"
	"reflect"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Gemini.
//
// # Warnings
//
//   - Gemini supports basically anything, but often on "preview" and "experimental" models. This means that
//     the hardcoded model names in this scoreboard will have to be updated once stable models are released.
//   - Gemini removed thinking in January 2025 and announced they will add a summarized version within 2025.
//   - Not all features supported by Gemini are implemented, there's just so many.
//   - Tool calling is excellent and unbiased for non "-lite" models.
//   - Files can be referenced by URL but only if they have been uploaded via the file API, which is not
//     implemented yet.
//   - Rate limit is based on how much you spend per month: https://ai.google.dev/gemini-api/docs/rate-limits
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:     []genai.Modality{genai.ModalityText},
			Out:    []genai.Modality{genai.ModalityText},
			Models: []string{"gemini-2.0-flash-lite"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              genai.True,
				UnbiasedTool:       false,
				JSON:               true,
				JSONSchema:         true,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              genai.True,
				UnbiasedTool:       false,
				JSON:               true,
				JSONSchema:         true,
			},
		},
		{
			In:     []genai.Modality{genai.ModalityImage, genai.ModalityPDF, genai.ModalityAudio, genai.ModalityVideo, genai.ModalityText},
			Out:    []genai.Modality{genai.ModalityText},
			Models: []string{"gemini-2.0-flash"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              genai.True,
				UnbiasedTool:       true,
				JSON:               true,
				JSONSchema:         true,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              genai.True,
				UnbiasedTool:       true,
				JSON:               true,
				JSONSchema:         true,
			},
		},
		{
			In:     []genai.Modality{genai.ModalityText},
			Out:    []genai.Modality{genai.ModalityText, genai.ModalityImage},
			Models: []string{"gemini-2.0-flash-preview-image-generation"},
			Chat: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           true,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              genai.True,
				UnbiasedTool:       true,
				JSON:               true,
				JSONSchema:         true,
			},
			ChatStream: genai.Functionality{
				Inline:             true,
				URL:                false,
				Thinking:           true,
				ReportTokenUsage:   true,
				ReportFinishReason: true,
				MaxTokens:          true,
				StopSequence:       true,
				Tools:              genai.True,
				UnbiasedTool:       true,
				JSON:               true,
				JSONSchema:         true,
			},
		},
	},
}

// ChatOptions includes Gemini specific options.
type ChatOptions struct {
	genai.ChatOptions

	// ThinkingBudget is the maximum number of tokens the LLM can use to think about the answer. When 0,
	// thinking is disabled. It generally must be above 1024 and below MaxTokens and 24576.
	ThinkingBudget int64

	// ResponseModalities defines what the LLM can return, text, images or audio. Default to text.
	//
	// https://ai.google.dev/gemini-api/docs/image-generation
	ResponseModalities []Modality
}

// https://ai.google.dev/api/caching?hl=en#Blob
type Blob struct {
	MimeType string `json:"mimeType,omitzero"`
	Data     []byte `json:"data,omitzero"`
}

// https://ai.google.dev/api/caching?hl=en#FileData
type FileData struct {
	MimeType string `json:"mimeType,omitzero"`
	FileURI  string `json:"fileUri,omitzero"`
}

// https://protobuf.dev/reference/protobuf/google.protobuf/#struct
type StructValue map[string]any

// https://protobuf.dev/reference/protobuf/google.protobuf/#value
type Value struct {
	NullValue   int64       `json:"null_value,omitzero"`
	NumberValue float64     `json:"number_value,omitzero"`
	StringValue string      `json:"string_value,omitzero"`
	BoolValue   bool        `json:"bool_value,omitzero"`
	StructValue StructValue `json:"struct_value,omitzero"`
	ListValue   []Value     `json:"list_value,omitzero"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.SafetySetting
type SafetySetting struct {
	Category  string `json:"category"`  // https://ai.google.dev/api/generate-content?hl=en#v1beta.HarmCategory
	Threshold int64  `json:"threshold"` // https://ai.google.dev/api/generate-content?hl=en#HarmBlockThreshold
}

// https://ai.google.dev/api/caching?hl=en#Tool
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"functionDeclarations,omitzero"`
	// https://ai.google.dev/api/caching?hl=en#GoogleSearchRetrieval
	GoogleSearchRetrieval struct {
		// https://ai.google.dev/api/caching?hl=en#DynamicRetrievalConfig
		DynamicRetrievalConfig struct {
			// https://ai.google.dev/api/caching?hl=en#Mode
			Mode             string  `json:"mode,omitzero"` // MODE_UNSPECIFIED, MODE_DYNAMIC
			DynamicThreshold float64 `json:"dynamicThreshold,omitzero"`
		} `json:"dynamicRetrievalConfig,omitzero"`
	} `json:"googleSearchRetrieval,omitzero"`
	CodeExecution struct{} `json:"codeExecution,omitzero"`
	GoogleSearch  struct{} `json:"googleSearch,omitzero"`
}

// https://ai.google.dev/api/caching?hl=en#FunctionDeclaration
type FunctionDeclaration struct {
	Name        string `json:"name,omitzero"`
	Description string `json:"description,omitzero"`
	Parameters  Schema `json:"parameters,omitzero"`
	Response    Schema `json:"response,omitzero"`
}

// https://ai.google.dev/api/caching?hl=en#Schema
type Schema struct {
	// https://ai.google.dev/api/caching?hl=en#Type
	// https://spec.openapis.org/oas/v3.0.3#data-types
	Type             string            `json:"type,omitzero"`             // TYPE_UNSPECIFIED, STRING, NUMBER, INTEGER, BOOLEAN, ARRAY, OBJECT
	Format           string            `json:"format,omitzero"`           // NUMBER type: float, double for INTEGER type: int32, int64 for STRING type: enum, date-time
	Description      string            `json:"description,omitzero"`      //
	Nullable         bool              `json:"nullable,omitzero"`         //
	Enum             []string          `json:"enum,omitzero"`             // STRING
	MaxItems         int64             `json:"maxItems,omitzero"`         // ARRAY
	MinItems         int64             `json:"minItems,omitzero"`         // ARRAY
	Properties       map[string]Schema `json:"properties,omitzero"`       // OBJECT
	Required         []string          `json:"required,omitzero"`         // OBJECT
	PropertyOrdering []string          `json:"propertyOrdering,omitzero"` // OBJECT
	Items            *Schema           `json:"items,omitzero"`            // ARRAY
}

func (s *Schema) FromJSONSchema(j *jsonschema.Schema) {
	s.Type = j.Type
	// s.Format = j.Format
	s.Description = j.Description
	s.Nullable = false
	if len(j.Enum) != 0 {
		s.Enum = make([]string, len(j.Enum))
		for i, e := range j.Enum {
			switch v := e.(type) {
			case string:
				s.Enum[i] = v
			default:
				// TODO: Propagate the error.
				panic(fmt.Sprintf("invalid enum type: got %T, want string", e))
			}
		}
	}
	// s.MaxItems = j.MaxItems
	// s.MinItems = j.MinItems
	if l := j.Properties.Len(); l != 0 {
		s.Properties = make(map[string]Schema, l)
		for pair := j.Properties.Oldest(); pair != nil; pair = pair.Next() {
			i := Schema{}
			i.FromJSONSchema(pair.Value)
			s.Properties[pair.Key] = i
		}
	}
	s.Required = j.Required
	// s.PropertyOrdering = j.PropertyOrdering
	if j.Items != nil {
		s.Items = &Schema{}
		s.Items.FromJSONSchema(j.Items)
	}
}

// https://ai.google.dev/api/caching?hl=en#Mode_1
type ToolMode string

const (
	// Unspecified function calling mode. This value should not be used.
	ToolModeUnspecified ToolMode = "" // "MODE_UNSPECIFIED"
	// Default model behavior, model decides to predict either a function call or a natural language response.
	ToolModeAuto ToolMode = "AUTO"
	// Model is constrained to always predicting a function call only. If "allowedFunctionNames" are set, the
	// predicted function call will be limited to any one of "allowedFunctionNames", else the predicted function
	// call will be any one of the provided "functionDeclarations".
	ToolModeAny ToolMode = "ANY"
	// Model will not predict any function call. Model behavior is same as when not passing any function
	// declarations.
	ToolModeNone ToolMode = "NONE"
	// Model decides to predict either a function call or a natural language response, but will validate
	// function calls with constrained decoding.
	ToolModeValidated ToolMode = "VALIDATED"
)

// https://ai.google.dev/api/caching?hl=en#ToolConfig
type ToolConfig struct {
	// https://ai.google.dev/api/caching?hl=en#FunctionCallingConfig
	FunctionCallingConfig struct {
		Mode                 ToolMode `json:"mode,omitzero"`
		AllowedFunctionNames []string `json:"allowedFunctionNames,omitzero"`
	} `json:"functionCallingConfig,omitzero"`
}

// https://ai.google.dev/api/generate-content#Modality
type Modality string

const (
	ModalityUnspecified Modality = "" // "MODALITY_UNSPECIFIED"
	ModalityText        Modality = "TEXT"
	ModalityImage       Modality = "IMAGE"
	ModalityAudio       Modality = "AUDIO"
)

// https://ai.google.dev/api/generate-content#MediaResolution
type MediaResolution string

const (
	MediaResolutionUnspecified MediaResolution = ""       // "MEDIA_RESOLUTION_UNSPECIFIED"
	MediaResolutionLow         MediaResolution = "LOW"    // 64 tokens
	MediaResolutionMedium      MediaResolution = "MEDIUM" // 256 tokens
	MediaResolutionHigh        MediaResolution = "HIGH"   // zoomed reframing with 256 tokens
)

// https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
type ChatRequest struct {
	Contents          []Content       `json:"contents"`
	Tools             []Tool          `json:"tools,omitzero"`
	ToolConfig        ToolConfig      `json:"toolConfig,omitzero"`
	SafetySettings    []SafetySetting `json:"safetySettings,omitzero"`
	SystemInstruction Content         `json:"systemInstruction,omitzero"`
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerationConfig
	GenerationConfig struct {
		StopSequences              []string   `json:"stopSequences,omitzero"`
		ResponseMimeType           string     `json:"responseMimeType,omitzero"` // "text/plain", "application/json", "text/x.enum"
		ResponseSchema             Schema     `json:"responseSchema,omitzero"`   // Requires ResponseMimeType == "application/json"
		ResponseModalities         []Modality `json:"responseModalities,omitzero"`
		CandidateCount             int64      `json:"candidateCount,omitzero"` // >= 1
		MaxOutputTokens            int64      `json:"maxOutputTokens,omitzero"`
		Temperature                float64    `json:"temperature,omitzero"` // [0, 2]
		TopP                       float64    `json:"topP,omitzero"`
		TopK                       int64      `json:"topK,omitzero"`
		Seed                       int64      `json:"seed,omitzero"`
		PresencePenalty            float64    `json:"presencePenalty,omitzero"`
		FrequencyPenalty           float64    `json:"frequencyPenalty,omitzero"`
		ResponseLogprobs           bool       `json:"responseLogprobs,omitzero"`
		Logprobs                   int64      `json:"logProbs,omitzero"` // Number of logprobs to return
		EnableEnhancedCivicAnswers bool       `json:"enableEnhancedCivicAnswers,omitzero"`
		// https://ai.google.dev/api/generate-content?hl=en#SpeechConfig
		SpeechConfig struct {
			// https://ai.google.dev/api/generate-content?hl=en#VoiceConfig
			VoiceConfig struct {
				// https://ai.google.dev/api/generate-content?hl=en#PrebuiltVoiceConfig
				PrebuiltVoiceConfig struct {
					VoiceName string `json:"voiceName,omitzero"`
				} `json:"prebuiltVoiceConfig,omitzero"`
			} `json:"voiceConfig,omitzero"`
		} `json:"speechConfig,omitzero"`
		// See https://ai.google.dev/gemini-api/docs/thinking#rest
		// This is frustrating: it must be present for thinking models to make it possible to disable thinking. It
		// must NOT be present for non-thinking models, like "gemini-2.0-flash-lite" which we use for smoke tests.
		ThinkingConfig  *ThinkingConfig `json:"thinkingConfig,omitempty"`
		MediaResolution MediaResolution `json:"mediaResolution,omitzero"`
	} `json:"generationConfig,omitzero"`
	CachedContent string `json:"cachedContent,omitzero"` // Name of the cached content with "cachedContents/" prefix.
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable, model string) error {
	var errs []error
	var unsupported []string

	// Disable thinking by default.
	// Hard code which models accept it.
	// - Gemini only.
	// - Not lite
	// - Not 1.x
	// - Not 2.0-flash except if contains thinking.
	if !strings.Contains(model, "lite") &&
		strings.HasPrefix(model, "gemini-") &&
		!strings.HasPrefix(model, "gemini-1") &&
		(!strings.HasPrefix(model, "gemini-2.0-flash") || strings.Contains(model, "thinking")) {
		// Disable thinking.
		c.GenerationConfig.ThinkingConfig = &ThinkingConfig{}
	}

	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			// This doesn't seem to be well supported yet:
			//    in.GenerationConfig.ResponseLogprobs = true
			switch v := opts.(type) {
			case *ChatOptions:
				if v.ThinkingBudget > 0 {
					// https://ai.google.dev/gemini-api/docs/thinking
					c.GenerationConfig.ThinkingConfig = &ThinkingConfig{
						IncludeThoughts: true,
						ThinkingBudget:  v.ThinkingBudget,
					}
				}
				if len(v.ResponseModalities) != 0 {
					c.GenerationConfig.ResponseModalities = v.ResponseModalities
				}
				unsupported = c.initOptions(&v.ChatOptions, model)
			case *genai.ChatOptions:
				unsupported = c.initOptions(v, model)
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
			// Default to text generation.
			if len(c.GenerationConfig.ResponseModalities) == 0 {
				c.GenerationConfig.ResponseModalities = []Modality{ModalityText}
			}
		}
	}

	if err := msgs.Validate(); err != nil {
		errs = append(errs, err)
	} else {
		c.Contents = make([]Content, len(msgs))
		for i := range msgs {
			if err := c.Contents[i].From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message %d: %w", i, err))
			}
		}
	}
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	// There's no field to set, the URL is different.
}

func (c *ChatRequest) initOptions(v *genai.ChatOptions, model string) []string {
	var unsupported []string
	c.GenerationConfig.MaxOutputTokens = v.MaxTokens
	c.GenerationConfig.Temperature = v.Temperature
	c.GenerationConfig.TopP = v.TopP
	// For large ones, we could use cached storage.
	if v.SystemPrompt != "" {
		c.SystemInstruction.Parts = []Part{{Text: v.SystemPrompt}}
	}
	c.GenerationConfig.Seed = v.Seed
	c.GenerationConfig.TopK = v.TopK
	c.GenerationConfig.StopSequences = v.Stop
	if v.DecodeAs != nil {
		c.GenerationConfig.ResponseMimeType = "application/json"
		c.GenerationConfig.ResponseSchema.FromJSONSchema(jsonschema.Reflect(v.DecodeAs))
	} else if v.ReplyAsJSON {
		c.GenerationConfig.ResponseMimeType = "application/json"
	}
	if len(v.Tools) != 0 {
		switch v.ToolCallRequest {
		case genai.ToolCallAny:
			c.ToolConfig.FunctionCallingConfig.Mode = ToolModeValidated
		case genai.ToolCallRequired:
			c.ToolConfig.FunctionCallingConfig.Mode = ToolModeAny
		case genai.ToolCallNone:
			c.ToolConfig.FunctionCallingConfig.Mode = ToolModeNone
		}
		c.Tools = make([]Tool, len(v.Tools))
		for i, t := range v.Tools {
			params := Schema{}
			if t.InputSchemaOverride != nil {
				params.FromJSONSchema(t.InputSchemaOverride)
			} else {
				params.FromJSONSchema(t.GetInputSchema())
			}
			// See FunctionResponse.To().
			c.Tools[i].FunctionDeclarations = []FunctionDeclaration{{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			}}
			c.Tools[i].FunctionDeclarations[0].Response.FromJSONSchema(jsonschema.Reflect(functionResponse))
		}
	}
	return unsupported
}

var functionResponse struct {
	Response string `json:"response"`
}

// Content is the equivalent of Message for other providers.
// https://ai.google.dev/api/caching?hl=en#Content
type Content struct {
	Role string `json:"role,omitzero"` // "user", "model"
	// Parts can be both content and tool calls.
	Parts []Part `json:"parts"`
}

func (c *Content) From(in *genai.Message) error {
	switch in.Role {
	case genai.User:
		c.Role = "user"
	case genai.Assistant:
		c.Role = "model"
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	c.Parts = make([]Part, len(in.Contents)+len(in.ToolCalls)+len(in.ToolCallResults))
	for i := range in.Contents {
		if err := c.Parts[i].FromContent(&in.Contents[i]); err != nil {
			return fmt.Errorf("part %d: %w", i, err)
		}
	}
	offset := len(in.Contents)
	for i := range in.ToolCalls {
		if err := c.Parts[offset+i].FunctionCall.From(&in.ToolCalls[i]); err != nil {
			return fmt.Errorf("part %d: %w", offset+i, err)
		}
	}
	offset += len(in.ToolCalls)
	for i := range in.ToolCallResults {
		if err := c.Parts[offset+i].FunctionResponse.From(&in.ToolCallResults[i]); err != nil {
			return fmt.Errorf("part %d: %w", offset+i, err)
		}
	}
	return nil
}

func (c *Content) To(out *genai.Message) error {
	switch role := c.Role; role {
	case "system", "user":
		out.Role = genai.Role(role)
	case "model":
		out.Role = genai.Assistant
	default:
		return fmt.Errorf("unsupported role %q", role)
	}

	for _, part := range c.Parts {
		// There's no signal as to what it is, we have to test its content.
		// We need to split out content from tools.
		if part.Text != "" {
			out.Contents = append(out.Contents, genai.Content{Text: part.Text})
			continue
		}
		if part.InlineData.MimeType != "" {
			exts, err := mime.ExtensionsByType(part.InlineData.MimeType)
			if err != nil {
				return fmt.Errorf("failed to get extension for mime type %q: %w", part.InlineData.MimeType, err)
			}
			if len(exts) == 0 {
				return fmt.Errorf("mime type %q has no extension", part.InlineData.MimeType)
			}
			out.Contents = append(out.Contents, genai.Content{Filename: "content" + exts[0], Document: bytes.NewReader(part.InlineData.Data)})
			continue
		}
		if part.FileData.MimeType != "" {
			exts, err := mime.ExtensionsByType(part.InlineData.MimeType)
			if err != nil {
				return fmt.Errorf("failed to get extension for mime type %q: %w", part.InlineData.MimeType, err)
			}
			if len(exts) == 0 {
				return fmt.Errorf("mime type %q has no extension", part.InlineData.MimeType)
			}
			out.Contents = append(out.Contents, genai.Content{Filename: "content" + exts[0], URL: part.FileData.FileURI})
			continue
		}
		if part.FunctionCall.Name != "" {
			out.ToolCalls = append(out.ToolCalls, genai.ToolCall{})
			if err := part.FunctionCall.To(&out.ToolCalls[len(out.ToolCalls)-1]); err != nil {
				return err
			}
			continue
		}
		if reflect.ValueOf(part).IsZero() {
			continue
		}
		return fmt.Errorf("unsupported part %#v", part)
	}
	return nil
}

// Part is a union that only has one of the field set.
// Part is the equivalent of Content for other providers.
//
// https://ai.google.dev/api/caching?hl=en#Part
type Part struct {
	Text                string              `json:"text,omitzero"`
	InlineData          Blob                `json:"inlineData,omitzero"` // Uploaded with /v1beta/cachedContents. Content is deleted after 1 hour.
	FunctionCall        FunctionCall        `json:"functionCall,omitzero"`
	FunctionResponse    FunctionResponse    `json:"functionResponse,omitzero"`
	FileData            FileData            `json:"fileData,omitzero"`            // Uploaded with /upload/v1beta/files. Files are deleted after 2 days.
	ExecutableCode      ExecutableCode      `json:"executableCode,omitzero"`      // TODO
	CodeExecutionResult CodeExecutionResult `json:"codeExecutionResult,omitzero"` // TODO
}

func (p *Part) FromContent(in *genai.Content) error {
	if in.Text != "" {
		p.Text = in.Text
		return nil
	}
	mimeType := ""
	var data []byte
	if in.URL == "" {
		// If more than 20MB, we need to use
		// https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#large-pdfs-urls
		// cacheName, err := c.cacheContent(ctx, context, mime, sp)
		// When using cached content, system instruction, tools or tool_config cannot be used. Weird.
		// in.CachedContent = cacheName
		var err error
		if mimeType, data, err = in.ReadDocument(10 * 1024 * 1024); err != nil {
			return err
		}
		p.InlineData.MimeType = mimeType
		p.InlineData.Data = data
	} else {
		if mimeType = mime.TypeByExtension(path.Ext(in.URL)); mimeType == "" {
			return fmt.Errorf("could not determine mime type for URL %q", in.URL)
		}
		p.FileData.MimeType = mimeType
		p.FileData.FileURI = in.URL
	}
	return nil
}

// https://ai.google.dev/api/caching?hl=en#FunctionCall
type FunctionCall struct {
	ID   string      `json:"id,omitzero"`
	Name string      `json:"name,omitzero"`
	Args StructValue `json:"args,omitzero"`
}

func (f *FunctionCall) From(in *genai.ToolCall) error {
	f.ID = in.ID
	f.Name = in.Name
	if err := json.Unmarshal([]byte(in.Arguments), &f.Args); err != nil {
		return fmt.Errorf("failed to unmarshal arguments: %w", err)
	}
	return nil
}

func (f *FunctionCall) To(out *genai.ToolCall) error {
	out.ID = f.ID
	out.Name = f.Name
	raw, err := json.Marshal(f.Args)
	if err != nil {
		return fmt.Errorf("failed to marshal arguments: %w", err)
	}
	out.Arguments = string(raw)
	return nil
}

// https://ai.google.dev/api/caching?hl=en#FunctionResponse
type FunctionResponse struct {
	ID       string      `json:"id,omitzero"`
	Name     string      `json:"name,omitzero"`
	Response StructValue `json:"response,omitzero"`
}

func (f *FunctionResponse) From(in *genai.ToolCallResult) error {
	f.ID = in.ID
	f.Name = in.Name
	// Must match functionResponse
	f.Response = StructValue{"response": in.Result}
	return nil
}

// https://ai.google.dev/api/caching?hl=en#ExecutableCode
type ExecutableCode struct {
	Language string `json:"language,omitzero"` // Only PYTHON is supported as of March 2025.
	Code     string `json:"code,omitzero"`
}

// https://ai.google.dev/api/caching?hl=en#CodeExecutionResult
type CodeExecutionResult struct {
	Outcome string `json:"outcome,omitzero"` // One of OUTCOME_UNSPECIFIED, OUTCOME_OK, OUTCOME_FAILED, OUTCOME_DEADLINE_EXCEEDED
	Output  string `json:"output,omitzero"`
}

// https://ai.google.dev/api/generate-content?hl=en#ThinkingConfig
// See https://ai.google.dev/gemini-api/docs/thinking#rest
type ThinkingConfig struct {
	// IncludeThoughts has no effect since January 2025 according to
	// https://discuss.ai.google.dev/t/thoughts-are-missing-cot-not-included-anymore/63653/13
	IncludeThoughts bool  `json:"includeThoughts"` // Must not be omitted.
	ThinkingBudget  int64 `json:"thinkingBudget"`  // Must not be omitted. [0, 24576]
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerateContentResponse
type ChatResponse struct {
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.Candidate
	Candidates []struct {
		Content      Content      `json:"content"`
		FinishReason FinishReason `json:"finishReason"`
		// https://ai.google.dev/api/generate-content?hl=en#v1beta.SafetyRating
		SafetyRatings []struct {
			// https://ai.google.dev/api/generate-content?hl=en#v1beta.HarmCategory
			Category string `json:"category"`
			// https://ai.google.dev/api/generate-content?hl=en#HarmProbability
			Probability string `json:"probability"`
			Blocked     bool   `json:"blocked"`
		} `json:"safetyRatings"`
		// https://ai.google.dev/api/generate-content?hl=en#v1beta.CitationMetadata
		CitationMetadata struct {
			// https://ai.google.dev/api/generate-content?hl=en#CitationSource
			CitationSources []struct {
				StartIndex int64  `json:"startIndex"`
				EndIndex   int64  `json:"endIndex"`
				URI        string `json:"uri"`
				License    string `json:"license"`
			} `json:"citationSources"`
		} `json:"citationMetadata"`
		TokenCount int64 `json:"tokenCount"`
		// https://ai.google.dev/api/generate-content?hl=en#GroundingAttribution
		GroundingAttributions []struct {
			SourceID string  `json:"sourceId"`
			Countent Content `json:"countent"`
		} `json:"groundingAttributions"`
		// https://ai.google.dev/api/generate-content?hl=en#GroundingMetadata
		GroundingMetadata struct {
			// https://ai.google.dev/api/generate-content?hl=en#GroundingChunk
			GroundingChunks []struct {
				// https://ai.google.dev/api/generate-content?hl=en#Web
				Web struct {
					URI   string `json:"uri"`
					Title string `json:"title"`
				} `json:"web"`
			} `json:"groundingChunks"`
			// https://ai.google.dev/api/generate-content?hl=en#GroundingSupport
			GroundingSupports []struct {
				GroundingChunkIndices []int64   `json:"groundingChunkIndices"`
				ConfidenceScores      []float64 `json:"confidenceScores"`
				// https://ai.google.dev/api/generate-content?hl=en#Segment
				Segment struct {
					PartIndex  int64  `json:"partIndex"`
					StartIndex int64  `json:"startIndex"`
					EndIndex   int64  `json:"endIndex"`
					Text       string `json:"text"`
				} `json:"segment"`
			} `json:"groundingSupports"`
			WebSearchQueries []string `json:"webSearchQueries"`
			// https://ai.google.dev/api/generate-content?hl=en#SearchEntryPoint
			SearchEntryPoint struct {
				RenderedContent string `json:"renderedContent"`
				SDKBlob         []byte `json:"sdkBlob"` // JSON encoded list of (search term,search url) results
			} `json:"searchEntryPoint"`
			// https://ai.google.dev/api/generate-content?hl=en#RetrievalMetadata
			RetrievalMetadata struct {
				GoogleSearchDynamicRetrievalScore float64 `json:"googleSearchDynamicRetrievalScore"`
			} `json:"retrievalMetadata"`
		} `json:"groundingMetadata"`
		AvgLogprobs    float64 `json:"avgLogprobs"`
		LogprobsResult any     `json:"logprobsResult"`
		Index          int64   `json:"index"`
	} `json:"candidates"`
	PromptFeedback any           `json:"promptFeedback,omitzero"`
	UsageMetadata  UsageMetadata `json:"usageMetadata"`
	ModelVersion   string        `json:"modelVersion"`
	ResponseID     string        `json:"responseId"`
}

func (c *ChatResponse) ToResult() (genai.ChatResult, error) {
	out := genai.ChatResult{
		Usage: genai.Usage{
			InputTokens:       c.UsageMetadata.PromptTokenCount,
			InputCachedTokens: c.UsageMetadata.CachedContentTokenCount,
			OutputTokens:      c.UsageMetadata.TotalTokenCount,
		},
	}
	if len(c.Candidates) != 1 {
		return out, fmt.Errorf("unexpected number of candidates; expected 1, got %d", len(c.Candidates))
	}
	// Gemini is the only one returning uppercase so convert down for compatibility.
	out.FinishReason = c.Candidates[0].FinishReason.ToFinishReason()
	err := c.Candidates[0].Content.To(&out.Message)
	if len(out.ToolCalls) != 0 && out.FinishReason == genai.FinishedStop {
		// Lie for the benefit of everyone.
		out.FinishReason = genai.FinishedToolCalls
	}
	return out, err
}

// https://ai.google.dev/api/generate-content?hl=en#FinishReason
type FinishReason string

const (
	// Natural stop point of the model or provided stop sequence.
	FinishStop FinishReason = "STOP"
	// The maximum number of tokens as specified in the request was reached.
	FinishMaxTokens FinishReason = "MAX_TOKENS"
	// The response candidate content was flagged for safety reasons.
	FinishSafety FinishReason = "SAFETY"
	// The response candidate content was flagged for recitation reasons.
	FinishRecitation FinishReason = "RECITATION"
	// The response candidate content was flagged for using an unsupported language.
	FinishLanguage FinishReason = "LANGUAGE"
	// 	Unknown reason.
	FinishOther FinishReason = "OTHER"
	// Token generation stopped because the content contains forbidden terms.
	FinishBlocklist FinishReason = "BLOCKLIST"
	// Token generation stopped for potentially containing prohibited content.
	FinishProhibitedContent FinishReason = "PROHIBITED_CONTENT"
	// Token generation stopped because the content potentially contains Sensitive Personally Identifiable Information (SPII).
	FinishSPII FinishReason = "SPII"
	// The function call generated by the model is invalid.
	FinishMalformed FinishReason = "MALFORMED_FUNCTION_CALL"
	// Token generation stopped because generated images contain safety violations.
	FinishImageSafety FinishReason = "IMAGE_SAFETY"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishMaxTokens:
		return genai.FinishedLength
	case FinishSafety, FinishBlocklist, FinishProhibitedContent, FinishSPII, FinishImageSafety:
		// TODO: Confirm. We lose on nuance here but does it matter?
		return genai.FinishedContentFilter
	default:
		return genai.FinishReason(strings.ToLower(string(f)))
	}
}

// https://ai.google.dev/api/generate-content?hl=en#UsageMetadata
type UsageMetadata struct {
	PromptTokenCount           int64                `json:"promptTokenCount"`
	CachedContentTokenCount    int64                `json:"cachedContentTokenCount"`
	CandidatesTokenCount       int64                `json:"candidatesTokenCount"`
	ToolUsePromptTokenCount    int64                `json:"toolUsePromptTokenCount"`
	ThoughtsTokenCount         int64                `json:"thoughtsTokenCount"`
	TotalTokenCount            int64                `json:"totalTokenCount"`
	PromptTokensDetails        []ModalityTokenCount `json:"promptTokensDetails"`
	CacheTokensDetails         []ModalityTokenCount `json:"cacheTokensDetails"`
	CandidatesTokensDetails    []ModalityTokenCount `json:"candidatesTokensDetails"`
	ToolUsePromptTokensDetails []ModalityTokenCount `json:"toolUsePromptTokensDetails"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.ModalityTokenCount
type ModalityTokenCount struct {
	Modality   Modality `json:"modality"`
	TokenCount int64    `json:"tokenCount"`
}

type ChatStreamChunkResponse struct {
	Candidates []struct {
		Content      Content      `json:"content"`
		FinishReason FinishReason `json:"finishReason"`
		Index        int64        `json:"index"`
	} `json:"candidates"`
	UsageMetadata UsageMetadata `json:"usageMetadata"`
	ModelVersion  string        `json:"modelVersion"`
	ResponseID    string        `json:"responseId"`
}

// Caching

// https://ai.google.dev/api/caching?hl=en#request-body
// https://ai.google.dev/api/caching?hl=en#CachedContent
type CachedContent struct {
	Expiration
	Contents          []Content  `json:"contents,omitzero"`
	Tools             []Tool     `json:"tools,omitzero"`
	Name              string     `json:"name,omitzero"`
	DisplayName       string     `json:"displayName,omitzero"`
	Model             string     `json:"model"`
	SystemInstruction Content    `json:"systemInstruction,omitzero"`
	ToolConfig        ToolConfig `json:"toolConfig,omitzero"`

	// Not to be set on upload.
	CreateTime    string               `json:"createTime,omitzero"`
	UpdateTime    string               `json:"updateTime,omitzero"`
	UsageMetadata CachingUsageMetadata `json:"usageMetadata,omitzero"`
}

// Expiration must be embedded. Only one of the fields can be set.
type Expiration struct {
	ExpireTime time.Time `json:"expireTime,omitzero"` // ISO 8601
	TTL        Duration  `json:"ttl,omitzero"`        // Duration
}

type Duration time.Duration

func (d *Duration) IsZero() bool {
	return *d == 0
}

func (d *Duration) MarshalJSON() ([]byte, error) {
	v := time.Duration(*d)
	return json.Marshal(fmt.Sprintf("%1.fs", v.Seconds()))
}

func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return fmt.Errorf("invalid duration %q: %w", string(b), err)
	}
	v, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", s, err)
	}
	*d = Duration(v)
	return nil
}

// https://ai.google.dev/api/caching?hl=en#UsageMetadata
type CachingUsageMetadata struct {
	TotalTokenCount int64 `json:"totalTokenCount"`
}

// https://ai.google.dev/api/models#Model
type Model struct {
	Name                       string   `json:"name"`
	BaseModelID                string   `json:"baseModelId"`
	Version                    string   `json:"version"`
	DisplayName                string   `json:"displayName"`
	Description                string   `json:"description"`
	InputTokenLimit            int64    `json:"inputTokenLimit"`
	OutputTokenLimit           int64    `json:"outputTokenLimit"`
	SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
	Temperature                float64  `json:"temperature"`
	MaxTemperature             float64  `json:"maxTemperature"`
	TopP                       float64  `json:"topP"`
	TopK                       int64    `json:"topK"`
}

func (m *Model) GetID() string {
	return strings.TrimPrefix(m.Name, "models/")
}

func (m *Model) String() string {
	return fmt.Sprintf("%s: %s (%s) Context: %d/%d", m.GetID(), m.DisplayName, m.Description, m.InputTokenLimit, m.OutputTokenLimit)
}

func (m *Model) Context() int64 {
	return m.InputTokenLimit
}

// ModelsResponse represents the response structure for Gemini models listing
type ModelsResponse struct {
	Models        []Model `json:"models"`
	NextPageToken string  `json:"nextPageToken"`
}

// ToIModels converts Gemini models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Models))
	for i := range r.Models {
		models[i] = &r.Models[i]
	}
	return models
}

//

type ErrorResponse struct {
	Error ErrorResponseError `json:"error"`
}

func (e *ErrorResponse) String() string {
	return fmt.Sprintf("error %d (%s): %s", e.Error.Code, e.Error.Status, strings.TrimSpace(e.Error.Message))
}

type ErrorResponseError struct {
	Code    int64  `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
	Details []struct {
		Type     string `json:"@type"`
		Reason   string `json:"reason"`
		Domain   string `json:"domain"`
		Metadata struct {
			Service string `json:"service"`
		} `json:"metadata"`
		FieldViolations []struct {
			Field       string `json:"field"`
			Description string `json:"description"`
		} `json:"fieldViolations"`
		Locale  string `json:"locale"`
		Message string `json:"message"`
	} `json:"details"`
}

// Client implements genai.ProviderChat and genai.ProviderModel.
type Client struct {
	internal.ClientChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]

	apiKey string
}

// New creates a new client to talk to Google's Gemini platform API.
//
// If apiKey is not provided, it tries to load it from the GEMINI_API_KEY environment variable.
// If none is found, it returns an error.
// Get your API key at https://ai.google.dev/gemini-api/docs/getting-started
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://ai.google.dev/gemini-api/docs/models/gemini
//
// r can be used to throttle outgoing requests, record calls, etc. It defaults to http.DefaultTransport.
//
// See https://ai.google.dev/gemini-api/docs/file-prompting-strategies?hl=en
// for good ideas on how to prompt with images.
//
// https://ai.google.dev/gemini-api/docs/pricing
//
// Using large files requires a pinned model with caching support.
//
// Supported mime types for images:
// https://ai.google.dev/gemini-api/docs/vision?hl=en&lang=rest#prompting-images
// - image/png
// - image/jpeg
// - image/webp
// - image/heic
// - image/heif
//
// Supported mime types for videos:
// https://ai.google.dev/gemini-api/docs/vision?hl=en&lang=rest#technical-details-video
// - video/mp4
// - video/mpeg
// - video/mov
// - video/avi
// - video/x-flv
// - video/mpg
// - video/webm
// - video/wmv
// - video/3gpp
//
// Supported mime types for audio:
// https://ai.google.dev/gemini-api/docs/audio?hl=en&lang=rest#supported-formats
// - audio/wav
// - audio/mp3
// - audio/aiff
// - audio/aac
// - audio/ogg
// - audio/flac
//
// Supported mime types for documents:
// https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#technical-details
// - application/pdf
// - application/x-javascript, text/javascript
// - application/x-python, text/x-python
// - text/plain
// - text/html
// - text/css
// - text/md
// - text/csv
// - text/xml
// - text/rtf
//
// Visit https://ai.google.dev/gemini-api/docs/pricing for up to date information.
//
// As of May 2025, price on Pro model increases when more than 200k input tokens are used.
// Cached input tokens are 25% of the price of new tokens.
func New(apiKey, model string, r http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://ai.google.dev/gemini-api/docs/getting-started"
	if apiKey == "" {
		if apiKey = os.Getenv("GEMINI_API_KEY"); apiKey == "" {
			return nil, errors.New("gemini API key is required; get one at " + apiKeyURL)
		}
	}
	if r == nil {
		r = http.DefaultTransport
	}
	// Eventually, use OAuth https://ai.google.dev/gemini-api/docs/oauth#curl
	return &Client{
		ClientChat: internal.ClientChat[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			ChatURL:              "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + apiKey,
			ChatStreamURL:        "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":streamGenerateContent?alt=sse&key=" + apiKey,
			ProcessStreamPackets: processStreamPackets,
			LieToolCalls:         true,
			ClientBase: internal.ClientBase[*ErrorResponse]{
				ClientJSON: httpjson.Client{
					Client: &http.Client{Transport: &roundtrippers.PostCompressed{
						Transport: &roundtrippers.Retry{
							Transport: &roundtrippers.RequestID{
								Transport: r,
							},
						},
						// Google supports HTTP POST gzip compression!
						Encoding: "gzip",
					}},
					Lenient: internal.BeLenient,
				},
				APIKeyURL: apiKeyURL,
			},
		},
		apiKey: apiKey,
	}, nil
}

func (c *Client) Name() string {
	return "gemini"
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

// CacheAdd caches the content for later use.
//
// Includes tools and systemprompt.
//
// Default time to live (ttl) is 1 hour.
//
// The minimum cacheable object is 4096 tokens.
//
// Visit https://ai.google.dev/gemini-api/docs/pricing for up to date information.
//
// As of May 2025, price on Pro model cache cost is 4.50$ per 1MTok/hour and Flash is 1$ per 1Mok/hour.
//
// At certain volumes, using cached tokens is lower cost than passing in the same corpus of tokens repeatedly.
// The cost for caching depends on the input token size and how long you want the tokens to persist.
func (c *Client) CacheAdd(ctx context.Context, msgs genai.Messages, opts *genai.ChatOptions, name, displayName string, ttl time.Duration) (string, error) {
	// See https://ai.google.dev/gemini-api/docs/caching?hl=en&lang=rest#considerations
	// Useful when reusing the same large data multiple times to reduce token usage.
	// This requires a pinned model, with trailing -001.
	in := CachedContent{
		Model:       "models/" + c.Model,
		DisplayName: displayName,
	}
	if name != "" {
		in.Name = "cachedContents/" + name
	}
	if ttl > 0 {
		in.TTL = Duration(ttl)
	}
	if opts.SystemPrompt != "" {
		in.SystemInstruction.Parts = []Part{{Text: opts.SystemPrompt}}
	}
	// For large files, use https://ai.google.dev/gemini-api/docs/caching?hl=en&lang=rest#pdfs_1
	in.Contents = make([]Content, len(msgs))
	for i := range msgs {
		if err := in.Contents[i].From(&msgs[i]); err != nil {
			return "", fmt.Errorf("message %d: %w", i, err)
		}
	}
	// TODO: ToolConfig
	out := CachedContent{}
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=" + c.apiKey
	if err := c.DoRequest(ctx, "POST", url, &in, &out); err != nil {
		return "", err
	}
	name = strings.TrimPrefix(out.Name, "cachedContents/")
	slog.InfoContext(ctx, "gemini", "cached", name, "tokens", out.UsageMetadata.TotalTokenCount)
	return name, nil
}

func (c *Client) CacheExtend(ctx context.Context, name string, ttl time.Duration) error {
	// https://ai.google.dev/api/caching#method:-cachedcontents.patch
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + name + "?key=" + c.apiKey
	// Model is required.
	in := CachedContent{Model: "models/" + c.Model, Expiration: Expiration{TTL: Duration(ttl)}}
	out := CachedContent{}
	return c.DoRequest(ctx, "PATCH", url, &in, &out)
}

// CacheList retrieves the list of cached items.
func (c *Client) CacheList(ctx context.Context) ([]CachedContent, error) {
	// https://ai.google.dev/api/caching#method:-cachedcontents.list
	// pageSize, pageToken
	var data struct {
		CachedContents []CachedContent `json:"cachedContents"`
		NextPageToken  string          `json:"nextPageToken"`
	}
	baseURL := "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=" + c.apiKey + "&pageSize=100"
	var out []CachedContent
	for ctx.Err() == nil {
		url := baseURL
		if data.NextPageToken != "" {
			url += "&pageToken=" + data.NextPageToken
		}
		if data.CachedContents != nil {
			data.CachedContents = data.CachedContents[:0]
		}
		data.NextPageToken = ""
		if err := c.DoRequest(ctx, "GET", url, nil, &data); err != nil {
			return nil, err
		}
		for i := range data.CachedContents {
			data.CachedContents[i].Name = strings.TrimPrefix(data.CachedContents[i].Name, "cachedContents/")
		}
		out = append(out, data.CachedContents...)
		if len(data.CachedContents) == 0 || data.NextPageToken == "" {
			break
		}
	}
	return out, nil
}

func (c *Client) CacheGet(ctx context.Context, name string) (CachedContent, error) {
	// https://ai.google.dev/api/caching#method:-cachedcontents.get
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + name + "?key=" + c.apiKey
	out := CachedContent{}
	err := c.DoRequest(ctx, "GET", url, nil, &out)
	return out, err
}

// CacheDelete deletes a cached file.
func (c *Client) CacheDelete(ctx context.Context, name string) error {
	// https://ai.google.dev/api/caching#method:-cachedcontents.delete
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + name + "?key=" + c.apiKey
	var out struct{}
	return c.DoRequest(ctx, "DELETE", url, nil, &out)
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://ai.google.dev/api/models?hl=en#method:-models.list
	return internal.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.ClientBase, "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key="+c.apiKey)
}

// processStreamPackets is the function used to convert the chunks sent by Gemini's SSE data into
// MessageFragment.
func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if len(pkt.Candidates) != 1 {
			continue
		}
		if pkt.UsageMetadata.TotalTokenCount != 0 {
			result.InputTokens = pkt.UsageMetadata.PromptTokenCount
			result.OutputTokens = pkt.UsageMetadata.TotalTokenCount
			result.FinishReason = pkt.Candidates[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Candidates[0].Content.Role; role {
		case "model", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}

		// Gemini is the only one returning uppercase so convert down for compatibility.
		f := genai.MessageFragment{}
		for _, part := range pkt.Candidates[0].Content.Parts {
			f.TextFragment += part.Text
			if part.InlineData.MimeType != "" || len(part.InlineData.Data) != 0 {
				exts, err := mime.ExtensionsByType(part.InlineData.MimeType)
				if err != nil {
					return fmt.Errorf("failed to get extension for mime type %q: %w", part.InlineData.MimeType, err)
				}
				if len(exts) == 0 {
					return fmt.Errorf("mime type %q has no extension", part.InlineData.MimeType)
				}
				f.Filename = "content" + exts[0]
				f.DocumentFragment = part.InlineData.Data
				// return fmt.Errorf("implement inline blob %#v", part)
			}
			if part.FunctionCall.ID != "" || part.FunctionCall.Name != "" {
				// https://ai.google.dev/api/caching?hl=en#FunctionCall
				if err := part.FunctionCall.To(&f.ToolCall); err != nil {
					return err
				}
			}
			if part.FunctionResponse.ID != "" {
				// https://ai.google.dev/api/caching?hl=en#FunctionResponse
				return fmt.Errorf("implement function response %#v", part)
			}
			if part.FileData.MimeType != "" || part.FileData.FileURI != "" {
				return fmt.Errorf("implement file data %#v", part)
			}
			if part.ExecutableCode.Language != "" || part.ExecutableCode.Code != "" {
				// https://ai.google.dev/api/caching?hl=en#ExecutableCode
				return fmt.Errorf("implement executable code %#v", part)
			}
			if part.CodeExecutionResult.Outcome != "" || part.CodeExecutionResult.Output != "" {
				// https://ai.google.dev/api/caching?hl=en#CodeExecutionResult
				return fmt.Errorf("implement code execution result %#v", part)
			}
		}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	return nil
}

var (
	_ genai.ProviderChat       = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
