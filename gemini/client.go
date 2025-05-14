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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
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
	"golang.org/x/sync/errgroup"
)

// ChatOptions includes Gemini specific options.
type ChatOptions struct {
	genai.ChatOptions

	// ThinkingBudget is the maximum number of tokens the LLM can use to think about the answer. When 0,
	// thinking is disabled. It generally must be above 1024 and below MaxTokens and 24576.
	ThinkingBudget int64
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
				unsupported = c.initOptions(&v.ChatOptions, model)
			case *genai.ChatOptions:
				unsupported = c.initOptions(v, model)
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
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
	// https://ai.google.dev/gemini-api/docs/image-generation
	// SOON: "Image"
	c.GenerationConfig.ResponseModalities = []Modality{ModalityText}
	if v.ReplyAsJSON {
		c.GenerationConfig.ResponseMimeType = "application/json"
	}
	if v.DecodeAs != nil {
		c.GenerationConfig.ResponseMimeType = "application/json"
		c.GenerationConfig.ResponseSchema.FromJSONSchema(jsonschema.Reflect(v.DecodeAs))
	}
	if len(v.Tools) != 0 {
		if v.ToolCallRequired {
			c.ToolConfig.FunctionCallingConfig.Mode = ToolModeAny
		} else {
			c.ToolConfig.FunctionCallingConfig.Mode = ToolModeValidated
		}
		c.Tools = make([]Tool, len(v.Tools))
		for i, t := range v.Tools {
			params := Schema{}
			if t.InputsAs != nil {
				params.FromJSONSchema(jsonschema.Reflect(t.InputsAs))
			}
			c.Tools[i].FunctionDeclarations = []FunctionDeclaration{{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
				// Expose this eventually? We have to test if Google supports non-string answers.
				Response: Schema{Type: "string"},
			}}
		}
	}
	return unsupported
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
	c.Parts = make([]Part, len(in.Contents)+len(in.ToolCalls))
	for i := range in.Contents {
		if err := c.Parts[i].FromContent(&in.Contents[i]); err != nil {
			return fmt.Errorf("part %d: %w", i, err)
		}
	}
	offset := len(in.Contents)
	for i := range in.ToolCalls {
		if err := c.Parts[offset+i].FromToolCall(&in.ToolCalls[i]); err != nil {
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
			raw, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return fmt.Errorf("failed to marshal arguments: %w", err)
			}
			out.ToolCalls = append(out.ToolCalls,
				genai.ToolCall{
					ID:        part.FunctionCall.ID,
					Name:      part.FunctionCall.Name,
					Arguments: string(raw),
				})
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
	Text string `json:"text,omitzero"`
	// Uploaded with /v1beta/cachedContents. Content is deleted after 1 hour.
	InlineData Blob `json:"inlineData,omitzero"`
	// https://ai.google.dev/api/caching?hl=en#FunctionCall
	FunctionCall struct {
		ID   string      `json:"id,omitzero"`
		Name string      `json:"name,omitzero"`
		Args StructValue `json:"args,omitzero"`
	} `json:"functionCall,omitzero"`
	// https://ai.google.dev/api/caching?hl=en#FunctionResponse
	FunctionResponse struct {
		ID       string      `json:"id,omitzero"`
		Name     string      `json:"name,omitzero"`
		Response StructValue `json:"response,omitzero"`
	} `json:"functionResponse,omitzero"`
	// Uploaded with /upload/v1beta/files. Files are deleted after 2 days.
	FileData FileData `json:"fileData,omitzero"`
	// https://ai.google.dev/api/caching?hl=en#ExecutableCode
	ExecutableCode struct {
		Language string `json:"language,omitzero"` // Only PYTHON is supported as of March 2025.
		Code     string `json:"code,omitzero"`
	} `json:"executableCode,omitzero"` // TODO
	// https://ai.google.dev/api/caching?hl=en#CodeExecutionResult
	CodeExecutionResult struct {
		Outcome string `json:"outcome,omitzero"` // One of OUTCOME_UNSPECIFIED, OUTCOME_OK, OUTCOME_FAILED, OUTCOME_DEADLINE_EXCEEDED
		Output  string `json:"output,omitzero"`
	} `json:"codeExecutionResult,omitzero"` // TODO
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

func (p *Part) FromToolCall(in *genai.ToolCall) error {
	p.FunctionCall.ID = in.ID
	p.FunctionCall.Name = in.Name
	if err := json.Unmarshal([]byte(in.Arguments), &p.FunctionCall.Args); err != nil {
		return fmt.Errorf("failed to unmarshal arguments: %w", err)
	}
	return nil
}

func (p *Part) ToToolCall(out *genai.ToolCall) error {
	out.ID = p.FunctionCall.ID
	out.Name = p.FunctionCall.Name
	raw, err := json.Marshal(p.FunctionCall.Args)
	if err != nil {
		return fmt.Errorf("failed to marshal arguments: %w", err)
	}
	out.Arguments = string(raw)
	return nil
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
		Content Content `json:"content"`
		// https://ai.google.dev/api/generate-content?hl=en#FinishReason
		FinishReason string `json:"finishReason"` // "STOP" (uppercase)
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
		return out, fmt.Errorf("unexpected number of candidates; expected 1, got %v", c.Candidates)
	}
	// Gemini is the only one returning uppercase so convert down for compatibility.
	out.FinishReason = strings.ToLower(c.Candidates[0].FinishReason)
	err := c.Candidates[0].Content.To(&out.Message)
	return out, err
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
		Content      Content `json:"content"`
		FinishReason string  `json:"finishReason"` // STOP
	} `json:"candidates"`
	UsageMetadata UsageMetadata `json:"usageMetadata"`
	ModelVersion  string        `json:"modelVersion"`
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

//

type errorResponse struct {
	Error errorResponseError `json:"error"`
}

func (e *errorResponse) String() string {
	return fmt.Sprintf("error %d (%s): %s", e.Error.Code, e.Error.Status, e.Error.Message)
}

type errorResponseError struct {
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

// Client implements the REST JSON based API.
type Client struct {
	// Client is exported for testing replay purposes.
	Client httpjson.Client

	apiKey string
	model  string
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
func New(apiKey, model string) (*Client, error) {
	if apiKey == "" {
		if apiKey = os.Getenv("GEMINI_API_KEY"); apiKey == "" {
			return nil, errors.New("gemini API key is required; get one at " + apiKeyURL)
		}
	}
	// Eventually, use OAuth https://ai.google.dev/gemini-api/docs/oauth#curl
	return &Client{
		apiKey: apiKey,
		model:  model,
		Client: httpjson.Client{
			Client: &http.Client{Transport: &roundtrippers.PostCompressed{
				Transport: &roundtrippers.Retry{
					Transport: &roundtrippers.RequestID{
						Transport: http.DefaultTransport,
					},
				},
				// Google supports HTTP POST gzip compression!
				Encoding: "gzip",
			}},
			Lenient: internal.BeLenient,
		},
	}, nil
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
		Model:       "models/" + c.model,
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
	if err := c.doRequest(ctx, "POST", url, &in, &out); err != nil {
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
	in := CachedContent{Model: "models/" + c.model, Expiration: Expiration{TTL: Duration(ttl)}}
	out := CachedContent{}
	return c.doRequest(ctx, "PATCH", url, &in, &out)
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
		if err := c.doRequest(ctx, "GET", url, nil, &data); err != nil {
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
	err := c.doRequest(ctx, "GET", url, nil, &out)
	return out, err
}

// CacheDelete deletes a cached file.
func (c *Client) CacheDelete(ctx context.Context, name string) error {
	// https://ai.google.dev/api/caching#method:-cachedcontents.delete
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + name + "?key=" + c.apiKey
	er := errorResponse{}
	err := c.doRequest(ctx, "DELETE", url, nil, &er)
	if err != nil {
		return err
	}
	if er.Error.Code != 0 {
		return errors.New(er.String())
	}
	return nil
}

// Chat implements genai.ChatProvider.
//
// Visit https://ai.google.dev/gemini-api/docs/pricing for up to date information.
//
// As of May 2025, price on Pro model increases when more than 200k input tokens are used.
// Cached input tokens are 25% of the price of new tokens.
func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return genai.ChatResult{}, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
			}
		}
	}
	rpcin := ChatRequest{}
	var continuableErr error
	if err := rpcin.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return genai.ChatResult{}, err
		}
	}
	rpcout := ChatResponse{}
	if err := c.ChatRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.ChatResult{}, err
	}
	result, err := rpcout.ToResult()
	if err != nil {
		return result, err
	}
	// Return the continuable error if no other error occurred
	if continuableErr != nil {
		return result, continuableErr
	}
	return result, nil
}

func (c *Client) ChatRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	// https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
	if err := c.validate(); err != nil {
		return err
	}
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + c.model + ":generateContent?key=" + c.apiKey
	return c.doRequest(ctx, "POST", url, in, out)
}

// ChatStream implements genai.ChatProvider.
func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.Usage, error) {
	// Check for non-empty Opaque field
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return genai.Usage{}, fmt.Errorf("message #%d content #%d: Opaque field not supported", i, j)
			}
		}
	}

	in := ChatRequest{}
	usage := genai.Usage{}
	var continuableErr error
	if err := in.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return usage, err
		}
	}
	ch := make(chan ChatStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	finalUsage := UsageMetadata{}
	eg.Go(func() error {
		return processStreamPackets(ch, chunks, &finalUsage)
	})
	err := c.ChatStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	usage.InputTokens = finalUsage.PromptTokenCount
	usage.OutputTokens = finalUsage.TotalTokenCount
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return usage, continuableErr
	}
	return usage, err
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, finalUsage *UsageMetadata) error {
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
			*finalUsage = pkt.UsageMetadata
		}
		switch role := pkt.Candidates[0].Content.Role; role {
		case "model", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}

		finishReason := pkt.Candidates[0].FinishReason

		for _, part := range pkt.Candidates[0].Content.Parts {
			if part.Text != "" {
				fragment := genai.MessageFragment{TextFragment: part.Text}
				// Only set FinishReason on fragments that have it (typically last chunk)
				if finishReason != "" {
					// Gemini is the only one returning uppercase so convert down for compatibility.
					fragment.FinishReason = strings.ToLower(finishReason)
				}
				chunks <- fragment
			}
		}
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerateContentResponse
	if err := c.validate(); err != nil {
		return err
	}
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + c.model + ":streamGenerateContent?alt=sse&key=" + c.apiKey
	resp, err := c.Client.PostRequest(ctx, url, nil, in)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	for r := bufio.NewReader(resp.Body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) != 0 {
			if err := parseStreamLine(line, out); err != nil {
				return err
			}
		}
	}
}

func parseStreamLine(line []byte, out chan<- ChatStreamChunkResponse) error {
	const prefix = "data: "
	if !bytes.HasPrefix(line, []byte(prefix)) {
		return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
	}
	suffix := string(line[len(prefix):])
	d := json.NewDecoder(strings.NewReader(suffix))
	d.DisallowUnknownFields()
	d.UseNumber()
	msg := ChatStreamChunkResponse{}
	if err := d.Decode(&msg); err != nil {
		return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
	}
	out <- msg
	return nil
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

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://ai.google.dev/api/models?hl=en#method:-models.list
	var out struct {
		Models        []Model `json:"models"`
		NextPageToken string  `json:"nextPageToken"`
	}
	err := c.Client.Get(ctx, "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key="+c.apiKey, nil, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genai.Model, len(out.Models))
	for i := range out.Models {
		models[i] = &out.Models[i]
	}
	return models, err
}

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *Client) doRequest(ctx context.Context, method, url string, in, out any) error {
	resp, err := c.Client.Request(ctx, method, url, nil, in)
	if err != nil {
		return err
	}
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			// It's annoying that Google returns 400 instead of 401 for invalid API key.
			if herr.StatusCode == http.StatusBadRequest || herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: %s You can get a new API key at %s", herr, er.String(), apiKeyURL)
			}
			return fmt.Errorf("%w: %s", herr, er.String())
		}
		return errors.New(er.String())
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "gemini", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
			// Google may return an HTML page on invalid API key.
			if bytes.HasPrefix(herr.ResponseBody, []byte("<!DOCTYPE html>")) {
				return fmt.Errorf("%w: You can get a new API key at %s", herr, apiKeyURL)
			}
		} else {
			slog.WarnContext(ctx, "gemini", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://ai.google.dev/gemini-api/docs/getting-started"

var (
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
