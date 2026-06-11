// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Google Gemini REST API.
//
// API reference: https://ai.google.dev/api/?lang=rest
//
// Source: https://github.com/googleapis/go-genai

package gemini

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
)

// Schema types.

// Type is documented at https://ai.google.dev/api/caching#Type
//
// The official Go SDK documentation is at https://pkg.go.dev/google.golang.org/genai#Type
type Type string

// Type values.
const (
	TypeUnspecified Type = "TYPE_UNSPECIFIED"
	TypeString      Type = "STRING"
	TypeNumber      Type = "NUMBER"
	TypeInteger     Type = "INTEGER"
	TypeBoolean     Type = "BOOLEAN"
	TypeArray       Type = "ARRAY"
	TypeObject      Type = "OBJECT"
	TypeNull        Type = "NULL"
)

// Format is described at https://spec.openapis.org/oas/v3.0.3#data-types
//
// Values outside of the consts are acceptable.
type Format string

const (
	// FormatFloat is for TypeNumber only.
	FormatFloat Format = "float"
	// FormatDouble is for TypeNumber only.
	FormatDouble Format = "double"

	// FormatInt32 is for TypeInteger only.
	FormatInt32 Format = "int32"
	// FormatInt64 is for TypeInteger only.
	FormatInt64 Format = "int64"

	// FormatEnum is for TypeString only.
	FormatEnum Format = "enum"
	// FormatDate is for TypeString only.
	FormatDate Format = "date"
	// FormatDateTime is for TypeString only.
	FormatDateTime Format = "date-time"
	// FormatByte is for TypeString only.
	FormatByte Format = "byte"
	// FormatPassword is for TypeString only.
	FormatPassword Format = "password"
	// FormatEmail is for TypeString only.
	FormatEmail Format = "email"
	// FormatUUID is for TypeString only.
	FormatUUID Format = "uuid"
)

// Schema support is documented at https://ai.google.dev/gemini-api/docs/structured-output
//
// The Schema struct is documented at https://ai.google.dev/api/caching#Schema
// and in the official Go SDK documentation is at https://pkg.go.dev/google.golang.org/genai#Schema
//
// Adapted to have less pointers and use omitzero.
type Schema struct {
	// Optional. The value should be validated against any (one or more) of the subschemas
	// in the list.
	AnyOf []*Schema `json:"anyOf,omitzero"`
	// Optional. Default value of the data.
	Default json.RawMessage `json:"default,omitzero"`
	// Optional. The description of the data.
	Description string `json:"description,omitzero"`
	// Optional. Possible values of the element of primitive type with enum format. Examples:
	// 1. We can define direction as : {type:STRING, format:enum, enum:["EAST", NORTH",
	// "SOUTH", "WEST"]} 2. We can define apartment number as : {type:INTEGER, format:enum,
	// enum:["101", "201", "301"]}
	Enum []string `json:"enum,omitzero"`
	// Optional. Example of the object. Will only populated when the object is the root.
	Example json.RawMessage `json:"example,omitzero"`
	// Optional. The format of the data. Supported formats: for NUMBER type: "float", "double"
	// for INTEGER type: "int32", "int64" for STRING type: "email", "byte", etc
	Format Format `json:"format,omitzero"`
	// Optional. SCHEMA FIELDS FOR TYPE ARRAY Schema of the elements of Type.ARRAY.
	Items *Schema `json:"items,omitzero"`
	// Optional. Maximum number of the elements for Type.ARRAY.
	MaxItems int64 `json:"maxItems,omitzero,string"`
	// Optional. Maximum length of the Type.STRING
	MaxLength int64 `json:"maxLength,omitzero,string"`
	// Optional. Maximum number of the properties for Type.OBJECT.
	MaxProperties int64 `json:"maxProperties,omitzero,string"`
	// Optional. Maximum value of the Type.INTEGER and Type.NUMBER
	Maximum float64 `json:"maximum,omitzero"`
	// Optional. Minimum number of the elements for Type.ARRAY.
	MinItems int64 `json:"minItems,omitzero,string"`
	// Optional. SCHEMA FIELDS FOR TYPE STRING Minimum length of the Type.STRING
	MinLength int64 `json:"minLength,omitzero,string"`
	// Optional. Minimum number of the properties for Type.OBJECT.
	MinProperties int64 `json:"minProperties,omitzero,string"`
	// Optional. Minimum value of the Type.INTEGER and Type.NUMBER.
	Minimum float64 `json:"minimum,omitzero"`
	// Optional. Indicates if the value may be null.
	Nullable bool `json:"nullable,omitzero"`
	// Optional. Pattern of the Type.STRING to restrict a string to a regular expression.
	Pattern string `json:"pattern,omitzero"`
	// Optional. SCHEMA FIELDS FOR TYPE OBJECT Properties of Type.OBJECT.
	Properties map[string]Schema `json:"properties,omitzero"`
	// Optional. The order of the properties. Not a standard field in open API spec. Only
	// used to support the order of the properties.
	PropertyOrdering []string `json:"propertyOrdering,omitzero"`
	// Optional. Required properties of Type.OBJECT.
	Required []string `json:"required,omitzero"`
	// Optional. The title of the Schema.
	Title string `json:"title,omitzero"`
	// Optional. The type of the data.
	Type Type `json:"type,omitzero"`
}

// jsonSchemaNode is a minimal subset of JSON Schema used for conversion to Gemini's Schema.
type jsonSchemaNode struct {
	AnyOf       []jsonSchemaNode          `json:"anyOf"`
	Default     json.RawMessage           `json:"default"`
	Description string                    `json:"description"`
	Enum        []json.RawMessage         `json:"enum"`
	Example     json.RawMessage           `json:"example"`
	Format      string                    `json:"format"`
	Items       *jsonSchemaNode           `json:"items"`
	MaxItems    int64                     `json:"maxItems"`
	MaxLength   int64                     `json:"maxLength"`
	Maximum     float64                   `json:"maximum"`
	MinItems    int64                     `json:"minItems"`
	MinLength   int64                     `json:"minLength"`
	Minimum     float64                   `json:"minimum"`
	Nullable    bool                      `json:"nullable"`
	Properties  map[string]jsonSchemaNode `json:"properties"`
	Required    []string                  `json:"required"`
	Title       string                    `json:"title"`
	// Type handles both the single-string form ("string") and the
	// type-array nullable form (["string", "null"]).
	Type jsonSchemaType `json:"type"`
}

// jsonSchemaType unmarshals the JSON Schema "type" field which may be either a
// plain string ("string") or an array encoding a nullable type (["string", "null"]).
type jsonSchemaType struct {
	value    string
	nullable bool
}

func (t *jsonSchemaType) UnmarshalJSON(b []byte) error {
	if len(b) > 0 && b[0] == '[' {
		var arr []string
		if err := json.Unmarshal(b, &arr); err != nil {
			return err
		}
		for _, v := range arr {
			if v == "null" {
				t.nullable = true
			} else {
				t.value = v
			}
		}
		return nil
	}
	return json.Unmarshal(b, &t.value)
}

// FromJSONSchema populates s from a JSON Schema document.
//
// Accepts standard JSON Schema with lowercase type names ("string", "integer", etc.).
// Use with genai.GenOptionText.DecodeSchema() or genai.ToolDef.GetInputSchema().
func (s *Schema) FromJSONSchema(js genai.JSONSchema) error {
	var n jsonSchemaNode
	if err := json.Unmarshal(js, &n); err != nil {
		return fmt.Errorf("invalid JSON schema: %w", err)
	}
	return s.fromNode(&n)
}

func (s *Schema) fromNode(n *jsonSchemaNode) error {
	s.Description = n.Description
	s.Title = n.Title
	s.Default = n.Default
	s.Example = n.Example

	// anyOf: detect the nullable shorthand [T, null] or [null, T] vs a real union.
	if len(n.AnyOf) > 0 {
		if nonNull := nullableFrom(n.AnyOf); nonNull != nil {
			if err := s.fromNode(nonNull); err != nil {
				return err
			}
			s.Nullable = true
			return nil
		}
		s.AnyOf = make([]*Schema, len(n.AnyOf))
		for i := range n.AnyOf {
			sub := &Schema{}
			if err := sub.fromNode(&n.AnyOf[i]); err != nil {
				return fmt.Errorf("anyOf[%d]: %w", i, err)
			}
			s.AnyOf[i] = sub
		}
		return nil
	}

	// Nullable comes from the "nullable" field or the ["T","null"] type-array form.
	s.Nullable = n.Nullable || n.Type.nullable
	if len(n.Enum) > 0 {
		s.Enum = make([]string, len(n.Enum))
		for i, raw := range n.Enum {
			// Gemini's Schema.Enum is []string, but integer enums in JSON Schema are
			// encoded as numbers. Convert numbers to their string representation so that
			// e.g. {"type":"integer","enum":[101,201]} becomes Enum:["101","201"].
			// Use json.Decoder with UseNumber so large integers are not rounded via float64.
			dec := json.NewDecoder(bytes.NewReader(raw))
			dec.UseNumber()
			var v any
			if err := dec.Decode(&v); err != nil {
				return fmt.Errorf("enum[%d]: %w", i, err)
			}
			switch tv := v.(type) {
			case string:
				s.Enum[i] = tv
			case json.Number:
				s.Enum[i] = tv.String()
			default:
				return fmt.Errorf("enum[%d]: unsupported type %T, must be string or number", i, v)
			}
		}
	}
	s.Required = n.Required
	s.MinLength = n.MinLength
	s.MaxLength = n.MaxLength
	s.MinItems = n.MinItems
	s.MaxItems = n.MaxItems
	s.Minimum = n.Minimum
	s.Maximum = n.Maximum
	s.Format = Format(n.Format)

	switch n.Type.value {
	case "string":
		s.Type = TypeString
	case "integer":
		s.Type = TypeInteger
	case "number":
		s.Type = TypeNumber
	case "boolean":
		s.Type = TypeBoolean
	case "null":
		s.Type = TypeNull
	case "array":
		s.Type = TypeArray
		if n.Items != nil {
			s.Items = &Schema{}
			if err := s.Items.fromNode(n.Items); err != nil {
				return fmt.Errorf("items: %w", err)
			}
		}
	case "object":
		s.Type = TypeObject
		if len(n.Properties) > 0 {
			s.Properties = make(map[string]Schema, len(n.Properties))
			for k := range n.Properties {
				prop := Schema{}
				v := n.Properties[k]
				if err := prop.fromNode(&v); err != nil {
					return fmt.Errorf("property %q: %w", k, err)
				}
				s.Properties[k] = prop
			}
			// PropertyOrdering is not set: JSON Schema has no standard ordering field,
			// and map iteration is unordered. Callers that need deterministic field
			// ordering in Gemini structured output must set PropertyOrdering themselves.
		}
	case "":
		// No type — valid alongside anyOf/allOf or for schemaless nodes.
	default:
		return fmt.Errorf("unsupported JSON Schema type: %q", n.Type.value)
	}

	return nil
}

// nullableFrom returns the non-null node when anyOf encodes a nullable type as [T, null] or [null, T].
// Only the 2-element form is detected; a 3+-element anyOf that includes null is treated as a real union,
// leaving the null arm in place. invopop/jsonschema always emits the 2-element form for pointer types.
func nullableFrom(anyOf []jsonSchemaNode) *jsonSchemaNode {
	if len(anyOf) != 2 {
		return nil
	}
	if anyOf[0].Type.value == "null" {
		return &anyOf[1]
	}
	if anyOf[1].Type.value == "null" {
		return &anyOf[0]
	}
	return nil
}

// Content and Part types.

// Blob is documented at https://ai.google.dev/api/caching?hl=en#Blob
type Blob struct {
	MimeType string `json:"mimeType,omitzero"`
	Data     []byte `json:"data,omitzero"`
}

// FileData is documented at https://ai.google.dev/api/caching?hl=en#FileData
type FileData struct {
	MimeType string `json:"mimeType,omitzero"`
	FileURI  string `json:"fileUri,omitzero"`
}

// StructValue is documented at https://protobuf.dev/reference/protobuf/google.protobuf/#struct
type StructValue map[string]json.RawMessage

// Value is documented at https://protobuf.dev/reference/protobuf/google.protobuf/#value
type Value struct {
	NullValue   int64       `json:"null_value,omitzero"`
	NumberValue float64     `json:"number_value,omitzero"`
	StringValue string      `json:"string_value,omitzero"`
	BoolValue   bool        `json:"bool_value,omitzero"`
	StructValue StructValue `json:"struct_value,omitzero"`
	ListValue   []Value     `json:"list_value,omitzero"`
}

// SafetySetting is documented at https://ai.google.dev/api/generate-content?hl=en#v1beta.SafetySetting
type SafetySetting struct {
	Category  string `json:"category"`  // https://ai.google.dev/api/generate-content?hl=en#v1beta.HarmCategory
	Threshold int64  `json:"threshold"` // https://ai.google.dev/api/generate-content?hl=en#HarmBlockThreshold
}

// Tool is documented at https://ai.google.dev/api/caching?hl=en#Tool
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
	CodeExecution *struct{} `json:"codeExecution,omitzero"`
	URLContext    *struct{} `json:"urlContext,omitzero"`
	// GoogleSearch presence signifies that it should be enabled,
	GoogleSearch *GoogleSearch `json:"googleSearch,omitzero"`
	// FileSearch enables file search tool.
	FileSearch *FileSearch `json:"fileSearch,omitzero"`
}

// GoogleSearch is "documented" at https://ai.google.dev/gemini-api/docs/google-search
type GoogleSearch struct{}

// FileSearch is documented at https://ai.google.dev/gemini-api/docs/file-search
type FileSearch struct {
	FileSearchStoreNames []string `json:"fileSearchStoreNames,omitzero"`
	TopK                 int32    `json:"topK,omitzero"`
	MetadataFilter       string   `json:"metadataFilter,omitzero"`
}

// FunctionDeclaration is documented at https://ai.google.dev/api/caching?hl=en#FunctionDeclaration
type FunctionDeclaration struct {
	Name        string `json:"name,omitzero"`
	Description string `json:"description,omitzero"`
	Parameters  Schema `json:"parameters,omitzero"`
	Response    Schema `json:"response,omitzero"`
}

// ToolMode is documented at https://ai.google.dev/api/caching?hl=en#Mode_1
type ToolMode string

const (
	// ToolModeUnspecified is an unspecified function calling mode. This value should not be used.
	ToolModeUnspecified ToolMode = "" // "MODE_UNSPECIFIED"
	// ToolModeAuto is the default model behavior, model decides to predict either a function call or a natural language response.
	ToolModeAuto ToolMode = "AUTO"
	// ToolModeAny means the model is constrained to always predicting a function call only.
	ToolModeAny ToolMode = "ANY"
	// ToolModeNone means the model will not predict any function call.
	ToolModeNone ToolMode = "NONE"
	// ToolModeValidated means the model decides to predict either a function call or a natural language response, but will validate
	// function calls with constrained decoding.
	ToolModeValidated ToolMode = "VALIDATED"
)

// ToolConfig is documented at https://ai.google.dev/api/caching?hl=en#ToolConfig
type ToolConfig struct {
	// https://ai.google.dev/api/caching?hl=en#FunctionCallingConfig
	FunctionCallingConfig struct {
		Mode                 ToolMode `json:"mode,omitzero"`
		AllowedFunctionNames []string `json:"allowedFunctionNames,omitzero"`
	} `json:"functionCallingConfig,omitzero"`
}

// Modality is documented at https://ai.google.dev/api/generate-content#Modality
type Modality string

// Modality values.
const (
	ModalityUnspecified Modality = "" // "MODALITY_UNSPECIFIED"
	ModalityAudio       Modality = "AUDIO"
	ModalityImage       Modality = "IMAGE"
	ModalityText        Modality = "TEXT"
)

// MediaResolution is documented at https://ai.google.dev/api/generate-content#MediaResolution
type MediaResolution string

// Media resolution values.
const (
	MediaResolutionUnspecified MediaResolution = ""       // "MEDIA_RESOLUTION_UNSPECIFIED"
	MediaResolutionLow         MediaResolution = "LOW"    // 64 tokens
	MediaResolutionMedium      MediaResolution = "MEDIUM" // 256 tokens
	MediaResolutionHigh        MediaResolution = "HIGH"   // zoomed reframing with 256 tokens
)

// Request and response types.

// ChatRequest is documented at https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
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
		Logprobs                   int64      `json:"logprobs,omitzero"`
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
		// must NOT be present for non-thinking models.
		ThinkingConfig  *ThinkingConfig `json:"thinkingConfig,omitempty"`
		MediaResolution MediaResolution `json:"mediaResolution,omitzero"`
	} `json:"generationConfig,omitzero"`
	CachedContent string `json:"cachedContent,omitzero"` // Name of the cached content with "cachedContents/" prefix.
}

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	// There's no field to set, the URL is different.
}

// Content is the equivalent of Message for other providers.
// https://ai.google.dev/api/caching?hl=en#Content
type Content struct {
	Role string `json:"role,omitzero"` // "user", "model"
	// Parts can be both content and tool calls.
	Parts []Part `json:"parts"`
}

// Part is a union that only has one of the field set.
// Part is the equivalent of Content for other providers.
//
// https://ai.google.dev/api/caching?hl=en#Part
type Part struct {
	Thought          bool   `json:"thought,omitzero"`
	ThoughtSignature []byte `json:"thoughtSignature,omitzero"`

	// Union:
	Text                string              `json:"text,omitzero"`
	InlineData          Blob                `json:"inlineData,omitzero"` // Uploaded with /v1beta/cachedContents. Content is deleted after 1 hour.
	FunctionCall        FunctionCall        `json:"functionCall,omitzero"`
	FunctionResponse    FunctionResponse    `json:"functionResponse,omitzero"`
	FileData            FileData            `json:"fileData,omitzero"`            // Uploaded with /upload/v1beta/files. Files are deleted after 2 days.
	ExecutableCode      ExecutableCode      `json:"executableCode,omitzero"`      // TODO: Map to genai types.
	CodeExecutionResult CodeExecutionResult `json:"codeExecutionResult,omitzero"` // TODO: Map to genai types.

	// Union:
	VideoMetadata VideoMetadata `json:"videoMetadata,omitzero"`
}

// FunctionCall is documented at https://ai.google.dev/api/caching?hl=en#FunctionCall
type FunctionCall struct {
	ID               string      `json:"id,omitzero"`
	Name             string      `json:"name,omitzero"`
	Args             StructValue `json:"args,omitzero"`
	ThoughtSignature []byte      `json:"thoughtSignature,omitzero"` // Returned by some models with thinking
}

// FunctionResponse is documented at https://ai.google.dev/api/caching?hl=en#FunctionResponse
type FunctionResponse struct {
	ID       string      `json:"id,omitzero"`
	Name     string      `json:"name,omitzero"`
	Response StructValue `json:"response,omitzero"`
}

// ExecutableCode is documented at https://ai.google.dev/api/caching?hl=en#ExecutableCode
type ExecutableCode struct {
	Language string `json:"language,omitzero"` // Only PYTHON is supported as of March 2025.
	Code     string `json:"code,omitzero"`
}

// CodeExecutionResult is documented at https://ai.google.dev/api/caching?hl=en#CodeExecutionResult
type CodeExecutionResult struct {
	Outcome string `json:"outcome,omitzero"` // One of OUTCOME_UNSPECIFIED, OUTCOME_OK, OUTCOME_FAILED, OUTCOME_DEADLINE_EXCEEDED
	Output  string `json:"output,omitzero"`
}

// VideoMetadata is documented at https://ai.google.dev/api/caching#VideoMetadata
type VideoMetadata struct {
	StartOffset Duration `json:"startOffset,omitzero"`
	EndOffset   Duration `json:"endOffset,omitzero"`
	FPS         int64    `json:"fps,omitzero"` // Default: 1.0, range ]0, 24]
}

// ThinkingConfig is documented at https://ai.google.dev/api/generate-content?hl=en#ThinkingConfig
// See https://ai.google.dev/gemini-api/docs/thinking#rest
type ThinkingConfig struct {
	// IncludeThoughts has no effect since January 2025 according to
	// https://discuss.ai.google.dev/t/thoughts-are-missing-cot-not-included-anymore/63653/13
	IncludeThoughts bool  `json:"includeThoughts"` // Must not be omitted.
	ThinkingBudget  int64 `json:"thinkingBudget"`  // Must not be omitted. [0, 24576]
}

// ChatResponse is documented at https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerateContentResponse
type ChatResponse struct {
	Candidates     []ResponseCandidate `json:"candidates"`
	PromptFeedback struct{}            `json:"promptFeedback,omitzero"`
	UsageMetadata  UsageMetadata       `json:"usageMetadata"`
	ModelVersion   string              `json:"modelVersion"`
	ResponseID     string              `json:"responseId"`
}

// ResponseCandidate is described at https://ai.google.dev/api/generate-content?hl=en#v1beta.Candidate
//
// It is essentially a "Message".
type ResponseCandidate struct {
	Content       Content      `json:"content"`
	FinishReason  FinishReason `json:"finishReason"`
	FinishMessage string       `json:"finishMessage,omitzero"` // Newer models with thinking return this
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
	GroundingMetadata  GroundingMetadata  `json:"groundingMetadata"`
	UrlContextMetadata UrlContextMetadata `json:"urlContextMetadata"`
	AvgLogprobs        float64            `json:"avgLogprobs"`
	LogprobsResult     LogprobsResult     `json:"logprobsResult"`
	Index              int64              `json:"index"`
}

// Grounding types.

// GroundingChunkWeb is documented at https://ai.google.dev/api/generate-content?hl=en#Web
type GroundingChunkWeb struct {
	URI   string `json:"uri,omitzero"`
	Title string `json:"title,omitzero"`
}

// GroundingChunkRetrievedContext is documented at https://ai.google.dev/api/generate-content?hl=en#RetrievedContext
type GroundingChunkRetrievedContext struct {
	URI          string `json:"uri,omitzero"`
	Title        string `json:"title,omitzero"`
	Text         string `json:"text,omitzero"`
	DocumentName string `json:"documentName,omitzero"`
	// FileSearchStore is the store name that sourced this chunk. Not in the official Go SDK but returned by the API.
	FileSearchStore string `json:"fileSearchStore,omitzero"`
}

// GroundingChunk is documented at https://ai.google.dev/api/generate-content?hl=en#GroundingChunk
type GroundingChunk struct {
	Web              GroundingChunkWeb              `json:"web,omitzero"`
	RetrievedContext GroundingChunkRetrievedContext `json:"retrievedContext,omitzero"`
}

// Segment is documented at https://ai.google.dev/api/generate-content?hl=en#Segment
type Segment struct {
	PartIndex  int64  `json:"partIndex,omitzero"`
	StartIndex int64  `json:"startIndex,omitzero"`
	EndIndex   int64  `json:"endIndex,omitzero"`
	Text       string `json:"text,omitzero"`
}

// GroundingSupport is documented at https://ai.google.dev/api/generate-content?hl=en#GroundingSupport
type GroundingSupport struct {
	GroundingChunkIndices []int64   `json:"groundingChunkIndices,omitzero"`
	ConfidenceScores      []float64 `json:"confidenceScores,omitzero"`
	Segment               Segment   `json:"segment,omitzero"`
}

// SearchEntryPoint is documented at https://ai.google.dev/api/generate-content?hl=en#SearchEntryPoint
type SearchEntryPoint struct {
	RenderedContent string `json:"renderedContent,omitzero"`
	SDKBlob         []byte `json:"sdkBlob,omitzero"` // JSON encoded list of (search term,search url) results
}

// RetrievalMetadata is documented at https://ai.google.dev/api/generate-content?hl=en#RetrievalMetadata
type RetrievalMetadata struct {
	GoogleSearchDynamicRetrievalScore float64 `json:"googleSearchDynamicRetrievalScore,omitzero"`
}

// GroundingMetadata is documented at https://ai.google.dev/api/generate-content?hl=en#GroundingMetadata
type GroundingMetadata struct {
	GroundingChunks   []GroundingChunk   `json:"groundingChunks,omitzero"`
	GroundingSupports []GroundingSupport `json:"groundingSupports,omitzero"`
	WebSearchQueries  []string           `json:"webSearchQueries,omitzero"`
	SearchEntryPoint  SearchEntryPoint   `json:"searchEntryPoint,omitzero"`
	RetrievalMetadata RetrievalMetadata  `json:"retrievalMetadata,omitzero"`
}

// IsZero reports whether the value is zero.
func (g *GroundingMetadata) IsZero() bool {
	return len(g.GroundingChunks) == 0 && len(g.GroundingSupports) == 0 && len(g.WebSearchQueries) == 0 && len(g.SearchEntryPoint.SDKBlob) == 0 && g.RetrievalMetadata.GoogleSearchDynamicRetrievalScore == 0
}

// URL context types.

// UrlContextMetadata is documented at https://ai.google.dev/gemini-api/docs/url-context
type UrlContextMetadata struct {
	UrlMetadata []UrlMetadataEntry `json:"urlMetadata,omitzero"`
}

// UrlMetadataEntry is a single URL retrieval result.
type UrlMetadataEntry struct {
	RetrievedUrl       string `json:"retrievedUrl,omitzero"`
	UrlRetrievalStatus string `json:"urlRetrievalStatus,omitzero"`
}

// Logprobs types.

// LogprobsResult is documented at https://ai.google.dev/api/generate-content#LogprobsResult
type LogprobsResult struct {
	TopCandidates    []TopCandidate   `json:"topCandidates"`
	ChosenCandidates []TokenCandidate `json:"chosenCandidates"`
}

// TopCandidate is documented at https://ai.google.dev/api/generate-content#TopCandidates
type TopCandidate struct {
	Candidates []TokenCandidate `json:"candidates"`
}

// TokenCandidate is documented at https://ai.google.dev/api/generate-content#TokenCandidate
type TokenCandidate struct {
	Token          string  `json:"token"`
	TokenID        int64   `json:"tokenId"`
	LogProbability float64 `json:"logProbability"`
}

// FinishReason is documented at https://ai.google.dev/api/generate-content?hl=en#FinishReason
type FinishReason string

const (
	// FinishStop is the natural stop point of the model or provided stop sequence.
	FinishStop FinishReason = "STOP"
	// FinishMaxTokens means the maximum number of tokens as specified in the request was reached.
	FinishMaxTokens FinishReason = "MAX_TOKENS"
	// FinishSafety means the response candidate content was flagged for safety reasons.
	FinishSafety FinishReason = "SAFETY"
	// FinishRecitation means the response candidate content was flagged for recitation reasons.
	FinishRecitation FinishReason = "RECITATION"
	// FinishLanguage means the response candidate content was flagged for using an unsupported language.
	FinishLanguage FinishReason = "LANGUAGE"
	// FinishOther is an unknown reason.
	FinishOther FinishReason = "OTHER"
	// FinishBlocklist means token generation stopped because the content contains forbidden terms.
	FinishBlocklist FinishReason = "BLOCKLIST"
	// FinishProhibitedContent means token generation stopped for potentially containing prohibited content.
	FinishProhibitedContent FinishReason = "PROHIBITED_CONTENT"
	// FinishSPII means token generation stopped because the content potentially contains Sensitive Personally Identifiable Information.
	FinishSPII FinishReason = "SPII"
	// FinishMalformed means the function call generated by the model is invalid.
	FinishMalformed FinishReason = "MALFORMED_FUNCTION_CALL"
	// FinishImageSafety means token generation stopped because generated images contain safety violations.
	FinishImageSafety FinishReason = "IMAGE_SAFETY"
)

// ToFinishReason converts to a genai.FinishReason.
func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishMaxTokens:
		return genai.FinishedLength
	case FinishSafety, FinishBlocklist, FinishProhibitedContent, FinishSPII, FinishImageSafety:
		// TODO: Confirm. We lose on nuance here but does it matter?
		return genai.FinishedContentFilter
	case FinishRecitation, FinishLanguage, FinishOther, FinishMalformed:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(strings.ToLower(string(f)))
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(strings.ToLower(string(f)))
	}
}

// Usage types.

// UsageMetadata is documented at https://ai.google.dev/api/generate-content?hl=en#UsageMetadata
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
	ServiceTier                string               `json:"serviceTier,omitzero"`
}

// ModalityTokenCount is documented at
// https://ai.google.dev/api/generate-content?hl=en#v1beta.ModalityTokenCount
type ModalityTokenCount struct {
	Modality   Modality `json:"modality"`
	TokenCount int64    `json:"tokenCount"`
}

// Streaming types.

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse struct {
	Candidates []struct {
		Content            Content            `json:"content"`
		FinishReason       FinishReason       `json:"finishReason"`
		FinishMessage      string             `json:"finishMessage"`
		Index              int64              `json:"index"`
		GroundingMetadata  GroundingMetadata  `json:"groundingMetadata"`
		UrlContextMetadata UrlContextMetadata `json:"urlContextMetadata"`
	} `json:"candidates"`
	UsageMetadata UsageMetadata `json:"usageMetadata"`
	ModelVersion  string        `json:"modelVersion"`
	ResponseID    string        `json:"responseId"`
}

// Image types.

// ImageRequest is not really documented. It is used for both image and video generation.
//
// See https://ai.google.dev/gemini-api/docs/imagen#imagen
//
// See https://pkg.go.dev/google.golang.org/genai#GenerateImagesConfig for image generation and
// https://ai.google.dev/gemini-api/docs/video for video generation.
//
// See generateImagesConfigToMldev() in https://github.com/googleapis/go-genai/blob/main/models.go
// or GenerateImagesConfig and generateImagesInternal() in
// https://github.com/googleapis/js-genai/blob/main/src/models.ts and generateImagesParametersToMldev() and
// generateImagesConfigToMldev()
//
// As of 2025-08, Google Vertex AI API supports way more features than the Gemini API. For example: specifying
// the video's last frame, whether to generate audio, compression level, resolution, randomness seed, sending
// updates to a pubsub topic (instead of polling), etc.
type ImageRequest struct {
	// There should be only one instance.
	Instances  []ImageInstance `json:"instances"`
	Parameters ImageParameters `json:"parameters"`
}

// ImageInstance is not really documented, better to read the SDK code and guess, since they don't use proper
// structs there either and it's all hand written.
type ImageInstance struct {
	Prompt string `json:"prompt"`
	Image  struct {
		BytesBase64Encoded []byte `json:"bytesBase64Encoded,omitzero"`
		MimeType           string `json:"mimeType,omitzero"`
	} `json:"image,omitzero"`
}

// ImageParameters is not really documented, better to read the SDK code and guess, since they don't use proper
// structs there either and it's all hand written.
type ImageParameters struct {
	// Both image and video:
	SampleCount      int64  `json:"sampleCount,omitzero"` // Number of images to generate. Default to 4.
	AspectRatio      string `json:"aspectRatio,omitzero"` // "1:1", "3:4", "4:3", "9:16", and "16:9".
	PersonGeneration string `json:"personGeneration"`     // "dont_allow", "allow_adult", "allow_all" (not valid in EU, UK, CH, MENA locations)
	EnhancePrompt    bool   `json:"enhancePrompt,omitzero"`

	// Image only.
	GuidanceScale           float64     `json:"guidanceScale,omitzero"`
	Seed                    int64       `json:"seed,omitzero"`
	SafetySetting           string      `json:"safetySetting,omitzero"`
	IncludeSafetyAttributes bool        `json:"includeSafetyAttributes,omitzero"`
	IncludeRAIReason        bool        `json:"includeRAIReason,omitzero"`
	Language                string      `json:"language,omitzero"`
	OutputOptions           ImageOutput `json:"outputOptions,omitzero"`
	AddWatermark            bool        `json:"addWatermark,omitzero"`
	SampleImageSize         string      `json:"SampleImageSize,omitzero"`
	// VertexAI only:
	// OutputGCSURI             string  `json:"outputGcsUri,omitzero"`
	// NegativePrompt           string  `json:"negativePrompt,omitzero"`

	// Video only.
	DurationSeconds int64  `json:"durationSeconds,omitzero"`
	NegativePrompt  string `json:"negativePrompt,omitzero"`
	// VertexAI only:
	// FPS int64 `json:"fps,omitzero"`
	// Seed int64 `json:"seed,omitzero"`
	// Resolution string `json:"resolution,omitzero"`
	// PubSubTopic string `json:"pubSubTopic,omitzero"`
	// GenerateAudio string `json:"generateAudio,omitzero"`
	// LastFrame string `json:"lastFrame,omitzero"`
	// CompressionQuality string `json:"compressionQuality,omitzero"`
}

// ImageOutput is the provider-specific image output configuration.
type ImageOutput struct {
	MimeType           string  `json:"mimeType,omitzero"` // "image/jpeg"
	CompressionQuality float64 `json:"compressionQuality,omitzero"`
}

// ImageResponse is the provider-specific image generation response.
type ImageResponse struct {
	Predictions []struct {
		MimeType         string `json:"mimeType"`
		SafetyAttributes struct {
			Categories []string  `json:"categories"`
			Scores     []float64 `json:"scores"`
		} `json:"safetyAttributes"`
		BytesBase64Encoded []byte `json:"bytesBase64Encoded"`
		ContentType        string `json:"contentType"` // "Positive Prompt"
	} `json:"predictions"`
}

// Operation types.

// Operation is documented at https://ai.google.dev/api/batch-mode#Operation
//
// See generateVideosResponseFromMldev in js-genai for more details.
type Operation struct {
	Name     string                     `json:"name"`
	Metadata map[string]json.RawMessage `json:"metadata"`
	Done     bool                       `json:"done"`
	// One of the following:
	Error    Status `json:"error"`
	Response struct {
		Type string `json:"@type"`
		// Video generation response.
		GenerateVideoResponse struct {
			GeneratedSamples []struct {
				Video struct {
					URI        string `json:"uri"`
					VideoBytes []byte `json:"videoBytes"` // Not set in Gemini API
					MimeType   string `json:"mimeType"`   // Not set in Gemini API
				} `json:"video"`
				RAIMediaFilteredCount   int64    `json:"raiMediaFilteredCount"`
				RAIMediaFilteredReasons []string `json:"raiMediaFilteredReasons"`
			} `json:"generatedSamples"`
		} `json:"generateVideoResponse"`
		// File search store document upload response.
		DocumentName string `json:"documentName,omitzero"`
		Parent       string `json:"parent,omitzero"`
		MimeType     string `json:"mimeType,omitzero"`
		SizeBytes    string `json:"sizeBytes,omitzero"`
	} `json:"response"`
}

// Status is documented at https://ai.google.dev/api/files#v1beta.Status
type Status struct {
	Code    int64                      `json:"code"`
	Message string                     `json:"message"`
	Details map[string]json.RawMessage `json:"details"`
}

// File types.

// FileState represents the processing state of an uploaded file.
//
// https://ai.google.dev/api/files#State
type FileState string

// File state values.
const (
	FileStateUnspecified FileState = "STATE_UNSPECIFIED"
	FileStateProcessing  FileState = "PROCESSING"
	FileStateActive      FileState = "ACTIVE"
	FileStateFailed      FileState = "FAILED"
)

// FileMetadata is the metadata for an uploaded file.
//
// https://ai.google.dev/api/files#File
type FileMetadata struct {
	Name           string    `json:"name,omitzero"`
	DisplayName    string    `json:"displayName,omitzero"`
	MimeType       string    `json:"mimeType,omitzero"`
	SizeBytes      int64     `json:"sizeBytes,omitzero,string"`
	CreateTime     time.Time `json:"createTime,omitzero"`
	UpdateTime     time.Time `json:"updateTime,omitzero"`
	ExpirationTime time.Time `json:"expirationTime,omitzero"`
	SHA256Hash     string    `json:"sha256Hash,omitzero"`
	URI            string    `json:"uri,omitzero"`
	DownloadURI    string    `json:"downloadUri,omitzero"`
	State          FileState `json:"state,omitzero"`
	Source         string    `json:"source,omitzero"`
	Error          *Status   `json:"error,omitzero"`
}

// FileListResponse is the response from listing files.
//
// https://ai.google.dev/api/files#method:-files.list
type FileListResponse struct {
	Files         []FileMetadata `json:"files,omitzero"`
	NextPageToken string         `json:"nextPageToken,omitzero"`
}

// File search store types.

// FileSearchStoreState represents the state of a file search store.
type FileSearchStoreState string

// File search store state values.
const (
	FileSearchStoreStateUnspecified FileSearchStoreState = "STATE_UNSPECIFIED"
	FileSearchStoreStateActive      FileSearchStoreState = "STATE_ACTIVE"
)

// FileSearchStore represents a file search store resource.
//
// https://ai.google.dev/gemini-api/docs/file-search
type FileSearchStore struct {
	Name                 string               `json:"name,omitzero"`
	DisplayName          string               `json:"displayName,omitzero"`
	CreateTime           time.Time            `json:"createTime,omitzero"`
	UpdateTime           time.Time            `json:"updateTime,omitzero"`
	State                FileSearchStoreState `json:"state,omitzero"`
	ActiveDocumentsCount int64                `json:"activeDocumentsCount,omitzero,string"`
	SizeBytes            int64                `json:"sizeBytes,omitzero,string"`
	EmbeddingModel       string               `json:"embeddingModel,omitzero"`
}

// FileSearchStoreListResponse is the response from listing file search stores.
type FileSearchStoreListResponse struct {
	FileSearchStores []FileSearchStore `json:"fileSearchStores,omitzero"`
	NextPageToken    string            `json:"nextPageToken,omitzero"`
}

// FileSearchStoreDocumentState represents the state of a document in a file search store.
type FileSearchStoreDocumentState string

// File search store document state values.
const (
	FileSearchStoreDocumentStateUnspecified FileSearchStoreDocumentState = "STATE_UNSPECIFIED"
	FileSearchStoreDocumentStateProcessing  FileSearchStoreDocumentState = "STATE_PROCESSING"
	FileSearchStoreDocumentStateActive      FileSearchStoreDocumentState = "STATE_ACTIVE"
	FileSearchStoreDocumentStateFailed      FileSearchStoreDocumentState = "STATE_FAILED"
)

// FileSearchStoreDocument represents a document within a file search store.
//
// https://ai.google.dev/gemini-api/docs/file-search
type FileSearchStoreDocument struct {
	Name        string                       `json:"name,omitzero"`
	DisplayName string                       `json:"displayName,omitzero"`
	State       FileSearchStoreDocumentState `json:"state,omitzero"`
	CreateTime  time.Time                    `json:"createTime,omitzero"`
	UpdateTime  time.Time                    `json:"updateTime,omitzero"`
	SizeBytes   int64                        `json:"sizeBytes,omitzero,string"`
	MimeType    string                       `json:"mimeType,omitzero"`
	Error       *Status                      `json:"error,omitzero"`
}

// FileSearchStoreDocumentListResponse is the response from listing documents in a file search store.
type FileSearchStoreDocumentListResponse struct {
	Documents     []FileSearchStoreDocument `json:"documents,omitzero"`
	NextPageToken string                    `json:"nextPageToken,omitzero"`
}

// Caching types.

// CachedContent is documented at https://ai.google.dev/api/caching?hl=en#CachedContent
// https://ai.google.dev/api/caching?hl=en#request-body
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

// GetID implements genai.Model.
func (c *CachedContent) GetID() string {
	return c.Name
}

// GetDisplayName implements genai.CacheItem.
func (c *CachedContent) GetDisplayName() string {
	return c.DisplayName
}

// GetExpiry implements genai.CacheItem.
func (c *CachedContent) GetExpiry() time.Time {
	return c.ExpireTime
}

// Expiration must be embedded. Only one of the fields can be set.
type Expiration struct {
	ExpireTime time.Time `json:"expireTime,omitzero"` // ISO 8601
	TTL        Duration  `json:"ttl,omitzero"`        // Duration; input only
}

// Duration is a JSON-serializable time.Duration.
type Duration time.Duration

// IsZero reports whether the value is zero.
func (d *Duration) IsZero() bool {
	return *d == 0
}

// MarshalJSON implements json.Marshaler.
func (d *Duration) MarshalJSON() ([]byte, error) {
	v := time.Duration(*d)
	return json.Marshal(fmt.Sprintf("%1.fs", v.Seconds()))
}

// UnmarshalJSON implements json.Unmarshaler.
func (d *Duration) UnmarshalJSON(b []byte) error {
	s := ""
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

// CachingUsageMetadata is documented at https://ai.google.dev/api/caching?hl=en#UsageMetadata
type CachingUsageMetadata struct {
	TotalTokenCount int64 `json:"totalTokenCount"`
}

// Model types.

// Model is documented at  https://ai.google.dev/api/models#Model
type Model struct {
	Name                       string   `json:"name"`
	BaseModelID                string   `json:"baseModelId"`
	Version                    string   `json:"version"`
	DisplayName                string   `json:"displayName"`
	Description                string   `json:"description"`
	InputTokenLimit            int64    `json:"inputTokenLimit"`
	OutputTokenLimit           int64    `json:"outputTokenLimit"`
	SupportedGenerationMethods []string `json:"supportedGenerationMethods"` // "batchGenerateContent", "bidiGenerateContent", "createCachedContent", "countTokens", "countTextTokens", "embedText", "generateContent", "predict", "predictLongRunning"
	Temperature                float64  `json:"temperature"`
	MaxTemperature             float64  `json:"maxTemperature"`
	TopP                       float64  `json:"topP"`
	TopK                       int64    `json:"topK"`
	Thinking                   bool     `json:"thinking"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return strings.TrimPrefix(m.Name, "models/")
}

func (m *Model) String() string {
	if m.Description == "" {
		return fmt.Sprintf("%s: %s Context: %d/%d", m.GetID(), m.DisplayName, m.InputTokenLimit, m.OutputTokenLimit)
	}
	d := strings.TrimSuffix(m.Description, ".")
	return fmt.Sprintf("%s: %s (%s) Context: %d/%d", m.GetID(), m.DisplayName, d, m.InputTokenLimit, m.OutputTokenLimit)
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return m.InputTokenLimit
}

// ModelsResponse represents the response structure for Gemini models listing.
type ModelsResponse struct {
	Models        []Model `json:"models"`
	NextPageToken string  `json:"nextPageToken"`
}

// ToModels converts Gemini models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Models))
	for i := range r.Models {
		models[i] = &r.Models[i]
	}
	return models
}

// CountTokensResponse is the response from the countTokens API.
//
// https://ai.google.dev/api/tokens#v1beta.CountTokensResponse
type CountTokensResponse struct {
	TotalTokens             int64 `json:"totalTokens"`
	CachedContentTokenCount int64 `json:"cachedContentTokenCount,omitzero"`
}

// Error types.

// ErrorResponse represents the response structure for Gemini errors.
//
// It is returned as an error.
type ErrorResponse struct {
	ErrorVal ErrorResponseError `json:"error"`
}

func (e *ErrorResponse) Error() string {
	return fmt.Sprintf("%s (%d): %s", e.ErrorVal.Status, e.ErrorVal.Code, strings.TrimSpace(e.ErrorVal.Message))
}

// IsAPIError implements base.ErrorResponseI.
func (e *ErrorResponse) IsAPIError() bool {
	return true
}

// ErrorResponseError is the nested error in an error response.
type ErrorResponseError struct {
	Code    int64  `json:"code"` // 429
	Message string `json:"message"`
	Status  string `json:"status"` // "RESOURCE_EXHAUSTED"
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

		// "type.googleapis.com/google.rpc.QuotaFailure"
		Violations []struct {
			// "generativelanguage.googleapis.com/generate_requests_per_model_per_day"
			// "generativelanguage.googleapis.com/generate_requests_per_model"
			// "generativelanguage.googleapis.com/generate_content_paid_tier_input_token_count"
			QuotaMetric string `json:"quotaMetric"`
			// "GenerateRequestsPerDayPerProjectPerModel"
			// "GenerateRequestsPerMinutePerProjectPerModel"
			// "GenerateContentPaidTierInputTokensPerModelPerMinute"
			QuotaID         string `json:"quotaId"`
			QuotaDimensions struct {
				Location string `json:"location"` // "global"
				Model    string `json:"model"`    // model name
			} `json:"quotaDimensions"`
		} `json:"violations"`

		// "type.googleapis.com/google.rpc.Help"
		Links []struct {
			Description string `json:"description"`
			URL         string `json:"url"`
		} `json:"links"`

		// Type == "type.googleapis.com/google.rpc.RetryInfo"
		RetryDelay string `json:"retryDelay"` // "28s"
	} `json:"details"`
}
