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
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path"
	"reflect"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Gemini.
//
// See the following multi-modal sources:
//
// https://ai.google.dev/gemini-api/docs/image-understanding?hl=en&lang=rest#supported-formats
// https://ai.google.dev/gemini-api/docs/video-understanding?hl=en#supported-formats
// https://ai.google.dev/gemini-api/docs/audio?hl=en#supported-formats
// https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#technical-details
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// Options defines Gemini specific options.
type Options struct {
	// ThinkingBudget is the maximum number of tokens the LLM can use to reason about the answer.
	//
	// From https://ai.google.dev/gemini-api/docs/thinking#set-budget
	//
	// # gemini-2.5-pro
	//
	// - Default: Dynamic thinking, model decides when and how much to think
	// - Range: 128 to 32768
	// - Cannot disable thinking
	// - Dynamic thinking: -1
	//
	// # gemini-2.5-flash
	//
	// - Default: Dynamic thinking, model decides when and how much to think
	// - Range: 0 to 24576
	// - Disable thinking with 0
	// - Dynamic thinking: -1
	//
	// # gemini-2.5-flash-lite
	//
	// - Default: Model does not think
	// - Range: 512 to 24576
	// - Disable thinking with 0
	// - Dynamic thinking: -1
	ThinkingBudget int64
}

func (o *Options) Validate() error {
	return nil
}

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
type StructValue map[string]any

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
	CodeExecution struct{} `json:"codeExecution,omitzero"`
	// GoogleSearch presence signifies that it should be enabled,
	GoogleSearch *GoogleSearch `json:"googleSearch,omitzero"`
}

// GoogleSearch is "documented" at https://ai.google.dev/gemini-api/docs/google-search
type GoogleSearch struct{}

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

const (
	ModalityUnspecified Modality = "" // "MODALITY_UNSPECIFIED"
	ModalityAudio       Modality = "AUDIO"
	ModalityImage       Modality = "IMAGE"
	ModalityText        Modality = "TEXT"
)

// MediaResolution is documented at https://ai.google.dev/api/generate-content#MediaResolution
type MediaResolution string

const (
	MediaResolutionUnspecified MediaResolution = ""       // "MEDIA_RESOLUTION_UNSPECIFIED"
	MediaResolutionLow         MediaResolution = "LOW"    // 64 tokens
	MediaResolutionMedium      MediaResolution = "MEDIUM" // 256 tokens
	MediaResolutionHigh        MediaResolution = "HIGH"   // zoomed reframing with 256 tokens
)

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

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.Options) error {
	// Validate messages
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string

	for _, opt := range opts {
		switch v := opt.(type) {
		case *Options:
			// Accept both positive numbers and -1 (dynamic thinking)
			if v.ThinkingBudget != 0 {
				// https://ai.google.dev/gemini-api/docs/thinking
				c.GenerationConfig.ThinkingConfig = &ThinkingConfig{
					IncludeThoughts: true,
					ThinkingBudget:  v.ThinkingBudget,
				}
			} else {
				// We need to set it to disable thinking on gemini-2.5-flash.
				//
				// Most models really do not want the struct at all, e.g. gemini-2-5-flash-image-preview. Setting the
				// struct to empty will fail the RPC. :(
				if strings.HasPrefix(model, "gemini-2.5-flash") &&
					!strings.Contains(model, "image") &&
					!strings.Contains(model, "live") &&
					!strings.Contains(model, "tts") {
					c.GenerationConfig.ThinkingConfig = &ThinkingConfig{}
				}
			}
		case *genai.OptionsText:
			u, e := c.initOptionsText(v)
			unsupported = append(unsupported, u...)
			errs = append(errs, e...)
		case *genai.OptionsTools:
			u, e := c.initOptionsTools(v)
			unsupported = append(unsupported, u...)
			errs = append(errs, e...)
		case *genai.OptionsAudio:
			errs = append(errs, fmt.Errorf("todo: implement options type %T", opt))
		case *genai.OptionsImage:
			errs = append(errs, fmt.Errorf("todo: implement options type %T", opt))
		case *genai.OptionsVideo:
			errs = append(errs, fmt.Errorf("todo: implement options type %T", opt))
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opt))
		}
	}

	c.Contents = make([]Content, len(msgs))
	for i := range msgs {
		if err := c.Contents[i].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	// There's no field to set, the URL is different.
}

func (c *ChatRequest) initOptionsText(v *genai.OptionsText) ([]string, []error) {
	var unsupported []string
	var errs []error
	c.GenerationConfig.MaxOutputTokens = v.MaxTokens
	c.GenerationConfig.Temperature = v.Temperature
	c.GenerationConfig.TopP = v.TopP
	// For large ones, we could use cached storage.
	if v.SystemPrompt != "" {
		c.SystemInstruction.Parts = []Part{{Text: v.SystemPrompt}}
	}
	c.GenerationConfig.Seed = v.Seed
	if v.TopLogprobs > 0 {
		// TODO: It is unsupported when streaming, but we don't know here if streaming is enabled.
		c.GenerationConfig.Logprobs = v.TopLogprobs
		c.GenerationConfig.ResponseLogprobs = true
	}
	c.GenerationConfig.TopK = v.TopK
	c.GenerationConfig.StopSequences = v.Stop
	if v.DecodeAs != nil {
		c.GenerationConfig.ResponseMimeType = "application/json"
		if err := c.GenerationConfig.ResponseSchema.FromGoObj(v.DecodeAs); err != nil {
			errs = append(errs, fmt.Errorf("decodeAs: %w", err))
		}
	} else if v.ReplyAsJSON {
		c.GenerationConfig.ResponseMimeType = "application/json"
	}
	return unsupported, errs
}

func (c *ChatRequest) initOptionsTools(v *genai.OptionsTools) ([]string, []error) {
	var unsupported []string
	var errs []error
	if len(v.Tools) != 0 {
		switch v.Force {
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
				errs = append(errs, fmt.Errorf("%s: ToolDef.InputSchemaOverride is not yet implemented", t.Name))
			} else {
				if err := params.FromGoType(reflect.TypeOf(t.Callback).In(1).Elem(), reflect.StructTag(""), ""); err != nil {
					errs = append(errs, fmt.Errorf("%s: tool parameters: %w", t.Name, err))
				}
			}
			// See FunctionResponse.To().
			c.Tools[i].FunctionDeclarations = []FunctionDeclaration{{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			}}
			if err := c.Tools[i].FunctionDeclarations[0].Response.FromGoObj(functionResponse); err != nil {
				errs = append(errs, fmt.Errorf("%s: tool response: %w", t.Name, err))
			}
		}
	}
	if v.WebSearch {
		// https://ai.google.dev/gemini-api/docs/google-search
		c.Tools = append(c.Tools, Tool{GoogleSearch: &GoogleSearch{}})
	}
	return unsupported, errs
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
	switch r := in.Role(); r {
	case "user", "computer":
		c.Role = "user"
	case "assistant":
		c.Role = "model"
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	c.Parts = make([]Part, len(in.Requests)+len(in.Replies)+len(in.ToolCallResults))
	for i := range in.Requests {
		if err := c.Parts[i].FromRequest(&in.Requests[i]); err != nil {
			return fmt.Errorf("request #%d: %w", i, err)
		}
	}
	offset := len(in.Requests)
	for i := range in.Replies {
		if err := c.Parts[i].FromReply(&in.Replies[i]); err != nil {
			return fmt.Errorf("reply #%d: %w", i, err)
		}
	}
	offset += len(in.Replies)
	for i := range in.ToolCallResults {
		c.Parts[offset+i].FunctionResponse.From(&in.ToolCallResults[i])
	}
	return nil
}

func (c *Content) To(out *genai.Message) error {
	for _, part := range c.Parts {
		if part.Thought {
			out.Replies = append(out.Replies, genai.Reply{Reasoning: part.Text})
			continue
		}
		// There's no signal as to what it is, we have to test its content.
		// We need to split out content from tools.
		if part.Text != "" {
			out.Replies = append(out.Replies, genai.Reply{Text: part.Text})
			continue
		}
		if part.InlineData.MimeType != "" {
			exts, err := mime.ExtensionsByType(part.InlineData.MimeType)
			if err != nil {
				return &internal.BadError{Err: fmt.Errorf("failed to get extension for mime type %q: %w", part.InlineData.MimeType, err)}
			}
			if len(exts) == 0 {
				return fmt.Errorf("mime type %q has no extension", part.InlineData.MimeType)
			}
			out.Replies = append(out.Replies, genai.Reply{
				Doc: genai.Doc{Filename: "content" + exts[0], Src: &bb.BytesBuffer{D: part.InlineData.Data}},
			})
			continue
		}
		if part.FileData.MimeType != "" {
			exts, err := mime.ExtensionsByType(part.InlineData.MimeType)
			if err != nil {
				return &internal.BadError{Err: fmt.Errorf("failed to get extension for mime type %q: %w", part.InlineData.MimeType, err)}
			}
			if len(exts) == 0 {
				return &internal.BadError{Err: fmt.Errorf("mime type %q has no extension", part.InlineData.MimeType)}
			}
			out.Replies = append(out.Replies, genai.Reply{Doc: genai.Doc{Filename: "content" + exts[0], URL: part.FileData.FileURI}})
			continue
		}
		if part.FunctionCall.Name != "" {
			r := genai.Reply{}
			if len(part.ThoughtSignature) != 0 {
				r.ToolCall.Opaque = map[string]any{"signature": part.ThoughtSignature}
			}
			if err := part.FunctionCall.To(&r.ToolCall); err != nil {
				return err
			}
			out.Replies = append(out.Replies, r)
			continue
		}
		if reflect.ValueOf(part).IsZero() {
			continue
		}
		return &internal.BadError{Err: fmt.Errorf("implement support for part %#v", part)}
	}
	return nil
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
	ExecutableCode      ExecutableCode      `json:"executableCode,omitzero"`      // TODO
	CodeExecutionResult CodeExecutionResult `json:"codeExecutionResult,omitzero"` // TODO

	// Union:
	VideoMetadata VideoMetadata `json:"videoMetadata,omitzero"`
}

func (p *Part) FromRequest(in *genai.Request) error {
	if in.Text != "" {
		p.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType := ""
		var data []byte
		if in.Doc.URL == "" {
			// If more than 20MB, we need to use
			// https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#large-pdfs-urls
			// cacheName, err := c.cacheContent(ctx, context, mime, sp)
			// When using cached content, system instruction, tools or tool_config cannot be used. Weird.
			// in.CachedContent = cacheName
			var err error
			if mimeType, data, err = in.Doc.Read(10 * 1024 * 1024); err != nil {
				return err
			}
			// Gemini refuses text documents as attachment. WTF.
			if strings.HasPrefix(mimeType, "text/") {
				if in.Doc.URL != "" {
					return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
				}
				p.Text = string(data)
			} else {
				p.InlineData.MimeType = mimeType
				p.InlineData.Data = data
			}
		} else {
			if mimeType = base.MimeByExt(path.Ext(in.Doc.URL)); mimeType == "" {
				return fmt.Errorf("could not determine mime type for URL %q", in.Doc.URL)
			}
			p.FileData.MimeType = mimeType
			p.FileData.FileURI = in.Doc.URL
		}
		return nil
	}
	return errors.New("unknown Request type")
}

func (p *Part) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
	}
	if in.Reasoning != "" {
		p.Thought = true
		p.Text = in.Reasoning
		return nil
	}
	if in.Text != "" {
		p.Text = in.Text
		return nil
	}
	if !in.ToolCall.IsZero() {
		if err := p.FunctionCall.From(&in.ToolCall); err != nil {
			return err
		}
		o := in.ToolCall.Opaque
		if b, ok := o["signature"].([]byte); ok {
			p.ThoughtSignature = b
		}
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType := ""
		var data []byte
		if in.Doc.URL == "" {
			// If more than 20MB, we need to use
			// https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#large-pdfs-urls
			// cacheName, err := c.cacheContent(ctx, context, mime, sp)
			// When using cached content, system instruction, tools or tool_config cannot be used. Weird.
			// in.CachedContent = cacheName
			var err error
			if mimeType, data, err = in.Doc.Read(10 * 1024 * 1024); err != nil {
				return err
			}
			// Gemini refuses text documents as attachment. WTF.
			if strings.HasPrefix(mimeType, "text/") {
				if in.Doc.URL != "" {
					return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
				}
				p.Text = string(data)
			} else {
				p.InlineData.MimeType = mimeType
				p.InlineData.Data = data
			}
		} else {
			if mimeType = base.MimeByExt(path.Ext(in.Doc.URL)); mimeType == "" {
				return fmt.Errorf("could not determine mime type for URL %q", in.Doc.URL)
			}
			p.FileData.MimeType = mimeType
			p.FileData.FileURI = in.Doc.URL
		}
		return nil
	}
	return &internal.BadError{Err: errors.New("unknown Reply type")}
}

// FunctionCall is documented at https://ai.google.dev/api/caching?hl=en#FunctionCall
type FunctionCall struct {
	ID               string      `json:"id,omitzero"`
	Name             string      `json:"name,omitzero"`
	Args             StructValue `json:"args,omitzero"`
	ThoughtSignature []byte      `json:"thoughtSignature,omitzero"` // Returned by some models with thinking
}

func (f *FunctionCall) From(in *genai.ToolCall) error {
	f.ID = in.ID
	f.Name = in.Name
	if err := json.Unmarshal([]byte(in.Arguments), &f.Args); err != nil {
		return &internal.BadError{Err: fmt.Errorf("failed to unmarshal arguments: %w", err)}
	}
	return nil
}

func (f *FunctionCall) To(out *genai.ToolCall) error {
	out.ID = f.ID
	out.Name = f.Name
	raw, err := json.Marshal(f.Args)
	if err != nil {
		return &internal.BadError{Err: fmt.Errorf("failed to marshal arguments: %w", err)}
	}
	out.Arguments = string(raw)
	return nil
}

// FunctionResponse is documented at https://ai.google.dev/api/caching?hl=en#FunctionResponse
type FunctionResponse struct {
	ID       string      `json:"id,omitzero"`
	Name     string      `json:"name,omitzero"`
	Response StructValue `json:"response,omitzero"`
}

func (f *FunctionResponse) From(in *genai.ToolCallResult) {
	f.ID = in.ID
	f.Name = in.Name
	// Must match functionResponse
	f.Response = StructValue{"response": in.Result}
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

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Usage: genai.Usage{
			InputTokens:       c.UsageMetadata.PromptTokenCount,
			InputCachedTokens: c.UsageMetadata.CachedContentTokenCount,
			ReasoningTokens:   c.UsageMetadata.ThoughtsTokenCount,
			OutputTokens:      c.UsageMetadata.CandidatesTokenCount + c.UsageMetadata.ToolUsePromptTokenCount + c.UsageMetadata.ThoughtsTokenCount,
			TotalTokens:       c.UsageMetadata.TotalTokenCount,
		},
	}
	if len(c.Candidates) != 1 {
		return out, fmt.Errorf("unexpected number of candidates; expected 1, got %d", len(c.Candidates))
	}
	// Gemini is the only one returning uppercase so convert down for compatibility.
	out.Usage.FinishReason = c.Candidates[0].FinishReason.ToFinishReason()
	err := c.Candidates[0].To(&out.Message)
	if out.Usage.FinishReason == genai.FinishedStop && slices.ContainsFunc(out.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }) {
		// Lie for the benefit of everyone.
		out.Usage.FinishReason = genai.FinishedToolCalls
	}
	// It'd be nice to have citation support, but it's only filled with
	// https://ai.google.dev/api/semantic-retrieval/question-answering and the likes and as of June 2025, it
	// only works in English (!)

	out.Logprobs = c.Candidates[0].LogprobsResult.To()
	return out, err
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
	GroundingMetadata GroundingMetadata `json:"groundingMetadata"`
	AvgLogprobs       float64           `json:"avgLogprobs"`
	LogprobsResult    LogprobsResult    `json:"logprobsResult"`
	Index             int64             `json:"index"`
}

func (r *ResponseCandidate) To(out *genai.Message) error {
	if err := r.Content.To(out); err != nil {
		return err
	}
	if !internal.BeLenient {
		if len(r.SafetyRatings) > 0 {
			return &internal.BadError{Err: fmt.Errorf("implement safety rating: %v", r.SafetyRatings)}
		}
		if len(r.CitationMetadata.CitationSources) > 0 {
			return &internal.BadError{Err: fmt.Errorf("implement citation metadata: %v", r.CitationMetadata.CitationSources)}
		}
		if len(r.GroundingAttributions) > 0 {
			return &internal.BadError{Err: fmt.Errorf("implement grounding attributions: %v", r.GroundingAttributions)}
		}
	}
	if !r.GroundingMetadata.IsZero() {
		replies, err := r.GroundingMetadata.To()
		if err != nil {
			return err
		}
		out.Replies = append(out.Replies, replies...)
	}
	return nil
}

// GroundingMetadata is documented at https://ai.google.dev/api/generate-content?hl=en#GroundingMetadata
type GroundingMetadata struct {
	// https://ai.google.dev/api/generate-content?hl=en#GroundingChunk
	GroundingChunks []struct {
		// https://ai.google.dev/api/generate-content?hl=en#Web
		Web struct {
			URI   string `json:"uri,omitzero"`
			Title string `json:"title,omitzero"`
		} `json:"web,omitzero"`
	} `json:"groundingChunks,omitzero"`
	// https://ai.google.dev/api/generate-content?hl=en#GroundingSupport
	GroundingSupports []struct {
		GroundingChunkIndices []int64   `json:"groundingChunkIndices,omitzero"`
		ConfidenceScores      []float64 `json:"confidenceScores,omitzero"`
		// https://ai.google.dev/api/generate-content?hl=en#Segment
		Segment struct {
			PartIndex  int64  `json:"partIndex,omitzero"`
			StartIndex int64  `json:"startIndex,omitzero"`
			EndIndex   int64  `json:"endIndex,omitzero"`
			Text       string `json:"text,omitzero"`
		} `json:"segment,omitzero"`
	} `json:"groundingSupports,omitzero"`
	WebSearchQueries []string `json:"webSearchQueries,omitzero"`
	// https://ai.google.dev/api/generate-content?hl=en#SearchEntryPoint
	SearchEntryPoint struct {
		RenderedContent string `json:"renderedContent,omitzero"`
		SDKBlob         []byte `json:"sdkBlob,omitzero"` // JSON encoded list of (search term,search url) results
	} `json:"searchEntryPoint,omitzero"`
	// https://ai.google.dev/api/generate-content?hl=en#RetrievalMetadata
	RetrievalMetadata struct {
		GoogleSearchDynamicRetrievalScore float64 `json:"googleSearchDynamicRetrievalScore,omitzero"`
	} `json:"retrievalMetadata,omitzero"`
}

func (g *GroundingMetadata) IsZero() bool {
	return len(g.GroundingChunks) == 0 && len(g.GroundingSupports) == 0 && len(g.WebSearchQueries) == 0 && len(g.SearchEntryPoint.SDKBlob) == 0 && g.RetrievalMetadata.GoogleSearchDynamicRetrievalScore == 0
}

func (g *GroundingMetadata) To() ([]genai.Reply, error) {
	var out []genai.Reply
	var src []genai.CitationSource
	for _, q := range g.WebSearchQueries {
		src = append(src, genai.CitationSource{Type: genai.CitationWebQuery, Snippet: q})
	}
	for _, s := range g.GroundingSupports {
		c := genai.Citation{
			StartIndex: s.Segment.StartIndex,
			EndIndex:   s.Segment.EndIndex,
		}
		c.Sources = append(c.Sources, src...)
		// This will cause duplicate source.
		for _, idx := range s.GroundingChunkIndices {
			if idx < 0 || idx > int64(len(g.GroundingChunks)) {
				return out, &internal.BadError{Err: fmt.Errorf("invalid grounding chunk index: %v", idx)}
			}
			// TODO: The URL points to https://vertexaisearch.cloud.google.com/grounding-api-redirect/... which is
			// not good. We should to a HEAD request to get the actual URL.
			gc := g.GroundingChunks[idx]
			c.Sources = append(c.Sources, genai.CitationSource{
				Type:  genai.CitationWeb,
				URL:   gc.Web.URI,
				Title: gc.Web.Title,
			})
		}
		out = append(out, genai.Reply{Citation: c})
	}
	// Sometimes there's no GroundingSupports. Make sure to still flush the web query.
	if len(src) > 0 && len(g.GroundingSupports) == 0 {
		out = append(out, genai.Reply{Citation: genai.Citation{Sources: src}})
	}
	// SearchEntryPoint will contain some HTML.
	return out, nil
}

// LogprobsResult is documented at https://ai.google.dev/api/generate-content#LogprobsResult
type LogprobsResult struct {
	TopCandidates    []TopCandidate   `json:"topCandidates"`
	ChosenCandidates []TokenCandidate `json:"chosenCandidates"`
}

func (l *LogprobsResult) To() [][]genai.Logprob {
	var out [][]genai.Logprob
	for i, chosen := range l.ChosenCandidates {
		lp := make([]genai.Logprob, 1, len(l.TopCandidates[i].Candidates)+1)
		lp[0] = genai.Logprob{ID: chosen.TokenID, Text: chosen.Token, Logprob: chosen.LogProbability}
		for _, tc := range l.TopCandidates[i].Candidates {
			lp = append(lp, genai.Logprob{ID: tc.TokenID, Text: tc.Token, Logprob: tc.LogProbability})
		}
		out = append(out, lp)
	}
	return out
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
	case FinishRecitation, FinishLanguage, FinishOther, FinishMalformed:
		fallthrough
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(strings.ToLower(string(f)))
	}
}

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
}

// ModalityTokenCount is documented at
// https://ai.google.dev/api/generate-content?hl=en#v1beta.ModalityTokenCount
type ModalityTokenCount struct {
	Modality   Modality `json:"modality"`
	TokenCount int64    `json:"tokenCount"`
}

type ChatStreamChunkResponse struct {
	Candidates []struct {
		Content           Content           `json:"content"`
		FinishReason      FinishReason      `json:"finishReason"`
		Index             int64             `json:"index"`
		GroundingMetadata GroundingMetadata `json:"groundingMetadata"`
	} `json:"candidates"`
	UsageMetadata UsageMetadata `json:"usageMetadata"`
	ModelVersion  string        `json:"modelVersion"`
	ResponseID    string        `json:"responseId"`
}

// Image

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

func (i *ImageRequest) Init(msg genai.Message, model string, mod genai.Modalities, opts ...genai.Options) error {
	if err := msg.Validate(); err != nil {
		return err
	}
	var mime string
	var img []byte
	for i := range msg.Requests {
		if msg.Requests[i].Text != "" {
			continue
		}
		if msg.Requests[i].Doc.IsZero() {
			return errors.New("only text or image can be passed as input")
		}
		if len(img) != 0 {
			return errors.New("only one image can be passed as input")
		}
		var err error
		if mime, img, err = msg.Requests[i].Doc.Read(10 * 1024 * 1024); err != nil {
			return err
		}
	}
	if slices.Contains(mod, genai.ModalityImage) {
		i.Parameters = ImageParameters{
			IncludeSafetyAttributes: true,
			IncludeRAIReason:        true,
			OutputOptions:           ImageOutput{MimeType: "image/jpeg"},
		}
	}
	i.Instances = []ImageInstance{{Prompt: msg.String()}}
	if len(img) != 0 {
		i.Instances[0].Image.BytesBase64Encoded = img
		i.Instances[0].Image.MimeType = mime
	}
	// This is important otherwise it can return 2 images.
	// TODO: Expose it in OptionsImage.
	i.Parameters.SampleCount = 1
	// The acceptable value depends on the country the paying user account is associated with.
	i.Parameters.PersonGeneration = "allow_adult"
	// Seems like it's not supported?
	// i.Parameters.EnhancePrompt = true
	var uce error
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *Options:
		case *genai.OptionsImage:
			if v.Seed != 0 {
				// Seed is only supported with Vertex AI API.
				uce = &base.ErrNotSupported{Options: []string{"OptionsText.Seed"}}
			}
			// TODO: Width and Height
		case *genai.OptionsVideo:
			if v.Duration != 0 {
				i.Parameters.DurationSeconds = int64(v.Duration.Round(time.Second) / time.Second)
			}
		default:
			return fmt.Errorf("unsupported options type %T", opt)
		}
	}
	return uce
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

type ImageOutput struct {
	MimeType           string  `json:"mimeType,omitzero"` // "image/jpeg"
	CompressionQuality float64 `json:"compressionQuality,omitzero"`
}

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

// Operation is documented at https://ai.google.dev/api/batch-mode#Operation
//
// See generateVideosResponseFromMldev in js-genai for more details.
type Operation struct {
	Name     string         `json:"name"`
	Metadata map[string]any `json:"metadata"`
	Done     bool           `json:"done"`
	// One of the following:
	Error    Status `json:"error"`
	Response struct {
		Type                  string `json:"@type"` // "type.googleapis.com/google.ai.generativelanguage.v1beta.PredictLongRunningResponse"
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
	} `json:"response"`
}

// Status is documented at https://ai.google.dev/api/files#v1beta.Status
type Status struct {
	Code    int64          `json:"code"`
	Message string         `json:"message"`
	Details map[string]any `json:"details"`
}

// Caching

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

func (c *CachedContent) Init(msgs genai.Messages, model, name, displayName string, ttl time.Duration, opts ...genai.Options) error {
	if err := msgs.Validate(); err != nil {
		return err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		if o, ok := opt.(*genai.OptionsText); ok && o.SystemPrompt != "" {
			c.SystemInstruction.Parts = []Part{{Text: o.SystemPrompt}}
		}
	}
	// For large files, use https://ai.google.dev/gemini-api/docs/caching?hl=en&lang=rest#pdfs_1
	c.Contents = make([]Content, len(msgs))
	for i := range msgs {
		if err := c.Contents[i].From(&msgs[i]); err != nil {
			return fmt.Errorf("message #%d: %w", i, err)
		}
	}
	c.Model = "models/" + model
	c.DisplayName = displayName
	if name != "" {
		c.Name = "cachedContents/" + name
	}
	if ttl > 0 {
		c.TTL = Duration(ttl)
	}
	// TODO: ToolConfig
	return nil
}

func (c *CachedContent) GetID() string {
	return c.Name
}

func (c *CachedContent) GetDisplayName() string {
	return c.DisplayName
}

func (c *CachedContent) GetExpiry() time.Time {
	return c.ExpireTime
}

// Expiration must be embedded. Only one of the fields can be set.
type Expiration struct {
	ExpireTime time.Time `json:"expireTime,omitzero"` // ISO 8601
	TTL        Duration  `json:"ttl,omitzero"`        // Duration; input only
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

func (m *Model) Context() int64 {
	return m.InputTokenLimit
}

// ModelsResponse represents the response structure for Gemini models listing
type ModelsResponse struct {
	Models        []Model `json:"models"`
	NextPageToken string  `json:"nextPageToken"`
}

// ToModels converts Gemini models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Models))
	for i := range r.Models {
		models[i] = &r.Models[i]
	}
	return models
}

//

// ErrorResponse represents the response structure for Gemini errors.
//
// It is returned as an error.
type ErrorResponse struct {
	ErrorVal ErrorResponseError `json:"error"`
}

func (e *ErrorResponse) Error() string {
	return fmt.Sprintf("%s (%d): %s", e.ErrorVal.Status, e.ErrorVal.Code, strings.TrimSpace(e.ErrorVal.Message))
}

func (e *ErrorResponse) IsAPIError() bool {
	return true
}

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

// Client implements genai.Provider.
type Client struct {
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to Google's Gemini platform API.
//
// If opts.APIKey is not provided, it tries to load it from the GEMINI_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://ai.google.dev/gemini-api/docs/getting-started
//
// To use multiple models, create multiple clients.
// Use one of the model from https://ai.google.dev/gemini-api/docs/models/gemini
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
//
// See https://ai.google.dev/gemini-api/docs/file-prompting-strategies?hl=en
// for good ideas on how to prompt with images.
//
// Using large files requires a pinned model with caching support.
//
// Visit https://ai.google.dev/gemini-api/docs/pricing for up to date information.
//
// As of May 2025, price on Pro model increases when more than 200k input tokens are used.
// Cached input tokens are 25% of the price of new tokens.
func New(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if err := opts.Validate(); err != nil {
		return nil, err
	}
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://aistudio.google.com/apikey"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("GEMINI_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "GEMINI_API_KEY", URL: apiKeyURL}
		}
	}
	switch len(opts.OutputModalities) {
	case 0:
		// Auto-detect below.
	case 1:
		switch opts.OutputModalities[0] {
		case genai.ModalityAudio, genai.ModalityImage, genai.ModalityText, genai.ModalityVideo:
		case genai.ModalityDocument:
			fallthrough
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image, text, or video are supported", opts.OutputModalities)
		}
	case 2:
		// The only combination supported is image + text.
		mods := slices.Clone(opts.OutputModalities)
		slices.Sort(mods)
		if !slices.Equal(mods, []genai.Modality{genai.ModalityImage, genai.ModalityText}) {
			return nil, fmt.Errorf("unexpected option Modalities %s, only image+text are supported when grouped together", mods)
		}
	default:
		return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image, text, video, or image+text are supported", opts.OutputModalities)
	}
	// Google supports HTTP POST gzip compression!
	var t http.RoundTripper = &roundtrippers.PostCompressed{
		Transport: base.DefaultTransport,
		Encoding:  "gzip",
	}
	if wrapper != nil {
		t = wrapper(t)
	}
	// Eventually, use OAuth https://ai.google.dev/gemini-api/docs/oauth#curl
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			ProcessStream:   ProcessStream,
			PreloadedModels: opts.PreloadedModels,
			LieToolCalls:    true,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"x-goog-api-key": {apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch opts.Model {
		case genai.ModelNone:
		case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA, "":
			var mod genai.Modality
			switch len(opts.OutputModalities) {
			case 0:
				mod = genai.ModalityText
			case 1:
				mod = opts.OutputModalities[0]
			default:
				// TODO: Maybe it's possible, need to double check.
				return nil, fmt.Errorf("can't use model %s with option Modalities %s", opts.Model, opts.OutputModalities)
			}
			switch mod {
			case genai.ModalityText:
				if c.impl.Model, err = c.selectBestTextModel(ctx, opts.Model); err != nil {
					return nil, err
				}
				c.impl.GenSyncURL = "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":generateContent"
				c.impl.GenStreamURL = "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":streamGenerateContent?alt=sse"
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityImage:
				if c.impl.Model, err = c.selectBestImageModel(ctx, opts.Model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityAudio:
				fallthrough
			case genai.ModalityDocument:
				fallthrough
			case genai.ModalityVideo:
				fallthrough
			default:
				// TODO: Soon, because it's cool.
				return nil, fmt.Errorf("automatic model selection is not implemented yet for modality %s (send PR to add support)", opts.OutputModalities)
			}
		default:
			c.impl.Model = opts.Model
			if len(opts.OutputModalities) == 0 {
				c.impl.OutputModalities, err = c.detectModelModalities(ctx, c.impl.Model)
			} else {
				c.impl.OutputModalities = opts.OutputModalities
			}
			c.impl.GenSyncURL = "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":generateContent"
			c.impl.GenStreamURL = "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":streamGenerateContent?alt=sse"
		}
	}
	return c, err
}

// detectModelModalities tries its best to figure out the modality of a model
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) detectModelModalities(ctx context.Context, model string) (genai.Modalities, error) {
	// It's tricky because modalities are not directly returned by ListModels.
	if strings.HasPrefix(model, "gem") {
		return genai.Modalities{genai.ModalityText}, nil
	} else if strings.HasPrefix(model, "ima") {
		return genai.Modalities{genai.ModalityImage}, nil
	} else if strings.HasPrefix(model, "veo") {
		return genai.Modalities{genai.ModalityVideo}, nil
	}
	// We probably want to fetch SupportedGenerationMethods from the model anyway to make sure the right
	// API is used. We can keep a predefined table for known model, I'd be surprised the generation
	// methods would change per model.
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to automatically detect the model modality: %w", err)
	}
	for _, mdl := range mdls {
		if m := mdl.(*Model); m.GetID() == model {
			if slices.Contains(m.SupportedGenerationMethods, "generateContent") {
				return genai.Modalities{genai.ModalityText}, nil
			} else if slices.Contains(m.SupportedGenerationMethods, "predict") {
				return genai.Modalities{genai.ModalityImage}, nil
			} else if slices.Contains(m.SupportedGenerationMethods, "predictLongRunning") {
				return genai.Modalities{genai.ModalityVideo}, nil
			}
			return nil, fmt.Errorf("failed to automatically detect the model modality with methods: %s", m.SupportedGenerationMethods)
		}
	}
	return nil, fmt.Errorf("failed to automatically detect the model modality: model %s not found", model)
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}

	// Build a map of available models for quick lookup.
	availableModels := map[string]struct{}{}
	for _, mdl := range mdls {
		availableModels[strings.TrimPrefix(mdl.(*Model).Name, "models/")] = struct{}{}
	}
	cheap := preference == genai.ModelCheap
	good := preference == genai.ModelGood || preference == ""
	if cheap {
		if _, ok := availableModels["gemini-flash-lite-latest"]; ok {
			return "gemini-flash-lite-latest", nil
		}
	} else if good {
		if _, ok := availableModels["gemini-flash-latest"]; ok {
			return "gemini-flash-latest", nil
		}
	} else {
		if _, ok := availableModels["gemini-pro-latest"]; ok {
			return "gemini-pro-latest", nil
		}
	}

	// If the preferred "-latest" model is not available, find the best from the list.
	selectedModel := ""
	var tokens int64
	selectedPinned := false
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Skip experimental models
		if !slices.Contains(m.SupportedGenerationMethods, "generateContent") || strings.Contains(m.Name, "tts") || (tokens != 0 && tokens > m.OutputTokenLimit) || strings.Contains(m.Name, "-exp") {
			continue
		}
		name := strings.TrimPrefix(m.Name, "models/")
		matches := false
		if cheap {
			matches = strings.HasPrefix(name, "gemini") && strings.Contains(name, "-flash-lite")
		} else if good {
			matches = strings.HasPrefix(name, "gemini") && strings.Contains(name, "-flash") && !strings.Contains(name, "-lite")
		} else {
			matches = strings.HasPrefix(name, "gemini") && strings.Contains(name, "-pro")
		}
		if !matches {
			continue
		}
		// Check if this model is better than the current selection.
		isCandidatePinned := isPinnedModel(name)
		shouldUpdate := false
		if selectedModel == "" {
			shouldUpdate = true
		} else if isCandidatePinned && !selectedPinned {
			// Prefer pinned models (with -NNN suffix) over unpinned ones
			shouldUpdate = true
		} else if isCandidatePinned == selectedPinned {
			// Both pinned or both unpinned; compare versions
			if compareVersions(name, selectedModel) > 0 {
				shouldUpdate = true
			}
		}
		if shouldUpdate {
			selectedModel = name
			selectedPinned = isCandidatePinned
			tokens = m.OutputTokenLimit
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// isPinnedModel returns true if the model name ends with a version number (e.g., -001, -002).
func isPinnedModel(name string) bool {
	parts := strings.Split(name, "-")
	if len(parts) == 0 {
		return false
	}
	lastPart := parts[len(parts)-1]
	// A pinned model ends with 3 digits and is not "-latest"
	if lastPart == "latest" {
		return false
	}
	return len(lastPart) == 3 && lastPart[0] >= '0' && lastPart[0] <= '9'
}

// compareVersions compares two model names by their version numbers.
// Returns >0 if a > b, <0 if a < b, 0 if equal or incomparable.
// Example: "gemini-3-flash" > "gemini-2.5-flash"
func compareVersions(a, b string) int {
	// Extract version parts (e.g., "2.5" from "gemini-2.5-flash")
	getVersionParts := func(name string) (major, minor int) {
		parts := strings.Split(name, "-")
		for i, part := range parts {
			if part == "gemini" && i+1 < len(parts) {
				versionStr := parts[i+1]
				// Check if the next meaningful part is a type (flash, pro, etc)
				var isVersionNumber bool
				for j := i + 2; j < len(parts); j++ {
					if parts[j] == "flash" || parts[j] == "pro" || parts[j] == "lite" {
						isVersionNumber = true
						break
					}
				}
				if isVersionNumber {
					// Parse major.minor version
					versionParts := strings.Split(versionStr, ".")
					if len(versionParts) >= 1 {
						if v, err := strconv.Atoi(versionParts[0]); err == nil {
							major = v
						}
					}
					if len(versionParts) >= 2 {
						if v, err := strconv.Atoi(versionParts[1]); err == nil {
							minor = v
						}
					}
					return
				}
			}
		}
		return
	}

	aMajor, aMinor := getVersionParts(a)
	bMajor, bMinor := getVersionParts(b)
	if aMajor != bMajor {
		return aMajor - bMajor
	}
	return aMinor - bMinor
}

// selectBestImageModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestImageModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == genai.ModelCheap
	good := preference == genai.ModelGood || preference == ""
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model)
		if !slices.Contains(m.SupportedGenerationMethods, "predict") || strings.Contains(m.Name, "tts") {
			continue
		}
		// TODO: Do numerical comparison? There are version numbers and tokens limits we can test against.
		if name := strings.TrimPrefix(m.Name, "models/"); selectedModel == "" || name > selectedModel {
			isFast := strings.Contains(name, "fast")
			isUltra := strings.Contains(name, "ultra")
			if cheap {
				if isFast {
					selectedModel = name
				}
			} else if good {
				if !isFast && !isUltra {
					selectedModel = name
				}
			} else {
				if isUltra {
					selectedModel = name
				}
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "gemini"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to fetch results (e.g. videos) generated by the provider.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Result, error) {
	if !slices.Contains(c.impl.OutputModalities, genai.ModalityText) {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.genDoc(ctx, msgs[0], opts...)
	}
	// GenSync must be inlined because we need to call our GenSyncRaw.
	res := genai.Result{}
	in := &ChatRequest{}
	if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
		return res, err
	}
	out := &ChatResponse{}
	if err := c.GenSyncRaw(ctx, in, out); err != nil {
		return res, err
	}
	res, err := out.ToResult()
	if err != nil {
		return res, err
	}
	if err := res.Validate(); err != nil {
		// Catch provider implementation bugs.
		return res, err
	}

	lastResp := c.impl.LastResponseHeaders()
	if c.impl.ProcessHeaders != nil && lastResp != nil {
		res.Usage.Limits = c.impl.ProcessHeaders(lastResp)
	}
	return res, nil
}

// GenSyncRaw provides access to the raw API.
//
// It calls the Gemini API method generateContent.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	if len(in.GenerationConfig.ResponseModalities) == 0 {
		in.GenerationConfig.ResponseModalities = make([]Modality, len(c.impl.OutputModalities))
		for i, m := range c.impl.OutputModalities {
			switch m {
			case genai.ModalityAudio:
				in.GenerationConfig.ResponseModalities[i] = ModalityAudio
			case genai.ModalityImage:
				in.GenerationConfig.ResponseModalities[i] = ModalityImage
			case genai.ModalityText:
				in.GenerationConfig.ResponseModalities[i] = ModalityText
			case genai.ModalityDocument, genai.ModalityVideo:
				fallthrough
			default:
				return fmt.Errorf("unsupported modality %s", m)
			}
		}
	}
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if !slices.Contains(c.impl.OutputModalities, genai.ModalityText) {
		return base.SimulateStream(ctx, c, msgs, opts...)
	}
	// GenStream must be inlined because we need to call our GenStreamRaw.
	res := genai.Result{}
	var finalErr error

	fnFragments := func(yield func(genai.Reply) bool) {
		in := &ChatRequest{}
		if err := in.Init(msgs, c.impl.Model, opts...); err != nil {
			finalErr = &internal.BadError{Err: err}
			return
		}
		// Generate parsed chunks from the raw JSON SSE stream.
		chunks, finish1 := c.GenStreamRaw(ctx, in)
		// Converts raw chunks into fragments.
		fragments, finish2 := c.impl.ProcessStream(chunks)
		for f := range fragments {
			if f.IsZero() {
				continue
			}
			if err := f.Validate(); err != nil {
				// Catch provider implementation bugs.
				finalErr = &internal.BadError{Err: err}
				break
			}
			if err := res.Accumulate(f); err != nil {
				finalErr = &internal.BadError{Err: err}
				break
			}
			if !yield(f) {
				break
			}
		}
		if err := finish1(); finalErr == nil {
			finalErr = err
		}
		var err error
		res.Usage, res.Logprobs, err = finish2()
		if finalErr == nil {
			finalErr = err
		}
		lastResp := c.impl.LastResponseHeaders()
		if c.impl.ProcessHeaders != nil && lastResp != nil {
			res.Usage.Limits = c.impl.ProcessHeaders(lastResp)
		}
		if c.impl.LieToolCalls && res.Usage.FinishReason == genai.FinishedStop {
			for i := range res.Replies {
				if !res.Replies[i].ToolCall.IsZero() {
					// Lie for the benefit of everyone.
					res.Usage.FinishReason = genai.FinishedToolCalls
					break
				}
			}
		}
	}
	fnFinish := func() (genai.Result, error) {
		if finalErr != nil {
			return res, finalErr
		}
		if err := res.Validate(); err != nil {
			// Catch provider implementation bugs.
			return res, err
		}
		return res, nil
	}
	return fnFragments, fnFinish
}

// GenStreamRaw provides access to the raw API.
//
// It calls the Gemini API method streamGenerateContent?alt=sse.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	if len(in.GenerationConfig.ResponseModalities) == 0 {
		in.GenerationConfig.ResponseModalities = make([]Modality, len(c.impl.OutputModalities))
		for i, m := range c.impl.OutputModalities {
			switch m {
			case genai.ModalityAudio:
				in.GenerationConfig.ResponseModalities[i] = ModalityAudio
			case genai.ModalityImage:
				in.GenerationConfig.ResponseModalities[i] = ModalityImage
			case genai.ModalityText:
				in.GenerationConfig.ResponseModalities[i] = ModalityText
			case genai.ModalityDocument, genai.ModalityVideo:
				fallthrough
			default:
				return yieldNothing[ChatStreamChunkResponse], func() error {
					return &internal.BadError{Err: fmt.Errorf("unsupported modality %s", m)}
				}
			}
		}
	}
	return c.impl.GenStreamRaw(ctx, in)
}

// CacheAddRequest caches the content for later use.
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
func (c *Client) CacheAddRequest(ctx context.Context, msgs genai.Messages, name, displayName string, ttl time.Duration, opts ...genai.Options) (string, error) {
	// See https://ai.google.dev/gemini-api/docs/caching?hl=en&lang=rest#considerations
	// Useful when reusing the same large data multiple times to reduce token usage.
	// This requires a pinned model, with trailing -001.
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	in := CachedContent{}
	if err := in.Init(msgs, c.impl.Model, name, displayName, ttl, opts...); err != nil {
		return "", err
	}
	out := CachedContent{}
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents"
	if err := c.impl.DoRequest(ctx, "POST", url, &in, &out); err != nil {
		return "", err
	}
	name = strings.TrimPrefix(out.Name, "cachedContents/")
	// slog.InfoContext(ctx, "gemini", "cached", name, "tokens", out.UsageMetadata.TotalTokenCount)
	return name, nil
}

func (c *Client) CacheExtend(ctx context.Context, name string, ttl time.Duration) error {
	// https://ai.google.dev/api/caching#method:-cachedcontents.patch
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + url.PathEscape(name)
	// Model is required.
	in := CachedContent{Model: "models/" + c.impl.Model, Expiration: Expiration{TTL: Duration(ttl)}}
	out := CachedContent{}
	return c.impl.DoRequest(ctx, "PATCH", url, &in, &out)
}

func (c *Client) CacheList(ctx context.Context) ([]genai.CacheEntry, error) {
	l, err := c.CacheListRaw(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]genai.CacheEntry, len(l))
	for i := range l {
		out[i] = &l[i]
	}
	return out, nil
}

// CacheListRaw retrieves the list of cached items.
func (c *Client) CacheListRaw(ctx context.Context) ([]CachedContent, error) {
	// https://ai.google.dev/api/caching#method:-cachedcontents.list
	// pageSize, pageToken
	var data struct {
		CachedContents []CachedContent `json:"cachedContents"`
		NextPageToken  string          `json:"nextPageToken"`
	}
	baseURL := "https://generativelanguage.googleapis.com/v1beta/cachedContents?pageSize=100"
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
		if err := c.impl.DoRequest(ctx, "GET", url, nil, &data); err != nil {
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

func (c *Client) CacheGetRaw(ctx context.Context, name string) (CachedContent, error) {
	// https://ai.google.dev/api/caching#method:-cachedcontents.get
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + url.PathEscape(name)
	out := CachedContent{}
	err := c.impl.DoRequest(ctx, "GET", url, nil, &out)
	return out, err
}

// CacheDelete deletes a cached file.
func (c *Client) CacheDelete(ctx context.Context, name string) error {
	// https://ai.google.dev/api/caching#method:-cachedcontents.delete
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents/" + url.PathEscape(name)
	var out struct{}
	return c.impl.DoRequest(ctx, "DELETE", url, nil, &out)
}

// genDoc is a simplified version of GenSync.
//
// Use it to generate images.
//
// genDoc is only supported for models that have "predict" reported in their Model.SupportedGenerationMethods.
func (c *Client) genDoc(ctx context.Context, msg genai.Message, opts ...genai.Options) (genai.Result, error) {
	// TODO: Smartly decide the method to use instead of hardcoding on the modality.
	if slices.Contains(c.impl.OutputModalities, genai.ModalityVideo) {
		id, err := c.GenAsync(ctx, genai.Messages{msg}, opts...)
		if err != nil {
			return genai.Result{}, err
		}
		// Loop until the image is available.
		waitForPoll := time.Second
		for _, opt := range opts {
			if v, ok := (opt).(*genai.OptionsVideo); ok && v.PollInterval != 0 {
				waitForPoll = v.PollInterval
			}
		}
		for {
			select {
			case <-ctx.Done():
				return genai.Result{}, ctx.Err()
			case <-time.After(waitForPoll):
				if res, err := c.PokeResult(ctx, id); res.Usage.FinishReason != genai.Pending {
					return res, err
				}
			}
		}
	}
	res := genai.Result{}
	req := ImageRequest{}
	if err := req.Init(msg, c.impl.Model, c.impl.OutputModalities, opts...); err != nil {
		return res, err
	}
	resp, err := c.PredictRaw(ctx, req)
	if err != nil {
		return res, err
	}
	// As of now, the last item in the list contains ContentType = "Positive Prompt".
	nbImages := 0
	for i := range resp.Predictions {
		if len(resp.Predictions[i].BytesBase64Encoded) > 0 {
			nbImages++
		}
	}
	for i := range resp.Predictions {
		if len(resp.Predictions[i].BytesBase64Encoded) == 0 {
			continue
		}
		if resp.Predictions[i].MimeType != "image/jpeg" {
			return res, fmt.Errorf("unsupported mime type %q", resp.Predictions[i].MimeType)
		}
		n := "content.jpg"
		if nbImages > 1 {
			n = fmt.Sprintf("content%d.jpg", i+1)
		}
		res.Replies = append(res.Replies, genai.Reply{Doc: genai.Doc{Filename: n, Src: &bb.BytesBuffer{D: resp.Predictions[i].BytesBase64Encoded}}})
	}
	if err = res.Validate(); err != nil {
		return res, err
	}
	return res, nil
}

// GenAsync implements genai.ProviderGenAsync.
//
// It requests the providers' asynchronous API and returns the job ID.
//
// GenAsync is only supported for models that have "predictLongRunning" reported in their
// Model.SupportedGenerationMethods.
//
// The resulting file is available for 48 hours. It requires the API key in the HTTP header to be fetched, so
// use the client's HTTP client.
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.Options) (genai.Job, error) {
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	if len(msgs) != 1 {
		return "", errors.New("only one message can be passed as input")
	}
	req := ImageRequest{}
	if err := req.Init(msgs[0], c.impl.Model, c.impl.OutputModalities, opts...); err != nil {
		return "", err
	}
	resp, err := c.PredictLongRunningRaw(ctx, req)
	return genai.Job(resp.Name), err
}

// PokeResult implements genai.ProviderGenAsync.
//
// It retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	op, err := c.PokeResultRaw(ctx, id)
	if err != nil {
		return res, err
	}
	if !op.Done {
		res.Usage.FinishReason = genai.Pending
		return res, nil
	}
	res.Usage.FinishReason = genai.FinishedStop
	for _, p := range op.Response.GenerateVideoResponse.GeneratedSamples {
		// This requires the Google API key to fetch!
		res.Replies = []genai.Reply{{Doc: genai.Doc{Filename: "content.mp4", URL: p.Video.URI}}}
	}
	return res, nil
}

// PredictRaw requests the providers' synchronous API to generate an image.
//
// The official documentation https://ai.google.dev/api/models?hl=en#method:-models.predict is not really
// helpful.
func (c *Client) PredictRaw(ctx context.Context, req ImageRequest) (ImageResponse, error) {
	res := ImageResponse{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	// https://ai.google.dev/api/models?hl=en#method:-models.predict
	u := "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":predict"
	err := c.impl.DoRequest(ctx, "POST", u, &req, &res)
	return res, err
}

// PredictLongRunningRaw requests the providers' asynchronous API to generate a video.
//
// The official documentation https://ai.google.dev/api/models?hl=en#method:-models.predictlongrunning is not really
// helpful.
func (c *Client) PredictLongRunningRaw(ctx context.Context, req ImageRequest) (Operation, error) {
	res := Operation{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	u := "https://generativelanguage.googleapis.com/v1beta/models/" + url.PathEscape(c.impl.Model) + ":predictLongRunning"
	err := c.impl.DoRequest(ctx, "POST", u, &req, &res)
	return res, err
}

// PokeResultRaw retrieves the result for a job ID if already available.
func (c *Client) PokeResultRaw(ctx context.Context, id genai.Job) (Operation, error) {
	res := Operation{}
	u := "https://generativelanguage.googleapis.com/v1beta/" + string(id)
	err := c.impl.DoRequest(ctx, "GET", u, nil, &res)
	if err != nil {
		return res, fmt.Errorf("failed to get job %q: %w", id, err)
	}
	return res, err
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://ai.google.dev/api/models?hl=en#method:-models.list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// TODO: To implement ProviderGenAsync, we need to use the Vertex API, not the API key based Gemini one.
// https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api
// This may require creating a whole new provider with Vertex AI API surface.

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if len(pkt.Candidates) != 1 {
					continue
				}
				if pkt.UsageMetadata.TotalTokenCount != 0 {
					// Not 100% sure.
					u.InputTokens = pkt.UsageMetadata.PromptTokenCount
					u.ReasoningTokens = pkt.UsageMetadata.ThoughtsTokenCount
					u.OutputTokens = pkt.UsageMetadata.CandidatesTokenCount + pkt.UsageMetadata.ToolUsePromptTokenCount + pkt.UsageMetadata.ThoughtsTokenCount
					u.TotalTokens = pkt.UsageMetadata.TotalTokenCount
				}
				if pkt.Candidates[0].FinishReason != "" {
					u.FinishReason = pkt.Candidates[0].FinishReason.ToFinishReason()
				}
				switch role := pkt.Candidates[0].Content.Role; role {
				case "model", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}

				f := genai.Reply{}

				if !pkt.Candidates[0].GroundingMetadata.IsZero() {
					replies, err := pkt.Candidates[0].GroundingMetadata.To()
					if err != nil {
						finalErr = &internal.BadError{Err: err}
						return
					}
					// Handle citations as a separate packet.
					for _, r := range replies {
						f.Citation = r.Citation
						if !yield(f) {
							return
						}
					}
					f = genai.Reply{}
				}

				for _, part := range pkt.Candidates[0].Content.Parts {
					if part.Thought {
						f.Reasoning += part.Text
					} else {
						f.Text += part.Text
					}
					if part.InlineData.MimeType != "" || len(part.InlineData.Data) != 0 {
						exts, err := mime.ExtensionsByType(part.InlineData.MimeType)
						if err != nil {
							finalErr = &internal.BadError{Err: fmt.Errorf("failed to get extension for mime type %q: %w", part.InlineData.MimeType, err)}
							return
						}
						if len(exts) == 0 {
							finalErr = &internal.BadError{Err: fmt.Errorf("mime type %q has no extension", part.InlineData.MimeType)}
							return
						}
						f.Doc.Filename = "content" + exts[0]
						f.Doc.Src = &bb.BytesBuffer{D: part.InlineData.Data}
					}
					if part.FunctionCall.ID != "" || part.FunctionCall.Name != "" {
						// https://ai.google.dev/api/caching?hl=en#FunctionCall
						if len(part.ThoughtSignature) != 0 {
							f.ToolCall.Opaque = map[string]any{"signature": part.ThoughtSignature}
						}
						if err := part.FunctionCall.To(&f.ToolCall); err != nil {
							finalErr = err
							return
						}
					}
					if part.FunctionResponse.ID != "" {
						// https://ai.google.dev/api/caching?hl=en#FunctionResponse
						finalErr = &internal.BadError{Err: fmt.Errorf("implement function response %#v", part)}
						return
					}
					if part.FileData.MimeType != "" || part.FileData.FileURI != "" {
						finalErr = &internal.BadError{Err: fmt.Errorf("implement file data %#v", part)}
						return
					}
					if part.ExecutableCode.Language != "" || part.ExecutableCode.Code != "" {
						// https://ai.google.dev/api/caching?hl=en#ExecutableCode
						finalErr = &internal.BadError{Err: fmt.Errorf("implement executable code %#v", part)}
						return
					}
					if part.CodeExecutionResult.Outcome != "" || part.CodeExecutionResult.Output != "" {
						// https://ai.google.dev/api/caching?hl=en#CodeExecutionResult
						finalErr = &internal.BadError{Err: fmt.Errorf("implement code execution result %#v", part)}
						return
					}
				}
				if !yield(f) {
					return
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

func yieldNothing[T any](yield func(T) bool) {
}

var (
	_ genai.Provider      = &Client{}
	_ genai.ProviderCache = &Client{}
)
