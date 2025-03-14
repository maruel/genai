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
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
)

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
type StructValue map[string]Value

// https://protobuf.dev/reference/protobuf/google.protobuf/#value
// TODO: Confirm.
type Value struct {
	NullValue   int64       `json:"null_value,omitzero"`
	NumberValue float64     `json:"number_value,omitzero"`
	StringValue string      `json:"string_value,omitzero"`
	BoolValue   bool        `json:"bool_value,omitzero"`
	StructValue StructValue `json:"struct_value,omitzero"`
	ListValue   []Value     `json:"list_value,omitzero"`
}

// https://ai.google.dev/api/caching?hl=en#Part
//
// Part is a union that only has one of the field set.
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

// https://ai.google.dev/api/caching?hl=en#Content
type Content struct {
	Parts []Part `json:"parts"`
	// Must be either 'user' or 'model'.
	Role string `json:"role,omitzero"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.SafetySetting
type SafetySetting struct {
	Category  string `json:"category"`  // https://ai.google.dev/api/generate-content?hl=en#v1beta.HarmCategory
	Threshold int64  `json:"threshold"` // https://ai.google.dev/api/generate-content?hl=en#HarmBlockThreshold
}

// https://ai.google.dev/api/caching?hl=en#Tool
type Tool struct {
	FunctionDeclarations []FunctionDeclaration `json:"nunctionDeclaration,omitzero"`
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
	Type             string            `json:"type,omitzero"`   // TYPE_UNSPECIFIED, STRING, NUMBER, INTEGER, BOOLEAN, ARRAY, OBJECT
	Format           string            `json:"format,omitzero"` // "json-schema"
	Description      string            `json:"description,omitzero"`
	Nullable         bool              `json:"nullable,omitzero"`
	Enum             []string          `json:"enum,omitzero"`
	MaxItems         int64             `json:"maxItems,omitzero"`
	MinItems         int64             `json:"minItems,omitzero"`
	Properties       map[string]Schema `json:"properties,omitzero"`
	Required         []string          `json:"required,omitzero"`
	PropertyOrdering []string          `json:"propertyOrdering,omitzero"`
	Items            *Schema           `json:"items,omitzero"` // When type is "array"
}

// https://ai.google.dev/api/caching?hl=en#ToolConfig
type ToolConfig struct {
	// https://ai.google.dev/api/caching?hl=en#FunctionCallingConfig
	FunctionCallingConfig struct {
		// https://ai.google.dev/api/caching?hl=en#Mode_1
		Mode                 string   `json:"mode,omitzero"` // MODE_UNSPECIFIED, AUTO, ANY, NONE
		AllowedFunctionNames []string `json:"allowedFunctionNames,omitzero"`
	} `json:"functionCallingConfig,omitzero"`
}

// https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
type CompletionRequest struct {
	Contents          []Content       `json:"contents"`
	Tools             []Tool          `json:"tools,omitzero"`
	ToolConfig        ToolConfig      `json:"toolConfig,omitzero"`
	SafetySettings    []SafetySetting `json:"safetySettings,omitzero"`
	SystemInstruction Content         `json:"systemInstruction,omitzero"`
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerationConfig
	GenerationConfig struct {
		StopSequences              []string `json:"stopSequences,omitzero"`
		ResponseMimeType           string   `json:"responseMimeType,omitzero"`
		ResponseSchema             Schema   `json:"responseSchema,omitzero"`
		ResponseModalities         []string `json:"responseModalities,omitzero"`
		CandidateCount             int64    `json:"candidateCount,omitzero"`
		MaxOutputTokens            int64    `json:"maxOutputTokens,omitzero"`
		Temperature                float64  `json:"temperature,omitzero"` // [0, 2]
		TopP                       float64  `json:"topP,omitzero"`
		TopK                       int64    `json:"topK,omitzero"`
		Seed                       int64    `json:"seed,omitzero"`
		PresencePenalty            float64  `json:"presencePenalty,omitzero"`
		FrequencyPenalty           float64  `json:"frequencyPenalty,omitzero"`
		ResponseLogprobs           bool     `json:"responseLogprobs,omitzero"`
		Logprobs                   int64    `json:"logProbs,omitzero"`
		EnableEnhancedCivicAnswers bool     `json:"enableEnhancedCivicAnswers,omitzero"`
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
		MediaResolution string `json:"mediaResolution,omitzero"`
	} `json:"generationConfig,omitzero"`
	CachedContent string `json:"cachedContent,omitzero"`
}

func (c *CompletionRequest) fromOpts(opts any) error {
	if opts != nil {
		// This doesn't seem to be well supported yet:
		//    in.GenerationConfig.ResponseLogprobs = true
		switch v := opts.(type) {
		case *genaiapi.CompletionOptions:
			c.GenerationConfig.MaxOutputTokens = v.MaxTokens
			c.GenerationConfig.Temperature = v.Temperature
			c.GenerationConfig.Seed = v.Seed
			if v.ReplyAsJSON || !v.JSONSchema.IsZero() {
				return errors.New("to be implemented")
			}
			if len(v.Tools) != 0 {
				return errors.New("tools support is not implemented yet")
			}
		default:
			return fmt.Errorf("unsupported options type %T", opts)
		}
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) (string, error) {
	c.Contents = make([]Content, 0, len(msgs))
	sp := ""
	for i, m := range msgs {
		if err := m.Validate(); err != nil {
			return "", fmt.Errorf("message %d: %w", i, err)
		}
		role := ""
		switch m.Role {
		case genaiapi.System:
			if i != 0 {
				return "", fmt.Errorf("message %d: system message must be first message", i)
			}
			// System prompt is passed differently.
			sp = m.Text
			continue
		case genaiapi.User:
			role = "user"
		case genaiapi.Assistant:
			role = "model"
		default:
			return "", fmt.Errorf("message %d: unsupported role %q", i, m.Role)
		}
		cont := Content{Parts: []Part{{}}, Role: role}
		switch m.Type {
		case genaiapi.Text:
			cont.Parts[0].Text = m.Text
		case genaiapi.Document:
			mimeType := ""
			var data []byte
			if m.URL == "" {
				// If more than 20MB, we need to use
				// https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#large-pdfs-urls
				// cacheName, err := c.cacheContent(ctx, context, mime, sp)
				// When using cached content, system instruction, tools or tool_config cannot be used. Weird.
				// in.CachedContent = cacheName
				var err error
				if mimeType, data, err = internal.ParseDocument(&m, 10*1024*1024); err != nil {
					return "", fmt.Errorf("message %d: %w", i, err)
				}
				cont.Parts[0].InlineData.MimeType = mimeType
				cont.Parts[0].InlineData.Data = data
			} else {
				if mimeType = mime.TypeByExtension(path.Base(m.URL)); mimeType == "" {
					return "", fmt.Errorf("message %d: unsupported mime type for URL %q", i, m.URL)
				}
				cont.Parts[0].FileData.MimeType = mimeType
				cont.Parts[0].FileData.FileURI = m.URL
			}
		default:
			return "", fmt.Errorf("message %d: unsupported content type %s", i, m.Type)
		}
		c.Contents = append(c.Contents, cont)
	}
	return sp, nil
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerateContentResponse
type CompletionResponse struct {
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.Candidate
	Candidates []struct {
		Content Content `json:"content"`
		// https://ai.google.dev/api/generate-content?hl=en#FinishReason
		FinishReason string `json:"finishReason"`
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
			} `json:"citaionSources"`
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
			GroundingChuncks []struct {
				// https://ai.google.dev/api/generate-content?hl=en#Web
				Web struct {
					URI   string `json:"uri"`
					Title string `json:"title"`
				} `json:"web"`
			} `json:"groundingChuncks"`
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
	PromptFeedback any `json:"promptFeedback,omitzero"`
	// https://ai.google.dev/api/generate-content?hl=en#UsageMetadata
	UsageMetadata struct {
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
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.ModalityTokenCount
type ModalityTokenCount struct {
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.ModalityTokenCount
	Modality   string `json:"modality"` // MODALITY_UNSPECIFIED, TEXT, IMAGE, AUDIO
	TokenCount int64  `json:"tokenCount"`
}

type CompletionStreamChunkResponse struct {
	Candidates []struct {
		Content      Content `json:"content"`
		FinishReason string  `json:"finishReason"` // STOP
	} `json:"candidates"`
	UsageMetadata struct {
		CandidatesTokenCount    int64                `json:"candidatesTokenCount"`
		PromptTokenCount        int64                `json:"promptTokenCount"`
		TotalTokenCount         int64                `json:"totalTokenCount"`
		PromptTokensDetails     []ModalityTokenCount `json:"promptTokensDetails"`
		CandidatesTokensDetails []ModalityTokenCount `json:"candidatesTokensDetails"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
}

// Caching

/*
// https://ai.google.dev/api/caching?hl=en#request-body
type cachedContentRequest struct {
	Contents          []Content  `json:"contents"`
	Tools             []Tool     `json:"tools,omitzero"`
	Expiration        expiration `json:"expiration,omitzero"`
	Name              string     `json:"name,omitzero"`
	DisplayName       string     `json:"displayName,omitzero"`
	Model             string     `json:"model"`
	SystemInstruction Content    `json:"systemInstruction"`
	ToolConfig        ToolConfig `json:"toolConfig,omitzero"`
}

// https://ai.google.dev/api/caching?hl=en#CachedContent
type cacheContentResponse struct {
	Contents          []Content            `json:"contents"`
	Tools             []Tool               `json:"tools,omitzero"`
	CreateTime        string               `json:"createTime"`
	UpdateTime        string               `json:"updateTime"`
	UsageMetadata     cachingUsageMetadata `json:"usageMetadata"`
	Expiration        expiration           `json:"expiration"`
	Name              string               `json:"name"`
	DisplayName       string               `json:"displayName"`
	Model             string               `json:"model"`
	SystemInstruction Content              `json:"systemInstruction"`
	ToolConfig        ToolConfig           `json:"toolConfig,omitzero"`
}

type expiration struct {
	ExpireTime string `json:"expireTime"` // ISO 8601
	TTL        string `json:"ttl"`        // Duration
}

// https://ai.google.dev/api/caching?hl=en#UsageMetadata
type cachingUsageMetadata struct {
	TotakTokenCount int64 `json:"totalTokenCount"`
}
*/

//

type errorResponse struct {
	Error errorResponseError `json:"error"`
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
	return &Client{apiKey: apiKey, model: model}, nil
}

/*
func (c *Client) cacheContent(ctx context.Context, data []byte, mime, systemInstruction string) (string, error) {
	// See https://ai.google.dev/gemini-api/docs/caching?hl=en&lang=rest#considerations
	// It's only useful when reusing the same large data multiple times.
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=" + c.apiKey
	in := cachedContentRequest{
		// This requires a pinned model, with trailing -001.
		Model: "models/" + c.model,
		Contents: []Content{
			{
				Parts: []Part{{InlineData: Blob{MimeType: mime, Data: data}}},
				Role:  "user",
			},
		},
		SystemInstruction: Content{Parts: []Part{{Text: systemInstruction}}},
		Expiration: expiration{
			TTL: "120s",
		},
	}
	out := cacheContentResponse{}
	if err := c.post(ctx, url, &in, &out); err != nil {
		return "", err
	}
	// TODO: delete cache.
	slog.InfoContext(ctx, "gemini", "cached", out.Name)
	return out.Name, nil
}
*/

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (genaiapi.Message, error) {
	msg := genaiapi.Message{}
	in := CompletionRequest{}
	if err := in.fromOpts(opts); err != nil {
		return msg, err
	}
	sp, err := in.fromMsgs(msgs)
	if err != nil {
		return msg, err
	}
	if sp != "" {
		in.SystemInstruction.Parts = []Part{{Text: sp}}
	}
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return msg, err
	}
	if len(out.Candidates) != 1 {
		return msg, fmt.Errorf("unexpected number of candidates; expected 1, got %v", out.Candidates)
	}
	parts := out.Candidates[0].Content.Parts
	msg.Type = genaiapi.Text
	msg.Text = parts[len(parts)-1].Text
	switch role := out.Candidates[0].Content.Role; role {
	case "system", "user":
		msg.Role = genaiapi.Role(role)
	case "model":
		msg.Role = genaiapi.Assistant
	default:
		return msg, fmt.Errorf("unsupported role %q", role)
	}
	return msg, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	// https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
	if err := c.validate(); err != nil {
		return err
	}
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + c.model + ":generateContent?key=" + c.apiKey
	return c.post(ctx, url, in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, chunks chan<- genaiapi.MessageChunk) error {
	in := CompletionRequest{}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if _, err := in.fromMsgs(msgs); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan error)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		lastRole := genaiapi.System
		for pkt := range ch {
			if len(pkt.Candidates) != 1 {
				continue
			}
			switch role := pkt.Candidates[0].Content.Role; role {
			case "system", "assistant", "user":
				lastRole = genaiapi.Role(role)
			case "model":
				lastRole = genaiapi.Assistant
			case "":
			default:
				cancel()
				// We need to empty the channel to avoid blocking the goroutine.
				for range ch {
				}
				end <- fmt.Errorf("unexpected role %q", role)
				return
			}
			for _, part := range pkt.Candidates[0].Content.Parts {
				if part.Text != "" {
					chunks <- genaiapi.MessageChunk{Role: lastRole, Type: genaiapi.Text, Text: part.Text}
				}
			}
		}
		end <- nil
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := <-end; err2 != nil {
		err = err2
	}
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerateContentResponse
	if err := c.validate(); err != nil {
		return err
	}
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + c.model + ":streamGenerateContent?alt=sse&key=" + c.apiKey
	// Eventually, use OAuth https://ai.google.dev/gemini-api/docs/oauth#curl
	p := httpjson.DefaultClient
	// Google supports HTTP POST gzip compression!
	p.PostCompress = "gzip"
	resp, err := p.PostRequest(ctx, url, nil, in)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	r := bufio.NewReader(resp.Body)
	for {
		line, err := r.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if err == io.EOF {
			err = nil
			if len(line) == 0 {
				return nil
			}
		}
		if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		const prefix = "data: "
		if !bytes.HasPrefix(line, []byte(prefix)) {
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		suffix := string(line[len(prefix):])
		d := json.NewDecoder(strings.NewReader(suffix))
		d.DisallowUnknownFields()
		d.UseNumber()
		msg := CompletionStreamChunkResponse{}
		if err = d.Decode(&msg); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		out <- msg
	}
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
	return fmt.Sprintf("%s: %s (%s) Context: %d", m.GetID(), m.DisplayName, m.Description, m.InputTokenLimit)
}

func (m *Model) Context() int64 {
	return m.InputTokenLimit
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://ai.google.dev/api/models?hl=en#method:-models.list
	var out struct {
		Models        []Model `json:"models"`
		NextPageToken string  `json:"nextPageToken"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key="+c.apiKey, nil, &out)
	if err != nil {
		return nil, err
	}
	models := make([]genaiapi.Model, len(out.Models))
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

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	// Eventually, use OAuth https://ai.google.dev/gemini-api/docs/oauth#curl
	p := httpjson.DefaultClient
	// Google supports HTTP POST gzip compression!
	p.PostCompress = "gzip"
	resp, err := p.PostRequest(ctx, url, nil, in)
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
				return fmt.Errorf("%w: error %d (%s): %s You can get a new API key at %s", herr, er.Error.Code, er.Error.Status, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error %d (%s): %s", herr, er.Error.Code, er.Error.Status, er.Error.Message)
		}
		return fmt.Errorf("error %d (%s): %s", er.Error.Code, er.Error.Status, er.Error.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "gemini", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "gemini", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://ai.google.dev/gemini-api/docs/getting-started"

var (
	_ genaiapi.CompletionProvider = &Client{}
	_ genaiapi.ModelProvider      = &Client{}
)
