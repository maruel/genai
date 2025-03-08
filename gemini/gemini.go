// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"unicode"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://ai.google.dev/api/caching?hl=en#Blob
type blob struct {
	MimeType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FileData
type fileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri,omitempty"`
}

// https://protobuf.dev/reference/protobuf/google.protobuf/#struct
type structValue map[string]value

// https://protobuf.dev/reference/protobuf/google.protobuf/#value
// TODO: Confirm.
type value struct {
	NullValue   int64       `json:"null_value,omitempty"`
	NumberValue float64     `json:"number_value,omitempty"`
	StringValue string      `json:"string_value,omitempty"`
	BoolValue   bool        `json:"bool_value,omitempty"`
	StructValue structValue `json:"struct_value,omitempty"`
	ListValue   []value     `json:"list_value,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#CodeExecutionResult
type codeExecutionResult struct {
	Outcome string `json:"outcome,omitempty"` // One of OUTCOME_UNSPECIFIED, OUTCOME_OK, OUTCOME_FAILED, OUTCOME_DEADLINE_EXCEEDED
	Output  string `json:"output,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#ExecutableCode
type executableCode struct {
	Language string `json:"language,omitempty"` // Only PYTHON is supported as of March 2025.
	Code     string `json:"code,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FunctionCall
type functionCall struct {
	ID   string      `json:"id,omitempty"`
	Name string      `json:"name,omitempty"`
	Args structValue `json:"args,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FunctionResponse
type functionResponse struct {
	ID       string      `json:"id,omitempty"`
	Name     string      `json:"name,omitempty"`
	Response structValue `json:"response,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#Part
//
// part is a union that only has one of the field set.
type part struct {
	Text string `json:"text,omitempty"`
	// Uploaded with /v1beta/cachedContents. Content is deleted after 1 hour.
	InlineData       blob             `json:"inlineData,omitzero"`
	FunctionCall     functionCall     `json:"functionCall,omitzero"`
	FunctionResponse functionResponse `json:"functionResponse,omitzero"`
	// Uploaded with /upload/v1beta/files. Files are deleted after 2 days.
	FileData            fileData            `json:"fileData,omitzero"`
	ExecutableCode      executableCode      `json:"executableCode,omitzero"`      // TODO
	CodeExecutionResult codeExecutionResult `json:"codeExecutionResult,omitzero"` // TODO
}

// https://ai.google.dev/api/caching?hl=en#Content
type content struct {
	Parts []part `json:"parts"`
	// Must be either 'user' or 'model'.
	Role string `json:"role,omitempty"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerationConfig
type generationConfig struct {
	StopSequences              []string `json:"stopSequences,omitempty"`
	ResponseMimeType           string   `json:"responseMimeType,omitempty"`
	ResponseSchema             any      `json:"responseSchema,omitempty"` // TODO
	ResponseModalities         []string `json:"responseModalities,omitempty"`
	CandidateCount             int64    `json:"candidateCount,omitempty"`
	MaxOutputTokens            int64    `json:"maxOutputTokens,omitempty"`
	Temperature                float64  `json:"temperature,omitempty"` // [0, 2]
	TopP                       float64  `json:"topP,omitempty"`
	TopK                       int64    `json:"topK,omitempty"`
	Seed                       int64    `json:"seed,omitempty"`
	PresencePenalty            float64  `json:"presencePenalty,omitempty"`
	FrequencyPenalty           float64  `json:"frequencyPenalty,omitempty"`
	ResponseLogProbs           bool     `json:"responseLogProbs,omitempty"`
	LogProbs                   int64    `json:"logProbs,omitempty"`
	EnableEnhancedCivicAnswers bool     `json:"enableEnhancedCivicAnswers,omitempty"`
	SpeechConfig               any      `json:"speechConfig,omitempty"` // TODO
	MediaResolution            string   `json:"mediaResolution,omitempty"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.SafetySetting
type safetySetting struct {
	Category  string `json:"category"`  // https://ai.google.dev/api/generate-content?hl=en#v1beta.HarmCategory
	Threshold int64  `json:"threshold"` // https://ai.google.dev/api/generate-content?hl=en#HarmBlockThreshold
}

// https://ai.google.dev/api/caching?hl=en#GoogleSearchRetrieval
type googleSearchRetrieval struct {
	DynamicRetrievalConfig dynamicRetrievalConfig `json:"dynamicRetrievalConfig,omitzero"`
}

// https://ai.google.dev/api/caching?hl=en#DynamicRetrievalConfig
type dynamicRetrievalConfig struct {
	// https://ai.google.dev/api/caching?hl=en#Mode
	Mode             string  `json:"mode,omitempty"` // MODE_UNSPECIFIED, MODE_DYNAMIC
	DynamicThreshold float64 `json:"dynamicThreshold,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#Tool
type tool struct {
	FunctionDeclarations  []functionDeclaration `json:"functionDeclaration,omitempty"`
	GoogleSearchRetrieval googleSearchRetrieval `json:"googleSearchRetrieval,omitempty"`
	CodeExecution         struct{}              `json:"codeExecution,omitempty"`
	GoogleSearch          struct{}              `json:"googleSearch,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FunctionDeclaration
type functionDeclaration struct {
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
	Parameters  schema `json:"parameters,omitempty"`
	Response    schema `json:"response,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#Schema
type schema struct {
	// https://ai.google.dev/api/caching?hl=en#Type
	Type             string            `json:"type,omitempty"`   // TYPE_UNSPECIFIED, STRING, NUMBER, INTEGER, BOOLEAN, ARRAY, OBJECT
	Format           string            `json:"format,omitempty"` // "json-schema"
	Description      string            `json:"description,omitempty"`
	Nullable         bool              `json:"nullable,omitempty"`
	Enum             []string          `json:"enum,omitempty"`
	MaxItems         int64             `json:"maxItems,omitempty"`
	MinItems         int64             `json:"minItems,omitempty"`
	Properties       map[string]schema `json:"properties,omitempty"`
	Required         []string          `json:"required,omitempty"`
	PropertyOrdering []string          `json:"propertyOrdering,omitempty"`
	Items            *schema           `json:"items,omitempty"` // When type is "array"
}

// https://ai.google.dev/api/caching?hl=en#ToolConfig
type toolConfig struct {
	FunctionCallingConfig functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FunctionCallingConfig
type functionCallingConfig struct {
	// https://ai.google.dev/api/caching?hl=en#Mode_1
	Mode                 string   `json:"mode,omitempty"` // MODE_UNSPECIFIED, AUTO, ANY, NONE
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
type generateContentRequest struct {
	Contents          []content        `json:"contents"`
	Tools             []tool           `json:"tools,omitempty"`
	ToolConfig        toolConfig       `json:"toolConfig,omitempty"`
	SafetySettings    []safetySetting  `json:"safetySettings,omitempty"`
	SystemInstruction content          `json:"systemInstruction,omitzero"`
	GenerationConfig  generationConfig `json:"generationConfig,omitempty"`
	CachedContent     string           `json:"cachedContent,omitempty"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerateContentResponse
type generateContentResponse struct {
	Candidates     []candidate                  `json:"candidates"`
	PromptFeedback any                          `json:"promptFeedback,omitempty"`
	UsageMetadata  generateContentUsageMetadata `json:"usageMetadata"`
	ModelVersion   string                       `json:"modelVersion"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.Candidate
type candidate struct {
	Content content `json:"content"`
	// https://ai.google.dev/api/generate-content?hl=en#FinishReason
	FinishReason          string                 `json:"finishReason"`
	SafetyRatings         []safetyRating         `json:"safetyRatings"`
	CitationMetadata      citationMetadata       `json:"citationMetadata"`
	TokenCount            int64                  `json:"tokenCount"`
	GroundingAttributions []groundingAttribution `json:"groundingAttributions"`
	GroundingMetadata     groundingMetadata      `json:"groundingMetadata"`
	AvgLogprobs           float64                `json:"avgLogprobs"`
	LogprobsResult        any                    `json:"logprobsResult"`
	Index                 int64                  `json:"index"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.CitationMetadata
type citationMetadata struct {
	CitaionSources []citationSource `json:"citaionSources"`
}

// https://ai.google.dev/api/generate-content?hl=en#CitationSource
type citationSource struct {
	StartIndex int64  `json:"startIndex"`
	EndIndex   int64  `json:"endIndex"`
	URI        string `json:"uri"`
	License    string `json:"license"`
}

// https://ai.google.dev/api/generate-content?hl=en#GroundingAttribution
type groundingAttribution struct {
	SourceID string  `json:"sourceId"`
	Countent content `json:"countent"`
}

// https://ai.google.dev/api/generate-content?hl=en#GroundingMetadata
type groundingMetadata struct {
	GroundingChuncks  []groundingChunk   `json:"groundingChuncks"`
	GroundingSupports []groundingSupport `json:"groundingSupports"`
	WebSearchQueries  []string           `json:"webSearchQueries"`
	SearchEntryPoint  searchEntryPoint   `json:"searchEntryPoint"`
	RetrievalMetadata retrievalMetadata  `json:"retrievalMetadata"`
}

// https://ai.google.dev/api/generate-content?hl=en#GroundingChunk
type groundingChunk struct {
	Web web `json:"web"`
}

// https://ai.google.dev/api/generate-content?hl=en#Web
type web struct {
	URI   string `json:"uri"`
	Title string `json:"title"`
}

// https://ai.google.dev/api/generate-content?hl=en#GroundingSupport
type groundingSupport struct {
	GroundingChunkIndices []int64   `json:"groundingChunkIndices"`
	ConfidenceScores      []float64 `json:"confidenceScores"`
	Segment               segment   `json:"segment"`
}

// https://ai.google.dev/api/generate-content?hl=en#Segment
type segment struct {
	PartIndex  int64  `json:"partIndex"`
	StartIndex int64  `json:"startIndex"`
	EndIndex   int64  `json:"endIndex"`
	Text       string `json:"text"`
}

// https://ai.google.dev/api/generate-content?hl=en#SearchEntryPoint
type searchEntryPoint struct {
	RenderedContent string `json:"renderedContent"`
	SDKBlob         []byte `json:"sdkBlob"` // JSON encoded list of (search term,search url) results
}

// https://ai.google.dev/api/generate-content?hl=en#RetrievalMetadata
type retrievalMetadata struct {
	GoogleSearchDynamicRetrievalScore float64 `json:"googleSearchDynamicRetrievalScore"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.SafetyRating
type safetyRating struct {
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.HarmCategory
	Category string `json:"category"`
	// https://ai.google.dev/api/generate-content?hl=en#HarmProbability
	Probability string `json:"probability"`
	Blocked     bool   `json:"blocked"`
}

// https://ai.google.dev/api/generate-content?hl=en#UsageMetadata
type generateContentUsageMetadata struct {
	PromptTokenCount           int64                `json:"promptTokenCount"`
	CachedContentTokenCount    int64                `json:"cachedContentTokenCount"`
	CandidatesTokenCount       int64                `json:"candidatesTokenCount"`
	ToolUsePromptTokenCount    int64                `json:"toolUsePromptTokenCount"`
	ThoughtsTokenCount         int64                `json:"thoughtsTokenCount"`
	TotalTokenCount            int64                `json:"totalTokenCount"`
	PromptTokensDetails        []modalityTokenCount `json:"promptTokensDetails"`
	CacheTokensDetails         []modalityTokenCount `json:"cacheTokensDetails"`
	CandidatesTokensDetails    []modalityTokenCount `json:"candidatesTokensDetails"`
	ToolUsePromptTokensDetails []modalityTokenCount `json:"toolUsePromptTokensDetails"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.ModalityTokenCount
type modalityTokenCount struct {
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.ModalityTokenCount
	Modality   string `json:"modality"` // MODALITY_UNSPECIFIED, TEXT, IMAGE, AUDIO
	TokenCount int64  `json:"tokenCount"`
}

// Caching

// https://ai.google.dev/api/caching?hl=en#request-body
type cachedContentRequest struct {
	Contents          []content  `json:"contents"`
	Tools             []tool     `json:"tools,omitempty"`
	Expiration        expiration `json:"expiration,omitempty"`
	Name              string     `json:"name,omitempty"`
	DisplayName       string     `json:"displayName,omitempty"`
	Model             string     `json:"model"`
	SystemInstruction content    `json:"systemInstruction"`
	ToolConfig        toolConfig `json:"toolConfig,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#CachedContent
type cacheContentResponse struct {
	Contents          []content            `json:"contents"`
	Tools             []tool               `json:"tools,omitempty"`
	CreateTime        string               `json:"createTime"`
	UpdateTime        string               `json:"updateTime"`
	UsageMetadata     cachingUsageMetadata `json:"usageMetadata"`
	Expiration        expiration           `json:"expiration"`
	Name              string               `json:"name"`
	DisplayName       string               `json:"displayName"`
	Model             string               `json:"model"`
	SystemInstruction content              `json:"systemInstruction"`
	ToolConfig        toolConfig           `json:"toolConfig,omitzero"`
}

type expiration struct {
	ExpireTime string `json:"expireTime"` // ISO 8601
	TTL        string `json:"ttl"`        // Duration
}

// https://ai.google.dev/api/caching?hl=en#UsageMetadata
type cachingUsageMetadata struct {
	TotakTokenCount int64 `json:"totalTokenCount"`
}

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
		Locale  string `json:"locale"`
		Message string `json:"message"`
	} `json:"details"`
}

// https://ai.google.dev/gemini-api/docs/pricing
// https://pkg.go.dev/github.com/google/generative-ai-go/genai no need to use this package, it imports too much.
type Client struct {
	ApiKey string
	// See models at https://ai.google.dev/gemini-api/docs/models/gemini
	// Using large files (over 32KB) requires a pinned model with caching
	// support.
	Model string
}

func (c *Client) cacheContent(ctx context.Context, data []byte, mime, systemInstruction string) (string, error) {
	// See https://ai.google.dev/gemini-api/docs/caching?hl=en&lang=rest#considerations
	// It's only useful when reusing the same large data multiple times.
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=" + c.ApiKey
	in := cachedContentRequest{
		// This requires a pinned model, with trailing -001.
		Model: "models/" + c.Model,
		Contents: []content{
			{
				Parts: []part{{InlineData: blob{MimeType: mime, Data: data}}},
				Role:  "user",
			},
		},
		SystemInstruction: content{Parts: []part{{Text: systemInstruction}}},
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

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	return c.CompletionContent(ctx, msgs, opts, "", nil)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) (string, error) {
	return "", errors.New("not implemented")
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, context []byte) (string, error) {
	// https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + c.Model + ":generateContent?key=" + c.ApiKey
	in := generateContentRequest{}
	sp, err := c.initPrompt(&in, msgs)
	if err != nil {
		return "", err
	}
	if len(context) > 0 {
		if len(context) >= 32768 {
			// If more than 20MB, we need to use https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#large-pdfs-urls
			// Pass the system instruction as a cached content while at it.
			cacheName, err := c.cacheContent(ctx, context, mime, sp)
			if err != nil {
				return "", err
			}
			// When using cached content, system instruction, tools or tool_config cannot be used. Weird.
			in.CachedContent = cacheName
		} else {
			// It's stronger when put there.
			in.SystemInstruction = content{Parts: []part{{Text: sp}}}
			in.Contents = append([]content{
				{
					Parts: []part{{InlineData: blob{MimeType: "text/plain", Data: context}}},
					Role:  "user",
				},
			}, in.Contents...)
		}
	}
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		in.GenerationConfig.MaxOutputTokens = v.MaxTokens
		in.GenerationConfig.Temperature = v.Temperature
		in.GenerationConfig.Seed = v.Seed
	default:
		return "", fmt.Errorf("unsupported options type %T", opts)
	}

	out := generateContentResponse{}
	if err := c.post(ctx, url, &in, &out); err != nil {
		return "", err
	}
	if len(out.Candidates) != 1 {
		return "", fmt.Errorf("unexpected number of candidates; expected 1, got %v", out.Candidates)
	}
	parts := out.Candidates[0].Content.Parts
	t := strings.TrimRightFunc(parts[len(parts)-1].Text, unicode.IsSpace)
	// slog.InfoContext(ctx, "gemini", "response", t, "usage", out.UsageMetadata)
	return t, nil
}

func (c *Client) initPrompt(r *generateContentRequest, msgs []genaiapi.Message) (string, error) {
	state := 0
	sp := ""
	for i, m := range msgs {
		switch m.Role {
		case genaiapi.System:
			if state != 0 {
				return "", fmt.Errorf("unexpected system message at index %d; state %d", i, state)
			}
			sp = m.Content
			state = 1
		case genaiapi.Assistant:
			state = 1
			r.Contents = append(r.Contents, content{Parts: []part{{Text: m.Content}}, Role: "model"})
		case genaiapi.User:
			state = 1
			r.Contents = append(r.Contents, content{Parts: []part{{Text: m.Content}}, Role: "user"})
		default:
			return sp, fmt.Errorf("unexpected role %q", m.Role)
		}
	}
	return sp, nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	if c.ApiKey == "" {
		return errors.New("gemini ApiKey is required; get one at " + apiKeyURL)
	}
	// Eventually, use OAuth https://ai.google.dev/gemini-api/docs/oauth#curl
	p := httpjson.DefaultClient
	// Google only support gzip but it's better than nothing.
	p.Compress = "gzip"
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
			if herr.StatusCode == http.StatusBadRequest || herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("http %d: error %d (%s): %s You can get a new API key at %s", herr.StatusCode, er.Error.Code, er.Error.Status, er.Error.Message, apiKeyURL)
			}
			return fmt.Errorf("http %d: error %d (%s): %s", herr.StatusCode, er.Error.Code, er.Error.Status, er.Error.Message)
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

var _ genaiapi.CompletionProvider = &Client{}
