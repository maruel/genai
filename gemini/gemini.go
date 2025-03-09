// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"unicode"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://ai.google.dev/api/caching?hl=en#Blob
type Blob struct {
	MimeType string `json:"mimeType,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FileData
type FileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri,omitempty"`
}

// https://protobuf.dev/reference/protobuf/google.protobuf/#struct
type structValue map[string]Value

// https://protobuf.dev/reference/protobuf/google.protobuf/#value
// TODO: Confirm.
type Value struct {
	NullValue   int64       `json:"null_value,omitempty"`
	NumberValue float64     `json:"number_value,omitempty"`
	StringValue string      `json:"string_value,omitempty"`
	BoolValue   bool        `json:"bool_value,omitempty"`
	StructValue structValue `json:"struct_value,omitempty"`
	ListValue   []Value     `json:"list_value,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#CodeExecutionResult
type CodeExecutionResult struct {
	Outcome string `json:"outcome,omitempty"` // One of OUTCOME_UNSPECIFIED, OUTCOME_OK, OUTCOME_FAILED, OUTCOME_DEADLINE_EXCEEDED
	Output  string `json:"output,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#ExecutableCode
type ExecutableCode struct {
	Language string `json:"language,omitempty"` // Only PYTHON is supported as of March 2025.
	Code     string `json:"code,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FunctionCall
type FunctionCall struct {
	ID   string      `json:"id,omitempty"`
	Name string      `json:"name,omitempty"`
	Args structValue `json:"args,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FunctionResponse
type FunctionResponse struct {
	ID       string      `json:"id,omitempty"`
	Name     string      `json:"name,omitempty"`
	Response structValue `json:"response,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#Part
//
// Part is a union that only has one of the field set.
type Part struct {
	Text string `json:"text,omitempty"`
	// Uploaded with /v1beta/cachedContents. Content is deleted after 1 hour.
	InlineData       Blob             `json:"inlineData,omitzero"`
	FunctionCall     FunctionCall     `json:"functionCall,omitzero"`
	FunctionResponse FunctionResponse `json:"functionResponse,omitzero"`
	// Uploaded with /upload/v1beta/files. Files are deleted after 2 days.
	FileData            FileData            `json:"fileData,omitzero"`
	ExecutableCode      ExecutableCode      `json:"executableCode,omitzero"`      // TODO
	CodeExecutionResult CodeExecutionResult `json:"codeExecutionResult,omitzero"` // TODO
}

// https://ai.google.dev/api/caching?hl=en#Content
type Content struct {
	Parts []Part `json:"parts"`
	// Must be either 'user' or 'model'.
	Role string `json:"role,omitempty"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerationConfig
type GenerationConfig struct {
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
	ResponseLogprobs           bool     `json:"responseLogprobs,omitempty"`
	Logprobs                   int64    `json:"logProbs,omitempty"`
	EnableEnhancedCivicAnswers bool     `json:"enableEnhancedCivicAnswers,omitempty"`
	SpeechConfig               any      `json:"speechConfig,omitempty"` // TODO
	MediaResolution            string   `json:"mediaResolution,omitempty"`
}

// https://ai.google.dev/api/generate-content?hl=en#v1beta.SafetySetting
type SafetySetting struct {
	Category  string `json:"category"`  // https://ai.google.dev/api/generate-content?hl=en#v1beta.HarmCategory
	Threshold int64  `json:"threshold"` // https://ai.google.dev/api/generate-content?hl=en#HarmBlockThreshold
}

// https://ai.google.dev/api/caching?hl=en#GoogleSearchRetrieval
type GoogleSearchRetrieval struct {
	DynamicRetrievalConfig DynamicRetrievalConfig `json:"dynamicRetrievalConfig,omitzero"`
}

// https://ai.google.dev/api/caching?hl=en#DynamicRetrievalConfig
type DynamicRetrievalConfig struct {
	// https://ai.google.dev/api/caching?hl=en#Mode
	Mode             string  `json:"mode,omitempty"` // MODE_UNSPECIFIED, MODE_DYNAMIC
	DynamicThreshold float64 `json:"dynamicThreshold,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#Tool
type Tool struct {
	FunctionDeclarations  []functionDeclaration `json:"functionDeclaration,omitempty"`
	GoogleSearchRetrieval GoogleSearchRetrieval `json:"googleSearchRetrieval,omitempty"`
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
type ToolConfig struct {
	FunctionCallingConfig functionCallingConfig `json:"functionCallingConfig,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#FunctionCallingConfig
type functionCallingConfig struct {
	// https://ai.google.dev/api/caching?hl=en#Mode_1
	Mode                 string   `json:"mode,omitempty"` // MODE_UNSPECIFIED, AUTO, ANY, NONE
	AllowedFunctionNames []string `json:"allowedFunctionNames,omitempty"`
}

// https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
type CompletionRequest struct {
	Contents          []Content        `json:"contents"`
	Tools             []Tool           `json:"tools,omitempty"`
	ToolConfig        ToolConfig       `json:"toolConfig,omitempty"`
	SafetySettings    []SafetySetting  `json:"safetySettings,omitempty"`
	SystemInstruction Content          `json:"systemInstruction,omitzero"`
	GenerationConfig  GenerationConfig `json:"generationConfig,omitempty"`
	CachedContent     string           `json:"cachedContent,omitempty"`
}

func (c *CompletionRequest) fromOpts(opts any) error {
	// This doesn't seem to be well supported yet:
	//    in.GenerationConfig.ResponseLogprobs = true
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		c.GenerationConfig.MaxOutputTokens = v.MaxTokens
		c.GenerationConfig.Temperature = v.Temperature
		c.GenerationConfig.Seed = v.Seed
	default:
		return fmt.Errorf("unsupported options type %T", opts)
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) (string, error) {
	state := 0
	sp := ""
	for i, m := range msgs {
		switch m.Role {
		case genaiapi.System:
			if state != 0 {
				return sp, fmt.Errorf("unexpected system message at index %d; state %d", i, state)
			}
			sp = m.Content
			state = 1
		case genaiapi.Assistant:
			state = 1
			c.Contents = append(c.Contents, Content{Parts: []Part{{Text: m.Content}}, Role: "model"})
		case genaiapi.User:
			state = 1
			c.Contents = append(c.Contents, Content{Parts: []Part{{Text: m.Content}}, Role: "user"})
		default:
			return sp, fmt.Errorf("unexpected role %q", m.Role)
		}
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
	PromptFeedback any `json:"promptFeedback,omitempty"`
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

// https://ai.google.dev/api/caching?hl=en#request-body
type cachedContentRequest struct {
	Contents          []Content  `json:"contents"`
	Tools             []Tool     `json:"tools,omitempty"`
	Expiration        expiration `json:"expiration,omitempty"`
	Name              string     `json:"name,omitempty"`
	DisplayName       string     `json:"displayName,omitempty"`
	Model             string     `json:"model"`
	SystemInstruction Content    `json:"systemInstruction"`
	ToolConfig        ToolConfig `json:"toolConfig,omitempty"`
}

// https://ai.google.dev/api/caching?hl=en#CachedContent
type cacheContentResponse struct {
	Contents          []Content            `json:"contents"`
	Tools             []Tool               `json:"tools,omitempty"`
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

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	return c.CompletionContent(ctx, msgs, opts, "", nil)
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	// https://ai.google.dev/api/generate-content?hl=en#text_gen_text_only_prompt-SHELL
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + c.Model + ":generateContent?key=" + c.ApiKey
	return c.post(ctx, url, in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	in := CompletionRequest{}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if _, err := in.fromMsgs(msgs); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan struct{})
	go func() {
		for msg := range ch {
			if len(msg.Candidates) != 1 {
				continue
			}
			parts := msg.Candidates[0].Content.Parts
			word := strings.TrimRightFunc(parts[len(parts)-1].Text, unicode.IsSpace)
			if word != "" {
				words <- word
			}
		}
		end <- struct{}{}
	}()
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	<-end
	return err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	// https://ai.google.dev/api/generate-content?hl=en#v1beta.GenerateContentResponse
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + c.Model + ":streamGenerateContent?alt=sse&key=" + c.ApiKey
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

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, context []byte) (string, error) {
	in := CompletionRequest{}
	if err := in.fromOpts(opts); err != nil {
		return "", err
	}
	sp, err := in.fromMsgs(msgs)
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
			in.SystemInstruction = Content{Parts: []Part{{Text: sp}}}
			in.Contents = append([]Content{
				{
					Parts: []Part{{InlineData: Blob{MimeType: "text/plain", Data: context}}},
					Role:  "user",
				},
			}, in.Contents...)
		}
	}

	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
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

func (c *Client) ListModels(ctx context.Context) ([]Model, error) {
	// https://ai.google.dev/api/models?hl=en#method:-models.list
	var out struct {
		Models        []Model `json:"models"`
		NextPageToken string  `json:"nextPageToken"`
	}
	err := httpjson.DefaultClient.Get(ctx, "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key="+c.ApiKey, nil, &out)
	return out.Models, err
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	if c.ApiKey == "" {
		return errors.New("gemini ApiKey is required; get one at " + apiKeyURL)
	}
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

var _ genaiapi.CompletionProvider = &Client{}
