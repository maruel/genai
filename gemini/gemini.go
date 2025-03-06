// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
	"unicode"
)

type partInlineData struct {
	MimeType string `json:"mime_type,omitempty"`
	Data     []byte `json:"data,omitempty"`
}

type fileData struct {
	MimeType string `json:"mime_type,omitempty"`
	FileURI  string `json:"file_uri,omitempty"`
}

type part struct {
	Text string `json:"text,omitempty"`
	// Uploaded with /v1beta/cachedContents. Content is deleted after 1 hour.
	InlineData partInlineData `json:"inline_data,omitzero"`
	// Uploaded with /upload/v1beta/files. Files are deleted after 2 days.
	FileData fileData `json:"file_data,omitzero"`
}

type content struct {
	Parts []part `json:"parts"`
	// Must be either 'user' or 'model'.
	Role string `json:"role,omitempty"`
}

type toolData struct {
	DynamicRetrievalConfig dynamicRetrievalConfig `json:"dynamic_retrieval_config,omitzero"`
}

type dynamicRetrievalConfig struct {
	Mode              string `json:"mode"`
	DynamicThreshold  int    `json:"dynamic_threshold"`
	MaxDynamicResults int    `json:"max_dynamic_results"`
}

type tool map[string]toolData

type generateContentRequest struct {
	Model    string    `json:"model,omitempty"`
	Contents []content `json:"contents"`
	/*
		// Range [0, 2]
		Temperature float32   `json:"temperature,omitzero"`
		TopP        float32   `json:"topP,omitzero"`
		TopK        int32     `json:"topK,omitzero"`
	*/
	/*
		SafetySettings:    transformSlice(m.SafetySettings, (*SafetySetting).toProto),
		ToolConfig:        m.ToolConfig.toProto(),
		GenerationConfig:  m.GenerationConfig.toProto(),
	*/
	Tools             []tool  `json:"tools,omitempty"`
	SystemInstruction content `json:"systemInstruction,omitzero"`
	CachedContent     string  `json:"cachedContent,omitempty"`
}

type safetyRatings struct {
	// Must be one of the following:
	// - "HARM_CATEGORY_HATE_SPEECH"
	// - "HARM_CATEGORY_DANGEROUS_CONTENT"
	// - "HARM_CATEGORY_HARASSMENT"
	// - "HARM_CATEGORY_SEXUALLY_EXPLICIT"
	Category string `json:"category"`
	// Must be one of the following:
	// - "VERY_UNLIKELY"
	// - "UNLIKELY"
	// - "POSSIBLE"
	// - "LIKELY"
	// - "VERY_LIKELY"
	Probability string `json:"probability"`
}

type errorResponse struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
}

type generateContentResponse struct {
	Candidates    []generateContentResponseCandidate `json:"candidates"`
	UsageMetadata usageMetadata                      `json:"usageMetadata"`
	ModelVersion  string                             `json:"modelVersion"`
	Error         errorResponse                      `json:"error"`
}

type generateContentResponseCandidate struct {
	Content content `json:"content"`
	// Likely "STOP"
	FinishReason  string          `json:"finishReason"`
	AvgLogprobs   float64         `json:"avgLogprobs"`
	SafetyRatings []safetyRatings `json:"safetyRatings,omitempty"`
	// https://ai.google.dev/gemini-api/docs/grounding?hl=en&lang=rest#grounded-response
	// groundingMetadata
}

type usageMetadata struct {
	PromptTokenCount        int                                                `json:"promptTokenCount"`
	CandidatesTokenCount    int                                                `json:"candidatesTokenCount"`
	TotalTokenCount         int                                                `json:"totalTokenCount"`
	CachedContentTokenCount int                                                `json:"cachedContentTokenCount"`
	PromptTokensDetails     []generateContentResponseUsageMetadataTokenDetails `json:"promptTokensDetails"`
	CandidatesTokensDetails []generateContentResponseUsageMetadataTokenDetails `json:"candidatesTokensDetails"`
}

type generateContentResponseUsageMetadataTokenDetails struct {
	Modality   string `json:"modality"`
	TokenCount int    `json:"tokenCount"`
}

// https://ai.google.dev/gemini-api/docs/pricing?hl=en
// https://pkg.go.dev/github.com/google/generative-ai-go/genai no need to use this package, it imports too much.
type Client struct {
	ApiKey string
	// See models at https://ai.google.dev/gemini-api/docs/models/gemini?hl=en.
	// Using large files (over 32KB) requires a pinned model with caching
	// support.
	Model string
}

type cachedContentRequest struct {
	Model             string    `json:"model"`
	Contents          []content `json:"contents"`
	SystemInstruction content   `json:"systemInstruction"`
	TTL               string    `json:"ttl"`
}

type cacheContentResponse struct {
	Name          string        `json:"name"`
	Model         string        `json:"model"`
	CreateTime    string        `json:"createTime"`
	UpdateTime    string        `json:"updateTime"`
	ExpireTime    string        `json:"expireTime"`
	DisplayName   string        `json:"displayName"`
	UsageMetadata usageMetadata `json:"usageMetadata"`
}

func (c *Client) cacheContent(ctx context.Context, data []byte, mime, systemInstruction string) (string, error) {
	// See https://ai.google.dev/gemini-api/docs/caching?hl=en&lang=rest#considerations
	// It's only useful when reusing the same large data multiple times.
	url := "https://generativelanguage.googleapis.com/v1beta/cachedContents?key=" + c.ApiKey
	request := cachedContentRequest{
		// This requires a pinned model, with trailing -001.
		Model: "models/" + c.Model,
		Contents: []content{
			{
				Parts: []part{{InlineData: partInlineData{MimeType: mime, Data: data}}},
				Role:  "user",
			},
		},
		SystemInstruction: content{Parts: []part{{Text: systemInstruction}}},
		TTL:               "120s",
	}
	response := cacheContentResponse{}
	if err := c.post(ctx, url, &request, &response); err != nil {
		return "", err
	}
	// TODO: delete cache.
	slog.Info("gemini", "cached", response.Name)
	return response.Name, nil
}

func (c *Client) Query(ctx context.Context, systemPrompt, query string) (string, error) {
	return c.QueryContent(ctx, systemPrompt, query, "", nil)
}

func (c *Client) QueryContent(ctx context.Context, systemPrompt, query, mime string, context []byte) (string, error) {
	// https://ai.google.dev/gemini-api/docs?hl=en#rest
	url := "https://generativelanguage.googleapis.com/v1beta/models/" + c.Model + ":generateContent?key=" + c.ApiKey
	request := generateContentRequest{Model: "models/" + c.Model}
	response := generateContentResponse{}
	if len(context) > 0 {
		if len(context) >= 32768 {
			// If more than 20MB, we need to use https://ai.google.dev/gemini-api/docs/document-processing?hl=en&lang=rest#large-pdfs-urls

			// Pass the system instruction as a cached content while at it.
			cacheName, err := c.cacheContent(ctx, context, mime, systemPrompt)
			if err != nil {
				return "", err
			}
			// When using cached content, system instruction, tools or tool_config cannot be used. Weird.
			request.CachedContent = cacheName
		} else {
			// It's stronger when put there.
			request.SystemInstruction = content{Parts: []part{{Text: systemPrompt}}}
			request.Contents = append(request.Contents, content{
				Parts: []part{{InlineData: partInlineData{
					MimeType: "text/plain",
					Data:     context,
				}}},
				Role: "user",
			})
			/* TODO
			request.Tools = []tool{
				{
					"google_search_retrieval": {
						DynamicRetrievalConfig: dynamicRetrievalConfig{
							Mode:              "MODE_DYNAMIC",
							DynamicThreshold:  1,
							MaxDynamicResults: 1,
						},
					},
				},
			}
			*/
		}
	}
	request.Contents = append(request.Contents, content{Parts: []part{{Text: query}}, Role: "user"})
	if err := c.post(ctx, url, &request, &response); err != nil {
		return "", err
	}
	if len(response.Candidates) != 1 {
		return "", fmt.Errorf("unexpected number of candidates; expected 1, got %v", response.Candidates)
	}
	parts := response.Candidates[0].Content.Parts
	t := strings.TrimRightFunc(parts[len(parts)-1].Text, unicode.IsSpace)
	slog.Info("gemini", "query", query, "response", t, "usage", response.UsageMetadata)
	return t, nil
}

func (c *Client) post(ctx context.Context, url string, request, response any) error {
	rawData, err := json.Marshal(request)
	if err != nil {
		return err
	}
	start := time.Now()
	slog.Debug("gemini", "method", "post", "url", url, "in", string(rawData))
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(rawData))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		slog.Error("gemini", "method", "post", "duration", time.Since(start), "err", err)
		return err
	}
	defer resp.Body.Close()
	if rawData, err = io.ReadAll(resp.Body); err != nil {
		err = fmt.Errorf("failed to read response for URL %s: %w", url, err)
		slog.Error("gemini", "method", "post", "duration", time.Since(start), "err", err)
		return err
	}
	if resp.StatusCode != 200 {
		// TODO: process .Error.
		err = fmt.Errorf("unexpected status code for URL %s: status %d\nOriginal data: %s", url, resp.StatusCode, rawData)
		slog.Error("gemini", "method", "post", "duration", time.Since(start), "err", err)
		return err
	}
	d := json.NewDecoder(bytes.NewReader(rawData))
	d.DisallowUnknownFields()
	if err := d.Decode(response); err != nil {
		err = fmt.Errorf("Failed to decode: %w\nOriginal data: %s", err, rawData)
		slog.Error("gemini", "method", "post", "duration", time.Since(start), "err", err)
		return err
	}
	slog.Debug("gemini", "method", "post", "duration", time.Since(start), "response", string(rawData))
	return nil
}
