// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cloudflare implements a client for the Cloudflare AI API.
//
// It is described at https://developers.cloudflare.com/api/resources/ai/
package cloudflare

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
	"strconv"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/httpjson"
)

// https://developers.cloudflare.com/api/resources/ai/methods/run/
type CompletionRequest struct {
	Messages          []Message      `json:"messages"`
	FrequencyPenalty  float64        `json:"frequency_penalty,omitzero"` // [0, 2.0]
	MaxTokens         int64          `json:"max_tokens,omitzero"`
	PresencePenalty   float64        `json:"presence_penalty,omitzero"`   // [0, 2.0]
	RepetitionPenalty float64        `json:"repetition_penalty,omitzero"` // [0, 2.0]
	ResponseFormat    ResponseFormat `json:"response_format,omitzero"`
	Seed              int64          `json:"seed,omitzero"`
	Stream            bool           `json:"stream,omitzero"`
	Temperature       float64        `json:"temperature,omitzero"` // [0, 5]
	Tools             []Tool         `json:"tools,omitzero"`       // Can be ToolFunction or ToolParameter
	TopK              int64          `json:"top_k,omitzero"`       // [1, 50]
	TopP              float64        `json:"top_p,omitzero"`       // [0, 2.0]

	// Functions         []function     `json:"functions,omitzero"`
}

type ResponseFormat struct {
	JSONSchema any    `json:"json_schema,omitzero"`
	Type       string `json:"type,omitzero"` // json_object, json_schema
}

type Message struct {
	Content string `json:"content"`
	Role    string `json:"role"`
}

type function struct {
	Code string `json:"code"`
	Name string `json:"name"`
}

type Tool struct {
	Function ToolFunction `json:"function"`
	Type     string       `json:"type"` // function
}

type ToolFunction struct {
	Description string         `json:"description"`
	Name        string         `json:"name"`
	Parameters  ToolParameters `json:"parameters"`
}

type ToolParameters struct {
	Properties map[string]ToolParameter `json:"properties"`
	Type       string                   `json:"type"`
	Required   []string                 `json:"required"`
}

type ToolParameter struct {
	Description string `json:"description"`
	Type        string `json:"type"`
}

type prompt struct {
	Prompt            string         `json:"prompt"`
	FrequencyPenalty  float64        `json:"frequency_penalty,omitzero"` // [0, 2.0]
	Lora              string         `json:"lora,omitzero"`
	MaxTokens         int64          `json:"max_tokens,omitzero"`
	PresencePenalty   float64        `json:"presence_penalty,omitzero"`   // [0, 2.0]
	Raw               bool           `json:"raw,omitzero"`                // Do not aply chat template
	RepetitionPenalty float64        `json:"repetition_penalty,omitzero"` // [0, 2.0]
	ResponseFormat    ResponseFormat `json:"response_format,omitzero"`
	Seed              int64          `json:"seed,omitzero"`
	Stream            bool           `json:"stream,omitzero"`
	Temperature       float64        `json:"temperature,omitzero"` // [0, 5]
	TopK              int64          `json:"top_k,omitzero"`       // [1, 50]
	TopP              float64        `json:"top_p,omitzero"`       // [0, 2.0]
}

type textClassification struct {
	Text string `json:"text"`
}

type textToImage struct {
	Prompt         string  `json:"prompt"`
	Guidance       float64 `json:"guidance,omitzero"`
	Height         int64   `json:"height,omitzero"` // [256, 2048]
	Image          []uint8 `json:"image,omitzero"`
	ImageB64       []byte  `json:"image_b64,omitzero"`
	Mask           []uint8 `json:"mask,omitzero"`
	NegativePrompt string  `json:"negative_prompt,omitzero"`
	NumSteps       int64   `json:"num_steps,omitzero"` // Max 20
	Seed           int64   `json:"seed,omitzero"`
	Strength       float64 `json:"strength,omitzero"` // [0, 1]
	Width          int64   `json:"width,omitzero"`    // [256, 2048]
}

type textToSpeech struct {
	Prompt string `json:"prompt"`
	Lang   string `json:"lang,omitzero"` // en, fr, etc
}

type textEmbeddings struct {
	Text []string `json:"text"`
}

type automaticSpeechRecognition struct {
	Audio      []uint8 `json:"audio"`
	SourceLang string  `json:"source_lang,omitzero"`
	TargetLang string  `json:"target_lang,omitzero"`
}

type imageClassification struct {
	Image []uint8 `json:"image"`
}

type objectDetection struct {
	Image []uint8 `json:"image,omitzero"`
}

type translation struct {
	TargetLang string  `json:"target_lang"`
	Text       string  `json:"text"`
	SourceLang *string `json:"source_lang,omitzero"`
}

type summarization struct {
	InputText string `json:"input_text"`
	MaxLength *int   `json:"max_length,omitzero"`
}

type imageToText struct {
	Image             []uint8 `json:"image"`
	FrequencyPenalty  float64 `json:"frequency_penalty,omitzero"`
	MaxTokens         int64   `json:"max_tokens,omitzero"`
	PresencePenalty   float64 `json:"presence_penalty,omitzero"`
	Prompt            string  `json:"prompt,omitzero"`
	Raw               bool    `json:"raw,omitzero"`
	RepetitionPenalty float64 `json:"repetition_penalty,omitzero"`
	Seed              int64   `json:"seed,omitzero"`
	Temperature       float64 `json:"temperature,omitzero"`
	TopK              int64   `json:"top_k,omitzero"`
	TopP              float64 `json:"top_p,omitzero"`
}

func (c *CompletionRequest) fromOpts(opts any) error {
	switch v := opts.(type) {
	case *genaiapi.CompletionOptions:
		c.MaxTokens = v.MaxTokens
		c.Seed = v.Seed
		c.Temperature = v.Temperature
	default:
		return fmt.Errorf("unsupported options type %T", opts)
	}
	return nil
}

func (c *CompletionRequest) fromMsgs(msgs []genaiapi.Message) error {
	c.Messages = make([]Message, len(msgs))
	for i, m := range msgs {
		c.Messages[i].Role = string(m.Role)
		c.Messages[i].Content = m.Content
	}
	return nil
}

type CompletionResponse struct {
	Result struct {
		Response  string `json:"response"`
		ToolCalls []struct {
			Arguments []string `json:"arguments"`
			Name      string   `json:"name"`
		} `json:"tool_calls"`
		Usage struct {
			CompletionTokens int64 `json:"completion_tokens"`
			PromptTokens     int64 `json:"prompt_tokens"`
			TotalTokens      int64 `json:"total_tokens"`
		} `json:"usage"`
	} `json:"result"`
	Success  bool       `json:"success"`
	Errors   []struct{} `json:"errors"`   // Annoyingly, it's included all the time
	Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
}

type CompletionStreamChunkResponse struct {
	Response string `json:"response"`
	P        string `json:"p"`
	Usage    struct {
		CompletionTokens int64 `json:"completion_tokens"`
		PromptTokens     int64 `json:"prompt_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

//

type errorResponse struct {
	Errors []struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"errors"`
	Success  bool       `json:"success"`
	Result   struct{}   `json:"result"`
	Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
}

type Client struct {
	// AccountID and ApiKey can be retrieved as "Workers AI" from https://dash.cloudflare.com/profile/api-tokens
	AccountID string
	ApiKey    string
	// Model to use, see https://developers.cloudflare.com/workers-ai/models/
	Model string
}

func (c *Client) Completion(ctx context.Context, msgs []genaiapi.Message, opts any) (string, error) {
	// https://developers.cloudflare.com/api/resources/ai/methods/run/
	in := CompletionRequest{}
	if err := in.fromOpts(opts); err != nil {
		return "", err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return "", err
	}
	out := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &in, &out); err != nil {
		return "", fmt.Errorf("failed to get chat response: %w", err)
	}
	return out.Result.Response, nil
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(true); err != nil {
		return err
	}
	url := "https://api.cloudflare.com/client/v4/accounts/" + c.AccountID + "/ai/run/" + c.Model
	return c.post(ctx, url, in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs []genaiapi.Message, opts any, words chan<- string) error {
	in := CompletionRequest{Stream: true}
	if err := in.fromOpts(opts); err != nil {
		return err
	}
	if err := in.fromMsgs(msgs); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	end := make(chan struct{})
	go func() {
		for msg := range ch {
			word := msg.Response
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
	if err := c.validate(true); err != nil {
		return err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	url := "https://api.cloudflare.com/client/v4/accounts/" + c.AccountID + "/ai/run/" + c.Model
	// Cloudflare doesn't HTTP POST support compression.
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
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
		const dataPrefix = "data: "
		switch {
		case bytes.HasPrefix(line, []byte(dataPrefix)):
			suffix := string(line[len(dataPrefix):])
			if suffix == "[DONE]" {
				return nil
			}
			d := json.NewDecoder(strings.NewReader(suffix))
			d.DisallowUnknownFields()
			d.UseNumber()
			msg := CompletionStreamChunkResponse{}
			if err = d.Decode(&msg); err != nil {
				return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
			}
			out <- msg
		default:
			er := errorResponse{}
			d := json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			d.UseNumber()
			if err = d.Decode(&er); err != nil || len(er.Errors) == 0 {
				return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
			}
			return fmt.Errorf("got server error: %s", er.Errors[0].Message)
		}
	}
}

func (c *Client) CompletionContent(ctx context.Context, msgs []genaiapi.Message, opts any, mime string, content []byte) (string, error) {
	return "", errors.New("not implemented")
}

type Model struct {
	ID          string `json:"id"`
	Source      int64  `json:"source"`
	Name        string `json:"name"`
	Description string `json:"description"`
	Task        struct {
		ID          string `json:"id"`
		Name        string `json:"name"`
		Description string `json:"description"`
	} `json:"task"`
	Tags       []string `json:"tags"`
	Properties []struct {
		PropertyID string `json:"property_id"`
		Value      string `json:"value"`
	} `json:"properties"`
}

func (m *Model) GetID() string {
	return m.Name
}

func (m *Model) String() string {
	var suffixes []string
	for _, p := range m.Properties {
		suffixes = append(suffixes, p.PropertyID+"="+p.Value)
	}
	suffix := ""
	if len(suffixes) != 0 {
		suffix = " (" + strings.Join(suffixes, ", ") + ")"
	}
	// Description is good but it's verbose and the models are well known.
	return fmt.Sprintf("%s%s", m.Name, suffix)
}

func (m *Model) Context() int64 {
	for _, p := range m.Properties {
		if p.PropertyID == "context_window" || p.PropertyID == "max_input_tokens" {
			if v, err := strconv.ParseInt(p.Value, 10, 64); err == nil {
				return v
			}
		}
	}
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genaiapi.Model, error) {
	// https://developers.cloudflare.com/api/resources/ai/subresources/models/methods/list/
	if err := c.validate(false); err != nil {
		return nil, err
	}
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	// See https://github.com/cloudflare/cloudflare-go/blob/main/internal/requestconfig/requestconfig.go
	h.Set("X-Stainless-Retry-Count", "0")
	h.Set("X-Stainless-Timeout", "0")
	var models []genaiapi.Model
	for page := 1; ; page++ {
		var out struct {
			Result     []Model `json:"result"`
			ResultInfo struct {
				Count      int64 `json:"count"`
				Page       int64 `json:"page"`
				PerPage    int64 `json:"per_page"`
				TotalCount int64 `json:"total_count"`
			} `json:"result_info"`
			Success  bool       `json:"success"`
			Errors   []struct{} `json:"errors"`   // Annoyingly, it's included all the time
			Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
		}
		// Cloudflare's pagination is surprisingly brittle.
		url := fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/models/search?page=%d&per_page=100&hide_experimental=false", c.AccountID, page)
		err := httpjson.DefaultClient.Get(ctx, url, h, &out)
		if err != nil {
			return nil, err
		}
		for i := range out.Result {
			models = append(models, &out.Result[i])
		}
		if len(models) >= int(out.ResultInfo.TotalCount) || len(out.Result) == 0 {
			break
		}
	}
	return models, nil
}

func (c *Client) validate(needModel bool) error {
	if c.AccountID == "" {
		return errors.New("cloudflare account ID is required; get one at " + apiKeyURL)
	}
	if c.ApiKey == "" {
		return errors.New("cloudflare Workers AI ApiKey is required; get one at " + apiKeyURL)
	}
	if needModel && c.Model == "" {
		return errors.New("a Model is required")
	}
	return nil
}

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	h := make(http.Header)
	h.Add("Authorization", "Bearer "+c.ApiKey)
	resp, err := httpjson.DefaultClient.PostRequest(ctx, url, h, in)
	if err != nil {
		return err
	}
	er := errorResponse{}
	switch i, err := httpjson.DecodeResponse(resp, out, &er); i {
	case 0:
		return nil
	case 1:
		if len(er.Errors) == 0 {
			return err
		}
		msg := er.Errors[0]
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			if herr.StatusCode == http.StatusUnauthorized {
				return fmt.Errorf("%w: error: %s. You can get a new API key at %s", herr, msg.Message, apiKeyURL)
			}
			return fmt.Errorf("%w: error: %s", herr, msg.Message)
		}
		return fmt.Errorf("error: %s", msg.Message)
	default:
		var herr *httpjson.Error
		if errors.As(err, &herr) {
			slog.WarnContext(ctx, "cloudflare", "url", url, "err", err, "response", string(herr.ResponseBody), "status", herr.StatusCode)
		} else {
			slog.WarnContext(ctx, "cloudflare", "url", url, "err", err)
		}
		return err
	}
}

const apiKeyURL = "https://dash.cloudflare.com/profile/api-tokens"

var _ genaiapi.CompletionProvider = &Client{}
