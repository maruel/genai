// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cloudflare implements a client for the Cloudflare AI API.
//
// It is described at https://developers.cloudflare.com/api/resources/ai/
package cloudflare

// See official client at https://github.com/cloudflare/cloudflare-go

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// https://developers.cloudflare.com/api/resources/ai/methods/run/
type ChatRequest struct {
	Messages          []Message `json:"messages"`
	FrequencyPenalty  float64   `json:"frequency_penalty,omitzero"` // [0, 2.0]
	MaxTokens         int64     `json:"max_tokens,omitzero"`
	PresencePenalty   float64   `json:"presence_penalty,omitzero"`   // [0, 2.0]
	RepetitionPenalty float64   `json:"repetition_penalty,omitzero"` // [0, 2.0]
	ResponseFormat    struct {
		Type       string             `json:"type,omitzero"` // json_object, json_schema
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Seed        int64   `json:"seed,omitzero"`
	Stream      bool    `json:"stream,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 5]
	Tools       []Tool  `json:"tools,omitzero"`
	TopK        int64   `json:"top_k,omitzero"` // [1, 50]
	TopP        float64 `json:"top_p,omitzero"` // [0, 2.0]

	// Functions         []function     `json:"functions,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Validatable, model string) error {
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.ChatOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.Seed = v.Seed
				c.TopK = v.TopK
				if len(v.Stop) != 0 {
					unsupported = append(unsupported, "Stop")
				}
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json_schema"
					c.ResponseFormat.JSONSchema = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					if v.ToolCallRequest != genai.ToolCallAny {
						// Cloudflare doesn't provide a way to force tool use.
						unsupported = append(unsupported, "ToolCallRequest")
					}
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						c.Tools[i].Function.Parameters = t.InputSchema()
					}
				}
			default:
				errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
			}
		}
	}

	if err := msgs.Validate(); err != nil {
		errs = append(errs, err)
	} else {
		offset := 0
		if sp != "" {
			offset = 1
		}
		c.Messages = make([]Message, len(msgs)+offset)
		if sp != "" {
			c.Messages[0].Role = "system"
			c.Messages[0].Content = sp
		}
		for i := range msgs {
			if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
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

// Message is not well specified in the API documentation.
// https://developers.cloudflare.com/api/resources/ai/methods/run/
type Message struct {
	Content string `json:"content"` // "system", "assistant", "user"
	Role    string `json:"role"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) > 1 {
		return errors.New("cloudflare doesn't support multiple content blocks; TODO split transparently")
	}
	if len(in.ToolCalls) != 0 {
		return errors.New("cloudflare tool calls are not supported yet")
	}
	if len(in.ToolCallResults) != 0 {
		return errors.New("cloudflare tool calls are not supported yet")
	}
	if in.Contents[0].Text != "" {
		m.Content = in.Contents[0].Text
	} else {
		return fmt.Errorf("unsupported content type %#v", in.Contents[0])
	}
	return nil
}

type Tool struct {
	Type     string `json:"type"` // "function"
	Function struct {
		Description string             `json:"description"`
		Name        string             `json:"name"`
		Parameters  *jsonschema.Schema `json:"parameters"`
	} `json:"function"`
}

/*
Maybe later

type function struct {
	Code string `json:"code"`
	Name string `json:"name"`
}

type prompt struct {
	Prompt            string  `json:"prompt"`
	FrequencyPenalty  float64 `json:"frequency_penalty,omitzero"` // [0, 2.0]
	Lora              string  `json:"lora,omitzero"`
	MaxTokens         int64   `json:"max_tokens,omitzero"`
	PresencePenalty   float64 `json:"presence_penalty,omitzero"`   // [0, 2.0]
	Raw               bool    `json:"raw,omitzero"`                // Do not apply chat template
	RepetitionPenalty float64 `json:"repetition_penalty,omitzero"` // [0, 2.0]
	ResponseFormat    struct {
		Type       string              `json:"type,omitzero"` // json_object, json_schema
		JSONSchema *jsonschema.Schema `json:"json_schema,omitzero"`
	} `json:"response_format,omitzero"`
	Seed        int64   `json:"seed,omitzero"`
	Stream      bool    `json:"stream,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 5]
	TopK        int64   `json:"top_k,omitzero"`       // [1, 50]
	TopP        float64 `json:"top_p,omitzero"`       // [0, 2.0]
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
*/

// https://developers.cloudflare.com/api/resources/ai/methods/run/
// See UnionMember7
type ChatResponse struct {
	Result struct {
		MessageResponse
		Usage Usage `json:"usage"`
	} `json:"result"`
	Success  bool       `json:"success"`
	Errors   []struct{} `json:"errors"`   // Annoyingly, it's included all the time
	Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
}

type MessageResponse struct {
	// Normally a string, or an object if response_format.type == "json_schema".
	Response  any        `json:"response"`
	ToolCalls []ToolCall `json:"tool_calls"`
}

func (msg *MessageResponse) To(schema string, out *genai.Message) error {
	out.Role = genai.Assistant
	if len(msg.ToolCalls) != 0 {
		// Starting 2025-03-17, "@hf/nousresearch/hermes-2-pro-mistral-7b" is finally returning structured tool call.
		out.ToolCalls = make([]genai.ToolCall, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			// Cloudflare doesn't support tool call IDs yet.
			id := fmt.Sprintf("tool_call%d", i)
			if err := tc.To(id, &out.ToolCalls[i]); err != nil {
				return err
			}
		}
		if msg.Response != nil {
			return fmt.Errorf("unexpected tool call and response %T: %v", msg.Response, msg.Response)
		}
		return nil
	}
	switch v := msg.Response.(type) {
	case string:
		// This is just sad.
		if strings.HasPrefix(v, "<tool_call>") {
			return fmt.Errorf("hacked up XML tool calls are not supported")
		} else {
			out.Contents = []genai.Content{{Text: v}}
		}
	default:
		if schema == "json_schema" {
			// Marshal back into JSON for now.
			b, err := json.Marshal(v)
			if err != nil {
				return fmt.Errorf("failed to JSON marshal type %T: %v: %w", v, v, err)
			}
			out.Contents = []genai.Content{{Text: string(b)}}
		} else {
			return fmt.Errorf("unexpected type %T: %v", v, v)
		}
	}
	return nil
}

func (c *ChatResponse) ToResult(rpcin *ChatRequest) (genai.ChatResult, error) {
	out := genai.ChatResult{
		// At the moment, Cloudflare doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Result.Usage.PromptTokens,
			OutputTokens: c.Result.Usage.CompletionTokens,
			// Cloudflare doesn't provide FinishReason.
		},
	}
	err := c.Result.To(rpcin.ResponseFormat.Type, &out.Message)
	return out, err
}

// If you find the documentation for this please tell me!
type ChatStreamChunkResponse struct {
	Response string `json:"response"`
	P        string `json:"p"`
	Usage    Usage  `json:"usage"`
}

type Usage struct {
	CompletionTokens int64 `json:"completion_tokens"`
	PromptTokens     int64 `json:"prompt_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

type ToolCall struct {
	Arguments any    `json:"arguments"`
	Name      string `json:"name"`
}

func (c *ToolCall) To(id string, out *genai.ToolCall) error {
	raw, err := json.Marshal(c.Arguments)
	if err != nil {
		return fmt.Errorf("failed to marshal tool call arguments: %w", err)
	}
	// Cloudflare doesn't support tool call IDs yet.
	out.ID = id
	out.Name = c.Name
	out.Arguments = string(raw)
	return nil
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

func (er *errorResponse) String() string {
	if len(er.Errors) == 0 {
		return fmt.Sprintf("error unknown (%#v)", er)
	}
	// Sometimes Code is set too.
	return fmt.Sprintf("error %s", er.Errors[0].Message)
}

// Client implements the REST JSON based API.
type Client struct {
	internal.ClientBase

	accountID string
	model     string
	chatURL   string
}

// New creates a new client to talk to the Cloudflare Workers AI platform API.
//
// If accountID is not provided, it tries to load it from the CLOUDFLARE_ACCOUNT_ID environment variable.
// If apiKey is not provided, it tries to load it from the CLOUDFLARE_API_KEY environment variable.
// If none is found, it returns an error.
// Get your account ID and API key at https://dash.cloudflare.com/profile/api-tokens
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the model from https://developers.cloudflare.com/workers-ai/models/
func New(accountID, apiKey, model string) (*Client, error) {
	const apiKeyURL = "https://dash.cloudflare.com/profile/api-tokens"
	if accountID == "" {
		if accountID = os.Getenv("CLOUDFLARE_ACCOUNT_ID"); accountID == "" {
			return nil, errors.New("cloudflare account ID is required; get one at " + apiKeyURL)
		}
	}
	if apiKey == "" {
		if apiKey = os.Getenv("CLOUDFLARE_API_KEY"); apiKey == "" {
			return nil, errors.New("cloudflare API key is required; get one at " + apiKeyURL)
		}
	}
	return &Client{
		accountID: accountID,
		model:     model,
		chatURL:   "https://api.cloudflare.com/client/v4/accounts/" + accountID + "/ai/run/" + model,
		ClientBase: internal.ClientBase{
			ClientJSON: httpjson.Client{
				Client: &http.Client{Transport: &roundtrippers.Header{
					Transport: &roundtrippers.Retry{
						Transport: &roundtrippers.RequestID{
							Transport: http.DefaultTransport,
						},
					},
					Header: http.Header{"Authorization": {"Bearer " + apiKey}},
				}},
				Lenient: internal.BeLenient,
			},
			APIKeyURL: apiKeyURL,
		},
	}, nil
}

func (c *Client) Chat(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.ChatResult, error) {
	// https://developers.cloudflare.com/api/resources/ai/methods/run/
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
	result, err := rpcout.ToResult(&rpcin)
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
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	return c.doRequest(ctx, "POST", c.chatURL, in, out)
}

func (c *Client) ChatStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) (genai.ChatResult, error) {
	result := genai.ChatResult{}
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return result, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
			}
		}
	}
	in := ChatRequest{}
	var continuableErr error
	if err := in.Init(msgs, opts, c.model); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return result, err
		}
	}
	ch := make(chan ChatStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks, &result)
	})
	err := c.ChatStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return result, continuableErr
	}
	return result, err
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.MessageFragment, result *genai.ChatResult) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if pkt.Usage.TotalTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
			// Cloudflare doesn't provide FinishReason.
		}
		// TODO: Tools.
		if word := pkt.Response; word != "" {
			f := genai.MessageFragment{TextFragment: word}
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	return nil
}

func (c *Client) ChatStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	// Investigate websockets?
	// https://blog.cloudflare.com/workers-ai-streaming/ and
	// https://developers.cloudflare.com/workers/examples/websockets/
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	resp, err := c.ClientJSON.Request(ctx, "POST", c.chatURL, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.DecodeError(ctx, c.chatURL, resp, &errorResponse{})
	}
	return processSSE(resp.Body, out)
}

func processSSE(body io.Reader, out chan<- ChatStreamChunkResponse) error {
	dataPrefix := []byte("data: ")
	eventPrefix := []byte("event:")
	done := []byte("[DONE]")
	for r := bufio.NewReader(body); ; {
		line, err := r.ReadBytes('\n')
		if line = bytes.TrimSpace(line); err == io.EOF {
			if len(line) == 0 {
				return nil
			}
		} else if err != nil {
			return fmt.Errorf("failed to get server response: %w", err)
		}
		if len(line) == 0 {
			continue
		}
		switch {
		case bytes.HasPrefix(line, dataPrefix):
			suffix := line[len(dataPrefix):]
			if bytes.Equal(suffix, done) {
				return nil
			}
			d := json.NewDecoder(bytes.NewReader(suffix))
			d.DisallowUnknownFields()
			d.UseNumber()
			msg := ChatStreamChunkResponse{}
			if err := d.Decode(&msg); err != nil {
				return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
			}
			out <- msg
		case bytes.HasPrefix(line, eventPrefix):
			// Ignore.
		default:
			// Error will be a single JSON line.
			d := json.NewDecoder(bytes.NewReader(line))
			d.DisallowUnknownFields()
			d.UseNumber()
			er := errorResponse{}
			if err := d.Decode(&er); err != nil || len(er.Errors) == 0 {
				return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
			}
			return errors.New(er.String())
		}
	}
}

// Time is a wrapper around time.Time to support unmarshalling for cloudflare non-standard encoding.
type Time time.Time

func (t *Time) UnmarshalJSON(b []byte) error {
	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	t2, err := time.Parse("2006-01-02 15:04:05.999999999", s)
	if err != nil {
		return err
	}
	*t = Time(t2)
	return nil
}

type Model struct {
	ID          string `json:"id"`
	Source      int64  `json:"source"`
	Name        string `json:"name"`
	Description string `json:"description"`
	CreatedAt   Time   `json:"created_at"`
	Task        struct {
		ID          string `json:"id"`
		Name        string `json:"name"`
		Description string `json:"description"`
	} `json:"task"`
	Tags       []string `json:"tags"`
	Properties []struct {
		PropertyID string `json:"property_id"`
		Value      any    `json:"value"` // sometimes a string, sometimes an array
	} `json:"properties"`
}

func (m *Model) GetID() string {
	return m.Name
}

func (m *Model) String() string {
	var suffixes []string
	for _, p := range m.Properties {
		if p.PropertyID == "info" || p.PropertyID == "terms" {
			continue
		}
		suffixes = append(suffixes, fmt.Sprintf("%s=%v", p.PropertyID, p.Value))
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
			if s, ok := p.Value.(string); ok {
				if v, err := strconv.ParseInt(s, 10, 64); err == nil {
					return v
				}
			}
		}
	}
	return 0
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://developers.cloudflare.com/api/resources/ai/subresources/models/methods/list/
	var models []genai.Model
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
		url := fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/models/search?page=%d&per_page=100&hide_experimental=false", c.accountID, page)
		err := c.doRequest(ctx, "GET", url, nil, &out)
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

func (c *Client) validate() error {
	if c.model == "" {
		return errors.New("a model is required")
	}
	return nil
}

func (c *Client) doRequest(ctx context.Context, method, url string, in, out any) error {
	return c.DoRequest(ctx, method, url, in, out, &errorResponse{})
}

var (
	_ genai.ChatProvider  = &Client{}
	_ genai.ModelProvider = &Client{}
)
