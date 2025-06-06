// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cloudflare implements a client for the Cloudflare AI API.
//
// It is described at https://developers.cloudflare.com/api/resources/ai/
package cloudflare

// See official client at https://github.com/cloudflare/cloudflare-go

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/provider"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Cloudflare.
//
// # Warnings
//
//   - FinishReason is not returned.
//   - StopSequence doesn't work.
//   - Usage tokens isn't reported when streaming or using JSON.
//   - ChatRequest format is model dependent.
//   - Tool calling is supported on some models but it's flaky.
//
// Given the fact that FinishReason, StopSequence and Usage are broken, I can't recommend this provider beside
// toys.
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"@cf/meta/llama-4-scout-17b-16e-instruct",
				"@cf/meta/llama-3.2-3b-instruct",
			},
			GenSync: genai.FunctionalityText{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				BrokenFinishReason: true,
				NoStopSequence:     true,
				Tools:              genai.Flaky,
				UnbiasedTool:       false,
				JSON:               true,
				JSONSchema:         true,
			},
			GenStream: genai.FunctionalityText{
				Inline:             true,
				URL:                false,
				Thinking:           false,
				BrokenFinishReason: true,
				NoStopSequence:     true,
				Tools:              genai.Flaky,
				UnbiasedTool:       false,
				JSON:               true,
				JSONSchema:         true,
			},
		},
	},
}

// ChatRequest structure depends on the model used.
//
// The general description is at https://developers.cloudflare.com/api/resources/ai/methods/run/
//
// A specific one is https://developers.cloudflare.com/workers-ai/models/llama-4-scout-17b-16e-instruct/
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
	GuidedJSON  *jsonschema.Schema `json:"guided_json,omitzero"`
	Seed        int64              `json:"seed,omitzero"`
	Stream      bool               `json:"stream,omitzero"`
	Temperature float64            `json:"temperature,omitzero"` // [0, 5]
	Tools       []Tool             `json:"tools,omitzero"`
	TopK        int64              `json:"top_k,omitzero"` // [1, 50]
	TopP        float64            `json:"top_p,omitzero"` // [0, 2.0]
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.OptionsText:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.Seed = v.Seed
				c.TopK = v.TopK
				if len(v.Stop) != 0 {
					errs = append(errs, errors.New("unsupported option Stop"))
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json_schema"
					c.ResponseFormat.JSONSchema = jsonschema.Reflect(v.DecodeAs)
				} else if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if len(v.Tools) != 0 {
					if v.ToolCallRequest != genai.ToolCallAny {
						// Cloudflare doesn't provide a way to force tool use. Don't fail.
						unsupported = append(unsupported, "ToolCallRequest")
					}
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						if c.Tools[i].Function.Parameters = t.InputSchemaOverride; c.Tools[i].Function.Parameters == nil {
							c.Tools[i].Function.Parameters = t.GetInputSchema()
						}
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

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

// Message is not well specified in the API documentation.
// https://developers.cloudflare.com/api/resources/ai/methods/run/
type Message struct {
	Role       string `json:"role"` // "system", "assistant", "user", "tool"
	Content    string `json:"content,omitzero"`
	ToolCallID string `json:"tool_call_id,omitzero"`
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
		if len(in.ToolCalls) > 1 {
			return errors.New("cloudflare doesn't support multiple tool replies in a single message yet")
		}
		if len(in.Contents) != 0 {
			return errors.New("cloudflare can't have both tool calls and contents in one message")
		}
		if len(in.ToolCallResults) != 0 {
			return errors.New("cloudflare can't have both tool calls and tool call results in one message")
		}
		m.ToolCallID = in.ToolCalls[0].ID
		m.Content = in.ToolCalls[0].Arguments
		return nil
	}
	if len(in.ToolCallResults) != 0 {
		if len(in.Contents) != 0 || len(in.ToolCalls) != 0 {
			// This could be worked around.
			return fmt.Errorf("can't have tool call result along content or tool calls")
		}
		if len(in.ToolCallResults) != 1 {
			// This could be worked around.
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
		m.Role = "tool"
		m.ToolCallID = in.ToolCallResults[0].ID
		m.Content = in.ToolCallResults[0].Result
		return nil
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

func (msg *MessageResponse) To(out *genai.Message) error {
	out.Role = genai.Assistant
	if len(msg.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(msg.ToolCalls))
		for i, tc := range msg.ToolCalls {
			if err := tc.To(&out.ToolCalls[i]); err != nil {
				return err
			}
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
		// Marshal back into JSON.
		b, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("failed to JSON marshal type %T: %v: %w", v, v, err)
		}
		out.Contents = []genai.Content{{Text: string(b)}}
	}
	return nil
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Cloudflare doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Result.Usage.PromptTokens,
			OutputTokens: c.Result.Usage.CompletionTokens,
			// Cloudflare doesn't provide FinishReason (!?)
		},
	}
	err := c.Result.To(&out.Message)
	return out, err
}

// If you find the documentation for this please tell me!
type ChatStreamChunkResponse struct {
	Response  string     `json:"response"`
	P         string     `json:"p"`
	ToolCalls []ToolCall `json:"tool_calls"`
	Usage     Usage      `json:"usage"`
}

type Usage struct {
	CompletionTokens int64 `json:"completion_tokens"`
	PromptTokens     int64 `json:"prompt_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// ToolCall can be populated differently depending on the model used.
type ToolCall struct {
	Type     string `json:"type,omitzero"` // "function"
	ID       string `json:"id,omitzero"`
	Index    int64  `json:"index,omitzero"`
	Function struct {
		Name      string `json:"name,omitzero"`
		Arguments string `json:"arguments"`
	} `json:"function,omitzero"`

	Arguments any    `json:"arguments"`
	Name      string `json:"name"`
}

func (c *ToolCall) To(out *genai.ToolCall) error {
	out.ID = c.ID
	if out.Name = c.Name; c.Name == "" {
		out.Name = c.Function.Name
	}
	if c.Function.Arguments != "" {
		out.Arguments = c.Function.Arguments
	} else {
		raw, err := json.Marshal(c.Arguments)
		if err != nil {
			return fmt.Errorf("failed to marshal tool call arguments: %w", err)
		}
		out.Arguments = string(raw)
	}
	return nil
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

type ModelPricing struct {
	Currency string  `json:"currency"`
	Price    float64 `json:"price"`
	Unit     string  `json:"unit"` // "per M input tokens", "per M output tokens"
}

func (m *Model) String() string {
	var suffixes []string
	pp := slices.Clone(m.Properties)
	sort.Slice(pp, func(i, j int) bool {
		return pp[i].PropertyID < pp[j].PropertyID
	})
	for _, p := range pp {
		if p.PropertyID == "info" || p.PropertyID == "terms" {
			continue
		}
		switch p.Value.(type) {
		case json.Number, string, int:
			suffixes = append(suffixes, fmt.Sprintf("%s=%v", p.PropertyID, p.Value))
		default:
			b, _ := json.Marshal(p.Value)
			d := json.NewDecoder(bytes.NewReader(b))
			d.DisallowUnknownFields()
			var mp []ModelPricing
			if err := d.Decode(&mp); err == nil {
				var s []string
				for _, l := range mp {
					s = append(s, fmt.Sprintf("%g$%s %s", l.Price, l.Currency, l.Unit))
				}
				suffixes = append(suffixes, fmt.Sprintf("%s=[%s]", p.PropertyID, strings.Join(s, ", ")))
			} else {
				suffixes = append(suffixes, fmt.Sprintf("%s=%v", p.PropertyID, p.Value))
			}
		}
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

// ModelsResponse represents the response structure for Cloudflare models listing
type ModelsResponse struct {
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

//

type ErrorResponse struct {
	Errors []struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"errors"`
	Success  bool       `json:"success"`
	Result   struct{}   `json:"result"`
	Messages []struct{} `json:"messages"` // Annoyingly, it's included all the time
}

func (er *ErrorResponse) String() string {
	if len(er.Errors) == 0 {
		return fmt.Sprintf("error unknown (%#v)", er)
	}
	// Sometimes Code is set too.
	return fmt.Sprintf("error %s", er.Errors[0].Message)
}

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	provider.BaseGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]

	accountID string
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
//
// r can be used to throttle outgoing requests, record calls, etc. It defaults to http.DefaultTransport.
func New(accountID, apiKey, model string, r http.RoundTripper) (*Client, error) {
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
	if r == nil {
		r = http.DefaultTransport
	}
	// Investigate websockets?
	// https://blog.cloudflare.com/workers-ai-streaming/ and
	// https://developers.cloudflare.com/workers/examples/websockets/
	return &Client{
		BaseGen: provider.BaseGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.cloudflare.com/client/v4/accounts/" + accountID + "/ai/run/" + model,
			ProcessStreamPackets: processStreamPackets,
			Base: provider.Base[*ErrorResponse]{
				ProviderName: "cloudflare",
				APIKeyURL:    apiKeyURL,
				ClientJSON: httpjson.Client{
					Lenient: internal.BeLenient,
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
							Transport: &roundtrippers.Retry{Transport: &roundtrippers.RequestID{Transport: r}},
						},
					},
				},
			},
		},
		accountID: accountID,
	}, nil
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://developers.cloudflare.com/api/resources/ai/subresources/models/methods/list/
	var models []genai.Model
	for page := 1; ; page++ {
		out := ModelsResponse{}
		// Cloudflare's pagination is surprisingly brittle.
		url := fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/models/search?page=%d&per_page=100&hide_experimental=false", c.accountID, page)
		err := c.DoRequest(ctx, "GET", url, nil, &out)
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

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
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
			f := genai.ContentFragment{TextFragment: word}
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	return nil
}

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
