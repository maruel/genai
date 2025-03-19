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
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"golang.org/x/sync/errgroup"
)

// https://developers.cloudflare.com/api/resources/ai/methods/run/
type CompletionRequest struct {
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
func (c *CompletionRequest) Init(msgs genai.Messages, opts genai.Validatable) error {
	var errs []error
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			errs = append(errs, err)
		} else {
			switch v := opts.(type) {
			case *genai.CompletionOptions:
				c.MaxTokens = v.MaxTokens
				c.Temperature = v.Temperature
				c.TopP = v.TopP
				sp = v.SystemPrompt
				c.Seed = v.Seed
				c.TopK = v.TopK
				if len(v.Stop) != 0 {
					errs = append(errs, errors.New("cloudflare doesn't support stop tokens"))
				}
				if v.ReplyAsJSON {
					c.ResponseFormat.Type = "json_object"
				}
				if v.DecodeAs != nil {
					c.ResponseFormat.Type = "json_schema"
					c.ResponseFormat.JSONSchema = jsonschema.Reflect(v.DecodeAs)
				}
				if len(v.Tools) != 0 {
					// Cloudflare doesn't provide a way to force tool use.
					c.Tools = make([]Tool, len(v.Tools))
					for i, t := range v.Tools {
						c.Tools[i].Type = "function"
						c.Tools[i].Function.Name = t.Name
						c.Tools[i].Function.Description = t.Description
						if t.InputsAs != nil {
							c.Tools[i].Function.Parameters = jsonschema.Reflect(t.InputsAs)
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
	if len(in.Contents) != 1 {
		return errors.New("cloudflare doesn't support multiple content blocks; TODO split transparently")
	}
	if len(in.ToolCalls) != 0 {
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
	Raw               bool    `json:"raw,omitzero"`                // Do not aply chat template
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
type CompletionResponse struct {
	Result struct {
		MessageResponse
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
			/*
				out.Contents[0].Type = genai.ToolCalls
				// Example with "@hf/nousresearch/hermes-2-pro-mistral-7b" that is
				// supposed to support tool calling (and yes it supply the tool call XML
				// tags):
				// "<tool_call>\n{'arguments': {'country': 'Canada'}, 'name': 'best_country'}\n</tool_call>\n<tool_call>\n{'arguments': {'country': 'US'}, 'name': 'best_country'}\n</tool_call>"

				var toolCalls ToolCalls
				if err := xml.Unmarshal([]byte("<root>"+v+"</root>"), &toolCalls); err != nil {
					return out, fmt.Errorf("failed to unmarshal tool calls XML: %w; content: %q", err, v)
				}
				for i, tc := range toolCalls.Content {
					var toolCall ToolCall
					tc = strings.TrimSpace(tc)
					if err := json.Unmarshal([]byte(tc), &toolCall); err != nil {
						// This is ugly.
						tc2 := strings.ReplaceAll(tc, "'", "\"")
						if err := json.Unmarshal([]byte(tc2), &toolCall); err != nil {
							return out, fmt.Errorf("failed to unmarshal tool call as JSON: %w; content: %q", err, tc)
						}
					}
					raw, err := json.Marshal(toolCall.Arguments)
					if err != nil {
						return out, fmt.Errorf("failed to marshal tool call arguments: %w", err)
					}
					// Cloudflare doesn't support tool call IDs yet.
					out.Contents[0].ToolCalls = append(out.Contents[0].ToolCalls, genai.ToolCall{ID: fmt.Sprintf("tool_call%d", i), Name: toolCall.Name, Arguments: string(raw)})
				}
			*/
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

func (c *CompletionResponse) ToResult(rpcin *CompletionRequest) (genai.CompletionResult, error) {
	out := genai.CompletionResult{
		Usage: genai.Usage{
			InputTokens:  c.Result.Usage.PromptTokens,
			OutputTokens: c.Result.Usage.CompletionTokens,
		},
	}
	err := c.Result.To(rpcin.ResponseFormat.Type, &out.Message)
	return out, err
}

// If you find the documentation for this please tell me!
type CompletionStreamChunkResponse struct {
	Response string `json:"response"`
	P        string `json:"p"`
	Usage    struct {
		CompletionTokens int64 `json:"completion_tokens"`
		PromptTokens     int64 `json:"prompt_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	} `json:"usage"`
}

/*
type toolCalls struct {
	XMLName xml.Name `xml:"root"`
	Content []string `xml:"tool_call"`
}
*/

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

// Client implements the REST JSON based API.
type Client struct {
	accountID string
	model     string
	c         httpjson.Client
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
		c: httpjson.Client{
			Client: &http.Client{
				Transport: &internal.TransportHeaders{
					R: internal.DefaultTransport,
					H: map[string]string{
						"Authorization": "Bearer " + apiKey,
						// See https://github.com/cloudflare/cloudflare-go/blob/main/internal/requestconfig/requestconfig.go
						"X-Stainless-Retry-Count": "0",
						"X-Stainless-Timeout":     "0",
					},
				},
			},
			// Cloudflare doesn't support HTTP POST compression.
			PostCompress: "",
		},
	}, nil
}

func (c *Client) Completion(ctx context.Context, msgs genai.Messages, opts genai.Validatable) (genai.CompletionResult, error) {
	// https://developers.cloudflare.com/api/resources/ai/methods/run/
	rpcin := CompletionRequest{}
	if err := rpcin.Init(msgs, opts); err != nil {
		return genai.CompletionResult{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.CompletionResult{}, fmt.Errorf("failed to get chat response: %w", err)
	}
	return rpcout.ToResult(&rpcin)
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = false
	url := "https://api.cloudflare.com/client/v4/accounts/" + c.accountID + "/ai/run/" + c.model
	return c.post(ctx, url, in, out)
}

func (c *Client) CompletionStream(ctx context.Context, msgs genai.Messages, opts genai.Validatable, chunks chan<- genai.MessageFragment) error {
	in := CompletionRequest{}
	if err := in.Init(msgs, opts); err != nil {
		return err
	}
	ch := make(chan CompletionStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processStreamPackets(ch, chunks)
	})
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	return err
}

func processStreamPackets(ch <-chan CompletionStreamChunkResponse, chunks chan<- genai.MessageFragment) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if word := pkt.Response; word != "" {
			chunks <- genai.MessageFragment{TextFragment: word}
		}
	}
	return nil
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	// Investigate websockets?
	// https://blog.cloudflare.com/workers-ai-streaming/ and
	// https://developers.cloudflare.com/workers/examples/websockets/
	if err := c.validate(); err != nil {
		return err
	}
	in.Stream = true
	url := "https://api.cloudflare.com/client/v4/accounts/" + c.accountID + "/ai/run/" + c.model
	resp, err := c.c.PostRequest(ctx, url, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get server response: %w", err)
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

func parseStreamLine(line []byte, out chan<- CompletionStreamChunkResponse) error {
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
		if err := d.Decode(&msg); err != nil {
			return fmt.Errorf("failed to decode server response %q: %w", string(line), err)
		}
		out <- msg
	default:
		er := errorResponse{}
		d := json.NewDecoder(bytes.NewReader(line))
		d.DisallowUnknownFields()
		d.UseNumber()
		if err := d.Decode(&er); err != nil || len(er.Errors) == 0 {
			return fmt.Errorf("unexpected line. expected \"data: \", got %q", line)
		}
		return fmt.Errorf("got server error: %s", er.Errors[0].Message)
	}
	return nil
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
		err := c.c.Get(ctx, url, nil, &out)
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

func (c *Client) post(ctx context.Context, url string, in, out any) error {
	resp, err := c.c.PostRequest(ctx, url, nil, in)
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

var (
	_ genai.CompletionProvider = &Client{}
	_ genai.ModelProvider      = &Client{}
)
