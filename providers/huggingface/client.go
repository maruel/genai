// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package huggingface implements a client for the HuggingFace serverless
// inference API.
//
// It is described at https://huggingface.co/docs/api-inference/
package huggingface

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Official python SDK: https://github.com/huggingface/huggingface_hub
//
// But the real spec source of truth is
// https://github.com/huggingface/huggingface.js/tree/main/packages/tasks/src/tasks/chat-completion

// Scoreboard for Huggingface.
//
// # Warnings
//
//   - Huggingface supports a ton of models on its serverless inference platform, more than any other
//     provider.
//   - Huggingface is also a router to other backends.
//   - Huggingface as a platform is generally unstable, with error not being properly reported. Use with care.
//   - It supports way more options than the client currently implements.
//   - Tool calling works very well but is biased; the model is lazy and when it's unsure, it will use the
//     tool's first argument.
var Scoreboard = genai.Scoreboard{
	Country:      "US",
	DashboardURL: "https://huggingface.co/settings/billing",
	// TODO: Huggingface obviously supports more modalities.
	Scenarios: []genai.Scenario{
		{
			Models: []string{"meta-llama/Llama-3.3-70B-Instruct"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Tools:      genai.True,
				BiasedTool: genai.True,
				JSONSchema: true,
				Seed:       true,
			},
			GenStream: &genai.FunctionalityText{
				BrokenFinishReason: true,
				Tools:              genai.True,
				BiasedTool:         genai.True,
				JSONSchema:         true,
				Seed:               true,
			},
		},
		{
			Models: []string{"Qwen/QwQ-32B"},
			In:     map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out:    map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &genai.FunctionalityText{
				Thinking:   true,
				Tools:      genai.True,
				BiasedTool: genai.True,
				JSONSchema: true,
				Seed:       true,
			},
			GenStream: &genai.FunctionalityText{
				Thinking:           true,
				BrokenFinishReason: true,
				Tools:              genai.True,
				BiasedTool:         genai.True,
				JSONSchema:         true,
				Seed:               true,
			},
		},
	},
}

// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type ChatRequest struct {
	Model            string    `json:"model,omitempty"` // It's already in the URL.
	Stream           bool      `json:"stream"`
	Messages         []Message `json:"messages"`
	FrequencyPenalty float64   `json:"frequency_penalty,omitzero"` // [-2.0, 2.0]
	Logprobs         bool      `json:"logprobs,omitzero"`
	MaxTokens        int64     `json:"max_tokens,omitzero"`
	PresencePenalty  float64   `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	ResponseFormat   struct {
		Type string `json:"type"` // "json", "regex"
		// Type == "regexp": a regex string.
		// Type == "json": a JSONSchema.
		Value *jsonschema.Schema `json:"value"`
	} `json:"response_format,omitzero"`
	Seed          int64    `json:"seed,omitzero"`
	Stop          []string `json:"stop,omitzero"` // Up to 4
	StreamOptions struct {
		IncludeUsage bool `json:"include_usage,omitzero"`
	} `json:"stream_options,omitzero"`
	Temperature float64 `json:"temperature,omitzero"` // [0, 2.0]
	// Alternative when forcing a specific function. This can probably be achieved
	// by providing a single tool and ToolChoice == "required".
	// ToolChoice struct {
	// 	Type     string `json:"type,omitzero"` // "function"
	// 	Function struct {
	// 		Name string `json:"name,omitzero"`
	// 	} `json:"function,omitzero"`
	// } `json:"tool_choice,omitzero"`
	ToolChoice  string  `json:"tool_choice,omitzero"` // "auto", "none", "required"
	ToolPrompt  string  `json:"tool_prompt,omitzero"`
	Tools       []Tool  `json:"tools,omitzero"`
	TopLogprobs int64   `json:"top_logprobs,omitzero"`
	TopP        float64 `json:"top_p,omitzero"` // [0, 1]
	// logit_bias, n are documented as ignored.
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsText:
			c.MaxTokens = v.MaxTokens
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			sp = v.SystemPrompt
			c.Seed = v.Seed
			if v.TopK != 0 {
				unsupported = append(unsupported, "TopK")
			}
			c.Stop = v.Stop
			if v.DecodeAs != nil {
				c.ResponseFormat.Type = "json"
				c.ResponseFormat.Value = jsonschema.Reflect(v.DecodeAs)
			} else if v.ReplyAsJSON {
				c.ResponseFormat.Type = "json"
			}
			if len(v.Tools) != 0 {
				switch v.ToolCallRequest {
				case genai.ToolCallAny:
					c.ToolChoice = "auto"
				case genai.ToolCallRequired:
					c.ToolChoice = "required"
				case genai.ToolCallNone:
					c.ToolChoice = "none"
				}
				c.Tools = make([]Tool, len(v.Tools))
				for i, t := range v.Tools {
					c.Tools[i].Type = "function"
					c.Tools[i].Function.Name = t.Name
					c.Tools[i].Function.Description = t.Description
					if c.Tools[i].Function.Arguments = t.InputSchemaOverride; c.Tools[i].Function.Arguments == nil {
						c.Tools[i].Function.Arguments = t.GetInputSchema()
					}
				}
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}

	offset := 0
	if sp != "" {
		offset = 1
	}
	c.Messages = make([]Message, len(msgs)+offset)
	if sp != "" {
		c.Messages[0].Role = "system"
		c.Messages[0].Content = []Content{{Type: ContentText, Text: sp}}
	}
	for i := range msgs {
		if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
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
	c.StreamOptions.IncludeUsage = stream
}

// Message is incorrectly documented at
// https://huggingface.co/docs/api-inference/tasks/chat-completion#api-specification
type Message struct {
	Role      string     `json:"role"` // "system", "assistant", "user", "tool"
	Content   Contents   `json:"content,omitzero"`
	ToolCalls []ToolCall `json:"tool_calls,omitzero"`
	Name      string     `json:"name,omitzero"` // Not really documented
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) != 0 {
		m.Content = make([]Content, len(in.Contents))
		for i := range in.Contents {
			if err := m.Content[i].From(&in.Contents[i]); err != nil {
				return fmt.Errorf("block %d: %w", i, err)
			}
		}
	}
	if len(in.ToolCalls) != 0 {
		m.ToolCalls = make([]ToolCall, len(in.ToolCalls))
		for j := range in.ToolCalls {
			m.ToolCalls[j].From(&in.ToolCalls[j])
		}
	}
	if len(in.ToolCallResults) != 0 {
		if len(in.Contents) != 0 || len(in.ToolCalls) != 0 {
			// This could be worked around.
			return fmt.Errorf("can't have tool call result along content or tool calls")
		}
		// Huggingface doesn't use tool ID in the result, hence only one tool can safely be called at a time.
		if len(in.ToolCallResults) != 1 {
			return fmt.Errorf("can't have more than one tool call result at a time")
		}
		m.Role = "tool"
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.Name = in.ToolCallResults[0].Name

	}
	return nil
}

type Content struct {
	Type     ContentType `json:"type"`
	Text     string      `json:"text,omitzero"`
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`
}

func (c *Content) From(in *genai.Content) error {
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	mimeType, data, err := in.ReadDocument(10 * 1024 * 1024)
	if err != nil {
		return err
	}
	switch {
	case (in.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
		c.Type = ContentImageURL
		if in.URL == "" {
			c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		} else {
			c.ImageURL.URL = in.URL
		}
	case strings.HasPrefix(mimeType, "text/plain"):
		c.Type = ContentText
		if in.URL != "" {
			return errors.New("text/plain documents must be provided inline, not as a URL")
		}
		c.Text = string(data)
	default:
		return fmt.Errorf("unsupported mime type %s", mimeType)
	}
	return nil
}

// Contents represents a slice of Content with custom unmarshalling to handle
// both string and Content struct types.
type Contents []Content

func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal(([]Content)(*c))
}

type ContentType string

const (
	ContentText     ContentType = "text"
	ContentImageURL ContentType = "image_url"
)

type ToolCall struct {
	Index    int64  `json:"index,omitzero"`
	ID       string `json:"id,omitzero"`
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string   `json:"name,omitzero"`
		Description struct{} `json:"description,omitzero"` // Passed in as null in response
		Arguments   string   `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

func (t *ToolCall) From(in *genai.ToolCall) {
	t.ID = in.ID
	t.Type = "function"
	t.Function.Name = in.Name
	// The API seems to flip-flop between JSON and string.
	// return json.Unmarshal([]byte(in.Arguments), &t.Function.Arguments)
	t.Function.Arguments = in.Arguments
}

func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	// b, err := json.Marshal(t.Function.Arguments)
	// if err != nil {
	//	return fmt.Errorf("failed to marshal arguments: %w", err)
	// }
	// out.Arguments = string(b)
	out.Arguments = t.Function.Arguments
}

type Tool struct {
	Type     string `json:"type,omitzero"` // "function"
	Function struct {
		Name        string             `json:"name,omitzero"`
		Description string             `json:"description,omitzero"`
		Arguments   *jsonschema.Schema `json:"arguments,omitzero"`
	} `json:"function,omitzero"`
}

type ChatResponse struct {
	Object            string    `json:"object"`
	ID                string    `json:"id"`
	Created           base.Time `json:"created"`
	Model             string    `json:"model"`
	SystemFingerprint string    `json:"system_fingerprint"`

	Choices []struct {
		FinishReason FinishReason    `json:"finish_reason"`
		Index        int64           `json:"index"`
		Message      MessageResponse `json:"message"`
		Logprobs     struct {
			Content []struct {
				Logprob     float64 `json:"logprob"`
				Token       string  `json:"token"`
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

type FinishReason string

const (
	FinishStop         FinishReason = "stop"
	FinishLength       FinishReason = "length"
	FinishStopSequence FinishReason = "stop_sequence"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishLength:
		return genai.FinishedLength
	case FinishStopSequence:
		return genai.FinishedStopSequence
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens"`
	CompletionTokens int64 `json:"completion_tokens"`
	TotalTokens      int64 `json:"total_tokens"`
}

// The structure is different than the request. :(
type MessageResponse struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCallID string     `json:"tool_call_id"`
	ToolCalls  []ToolCall `json:"tool_calls"`
}

func (m *MessageResponse) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant", "user":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	if len(m.ToolCalls) != 0 {
		out.ToolCalls = make([]genai.ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			m.ToolCalls[i].To(&out.ToolCalls[i])
		}
	}

	if m.Content != "" {
		out.Contents = []genai.Content{{Text: m.Content}}
	}
	return nil
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Huggingface doesn't support caching.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))
	}
	out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	if len(out.ToolCalls) != 0 && out.FinishReason == genai.FinishedStop {
		// Lie for the benefit of everyone.
		out.FinishReason = genai.FinishedToolCalls
	}
	return out, err
}

type ChatStreamChunkResponse struct {
	Object            string    `json:"object"`
	Created           base.Time `json:"created"`
	ID                string    `json:"id"`
	Model             string    `json:"model"`
	SystemFingerprint string    `json:"system_fingerprint"`
	Choices           []struct {
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Delta        struct {
			Role    genai.Role `json:"role"`
			Content string     `json:"content"`
			// ToolCallID string     `json:"tool_call_id"`
			ToolCalls []ToolCall `json:"tool_calls"`
		} `json:"delta"`
		Logprobs struct {
			Content []struct {
				Logprob     float64 `json:"logprob"`
				Token       string  `json:"token"`
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
	Usage Usage `json:"usage"`
}

type Model struct {
	ID            string    `json:"id"`
	ID2           string    `json:"_id"`
	Likes         int64     `json:"likes"`
	TrendingScore float64   `json:"trendingScore"`
	Private       bool      `json:"private"`
	Downloads     int64     `json:"downloads"`
	Tags          []string  `json:"tags"` // Tags can be a single word or key:value, like base_model, doi, license, region, arxiv.
	PipelineTag   string    `json:"pipeline_tag"`
	LibraryName   string    `json:"library_name"`
	CreatedAt     time.Time `json:"createdAt"`
	ModelId       string    `json:"modelId"`

	// When full=true is specified:
	Author       string    `json:"author"`
	Gated        bool      `json:"gated"`
	LastModified time.Time `json:"lastModified"`
	SHA          string    `json:"sha"`
	Siblings     []struct {
		RFilename string `json:"r_filename"`
	} `json:"siblings"`
}

func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s) %s Trending: %.1f", m.ID, m.CreatedAt.Format("2006-01-02"), m.PipelineTag, m.TrendingScore)
}

func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for Huggingface models listing
type ModelsResponse []Model

// ToModels converts Huggingface models to genai.Model interfaces
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(*r))
	for i := range *r {
		models[i] = &(*r)[i]
	}
	return models
}

//

type ErrorResponse struct {
	Error     ErrorError `json:"error"`
	ErrorType string     `json:"error_type"`
}

func (er *ErrorResponse) String() string {
	if er.Error.HTTPStatusCode != 0 {
		return fmt.Sprintf("http %d: %s", er.Error.HTTPStatusCode, er.Error.Message)
	}
	return "error " + er.Error.Message
}

type ErrorError struct {
	Message        string `json:"message"`
	HTTPStatusCode int64  `json:"http_status_code"`
}

func (ee *ErrorError) UnmarshalJSON(d []byte) error {
	s := ""
	if err := json.Unmarshal(d, &s); err == nil {
		ee.Message = s
		return nil
	}
	var x struct {
		Message        string `json:"message"`
		HTTPStatusCode int64  `json:"http_status_code"`
	}
	if err := json.Unmarshal(d, &x); err != nil {
		return err
	}
	ee.Message = x.Message
	ee.HTTPStatusCode = x.HTTPStatusCode
	return nil
}

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// TODO: Investigate https://huggingface.co/blog/inference-providers and https://huggingface.co/docs/inference-endpoints/

// New creates a new client to talk to the HuggingFace serverless inference API.
//
// If apiKey is not provided, it tries to load it from the HUGGINGFACE_API_KEY environment variable.
// Otherwise, it tries to load it from the huggingface python client's cache.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://huggingface.co/settings/tokens
//
// If no model is provided, only functions that do not require a model, like ListModels, will work.
// To use multiple models, create multiple clients.
// Use one of the tens of thousands of models to chose from at https://huggingface.co/models?inference=warm&sort=trending
//
// Pass model base.PreferredCheap to use a good cheap model, base.PreferredGood for a good model or
// base.PreferredSOTA to use its SOTA model. Keep in mind that as providers cycle through new models, it's
// possible the model is not available anymore.
//
// wrapper can be used to throttle outgoing requests, record calls, etc. It defaults to base.DefaultTransport.
// wrapper can also be used to add the HTTP header "X-HF-Bill-To" via roundtrippers.Header. See
// https://huggingface.co/docs/inference-providers/pricing#organization-billing
func New(apiKey, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://huggingface.co/settings/tokens"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("HUGGINGFACE_API_KEY"); apiKey == "" {
			// Fallback to loading from the python client's cache.
			h, errHome := os.UserHomeDir()
			if errHome != nil {
				err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
			} else {
				// TODO: Windows.
				b, errRead := os.ReadFile(filepath.Join(h, ".cache", "huggingface", "token"))
				if errRead != nil {
					err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
				} else {
					if apiKey = strings.TrimSpace(string(b)); apiKey == "" {
						err = &base.ErrAPIKeyRequired{EnvVar: "HUGGINGFACE_API_KEY", URL: apiKeyURL}
					}
				}
			}
		}
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://router.huggingface.co/hf-inference/models/" + url.PathEscape(model) + "/v1/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "huggingface",
				APIKeyURL:    apiKeyURL,
				ClientJSON: httpjson.Client{
					Lenient: internal.BeLenient,
					Client: &http.Client{
						Transport: &roundtrippers.Header{
							Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
							Transport: &roundtrippers.RequestID{Transport: t},
						},
					},
				},
			},
		},
	}
	if err == nil && (model == base.PreferredCheap || model == base.PreferredGood || model == base.PreferredSOTA) {
		// Warning: listing models from Huggingface takes a while.
		mdls, err2 := c.ListModels(context.Background())
		if err2 != nil {
			return nil, err2
		}
		cheap := model == base.PreferredCheap
		good := model == base.PreferredGood
		c.Model = ""
		c.GenSyncURL = ""
		trending := 0.
		weights := 0
		re := regexp.MustCompile(`(\d+)B`)
		for _, mdl := range mdls {
			m := mdl.(*Model)
			if m.TrendingScore < 2 {
				continue
			}
			if cheap {
				// HF doesn't report the number of weights in the model. Try to guess it.
				matches := re.FindAllStringSubmatch(m.ID, 1)
				if len(matches) != 1 {
					continue
				}
				w, err2 := strconv.Atoi(matches[0][1])
				if err2 != nil {
					continue
				}
				if strings.HasPrefix(m.ID, "meta-llama/Llama") && (weights == 0 || w < weights) {
					weights = w
					c.Model = m.ID
					c.GenSyncURL = "https://router.huggingface.co/hf-inference/models/" + url.PathEscape(c.Model) + "/v1/chat/completions"
				}
			} else if good {
				// HF doesn't report the number of weights in the model. Try to guess it.
				matches := re.FindAllStringSubmatch(m.ID, 1)
				if len(matches) != 1 {
					continue
				}
				w, err2 := strconv.Atoi(matches[0][1])
				if err2 != nil {
					continue
				}
				if strings.HasPrefix(m.ID, "Qwen/Qwen") && (weights == 0 || w > weights) {
					weights = w
					c.Model = m.ID
					c.GenSyncURL = "https://router.huggingface.co/hf-inference/models/" + url.PathEscape(c.Model) + "/v1/chat/completions"
				}
			} else {
				if strings.HasPrefix(m.ID, "deepseek-ai/") && !strings.Contains(m.ID, "Qwen") && !strings.Contains(m.ID, "Prover") && !strings.Contains(m.ID, "Distill") && (trending == 0 || trending < m.TrendingScore) {
					// Make it a popularity contest.
					trending = m.TrendingScore
					c.Model = m.ID
					c.GenSyncURL = "https://router.huggingface.co/hf-inference/models/" + url.PathEscape(c.Model) + "/v1/chat/completions"
				}
			}
		}
		if c.Model == "" {
			return nil, errors.New("failed to find a model automatically")
		}
	}
	return c, err
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://huggingface.co/docs/hub/api
	// There's 20k models warm as of March 2025. There's no way to sort by
	// trending. Sorting by download is not useful. There's no pagination.
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, "https://huggingface.co/api/models?inference=warm")
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	pendingCall := ToolCall{}
	for pkt := range ch {
		if pkt.Usage.PromptTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
		}
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Choices[0].FinishReason != "" {
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "assistant", "":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		// There's only one at a time ever.
		if len(pkt.Choices[0].Delta.ToolCalls) > 1 {
			return fmt.Errorf("implement multiple tool calls: %#v", pkt.Choices[0].Delta.ToolCalls)
		}
		f := genai.ContentFragment{TextFragment: pkt.Choices[0].Delta.Content}
		// Huggingface streams the arguments. Buffer the arguments to send the fragment as a whole tool call.
		if len(pkt.Choices[0].Delta.ToolCalls) == 1 {
			if t := pkt.Choices[0].Delta.ToolCalls[0]; t.ID == pendingCall.ID {
				// Continuation.
				pendingCall.Function.Arguments += t.Function.Arguments
				if !f.IsZero() {
					return fmt.Errorf("implement tool call with metadata: %#v", pkt)
				}
				continue
			} else {
				// A new call.
				if pendingCall.ID == "" {
					pendingCall = t
					if !f.IsZero() {
						return fmt.Errorf("implement tool call with metadata: %#v", pkt)
					}
					continue
				}
				// Flush.
				pendingCall.To(&f.ToolCall)
				pendingCall = t
			}
		} else if pendingCall.ID != "" {
			// Flush.
			pendingCall.To(&f.ToolCall)
			pendingCall = ToolCall{}
		}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	// Hugginface doesn't send an "ending" packet, FinishReason isn't even set on the last packet.
	if pendingCall.ID != "" {
		// Flush.
		f := genai.ContentFragment{}
		pendingCall.To(&f.ToolCall)
		if err := result.Accumulate(f); err != nil {
			return err
		}
		chunks <- f
	}
	return nil
}

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderModel      = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
