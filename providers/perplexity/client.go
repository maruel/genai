// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package perplexity implements a client for the Perplexity API.
//
// It is described at https://docs.perplexity.ai/api-reference
package perplexity

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Perplexity.
//
// # Warnings
//
//   - Thinking is not returned.
//   - Websearch, which is automatic for all models except r1-1776, is very expensive.
//   - Perplexity supports more than what the client supports.
//   - No tool calling support.
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"r1-1776",
				"sonar-deep-research",
			},
			GenSync: &genai.FunctionalityText{
				Thinking:       true,
				NoStopSequence: true,
				JSONSchema:     true,
			},
			GenStream: &genai.FunctionalityText{
				Thinking:       true,
				NoStopSequence: true,
				JSONSchema:     true,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"sonar",
				"sonar-pro",
			},
			GenSync: &genai.FunctionalityText{
				NoStopSequence: true,
				JSONSchema:     true,
				Citations:      true,
			},
			GenStream: &genai.FunctionalityText{
				NoStopSequence: true,
				Citations:      true,
			},
		},
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityText},
			Models: []string{
				"sonar-reasoning",
				"sonar-reasoning-pro",
			},
			GenSync: &genai.FunctionalityText{
				Thinking:       true,
				NoStopSequence: true,
				JSONSchema:     true,
				Citations:      true,
			},
			GenStream: &genai.FunctionalityText{
				Thinking:       true,
				NoStopSequence: true,
				Citations:      true,
			},
		},
	},
}

// https://docs.perplexity.ai/api-reference/chat-completions-post
type ChatRequest struct {
	Model                  string    `json:"model"`
	Messages               []Message `json:"messages"`
	SearchMode             string    `json:"search_mode,omitzero"`      // "web", "academic"
	ReasoningEffort        string    `json:"reasoning_effort,omitzero"` // "low", "medium", "high" (model: sonar-deep-research)
	MaxTokens              int64     `json:"max_tokens,omitzero"`
	Temperature            float64   `json:"temperature,omitzero"`
	TopP                   float64   `json:"top_p,omitzero"` // [0, 1.0]
	SearchDomainFilter     []string  `json:"search_domain_filter,omitzero"`
	ReturnImages           bool      `json:"return_images,omitzero"`
	ReturnRelatedQuestions bool      `json:"return_related_questions,omitzero"`
	SearchRecencyFilter    string    `json:"search_recency_filter,omitzero"`     // "month", "week", "day", "hour"
	SearchAfterDateFilter  string    `json:"search_after_date_filter,omitzero"`  // RFC3339 date
	SearchBeforeDateFilter string    `json:"search_before_date_filter,omitzero"` // RFC3339 date
	TopK                   int64     `json:"top_k,omitzero"`                     // [0, 2048^]
	Stream                 bool      `json:"stream"`
	PresencePenalty        float64   `json:"presence_penalty,omitzero"` // [-2.0, 2.0]
	FrequencyPenalty       float64   `json:"frequency_penalty,omitzero"`
	// Only available in higher tiers, see
	// https://docs.perplexity.ai/guides/usage-tiers and
	// https://docs.perplexity.ai/guides/structured-outputs
	ResponseFormat struct {
		Type       string `json:"type,omitzero"` // "json_schema", "regex"
		JSONSchema struct {
			Schema *jsonschema.Schema `json:"schema,omitzero"`
		} `json:"json_schema,omitzero"`
		Regex struct {
			Regex string `json:"regex,omitzero"`
		} `json:"regex,omitzero"`
	} `json:"response_format,omitzero"`
	WebSearchOptions struct {
		SearchContextSize string `json:"search_context_size,omitzero"` // "low", "medium", "high"
		UserLocation      struct {
			Latitude    float64 `json:"latitude,omitzero"`     // e.g. 37.7749
			Longitude   float64 `json:"longitude,omitzero"`    // e.g. -122.4194
			CountryCode string  `json:"country_code,omitzero"` // e.g. "US", "CA", "FR"
		} `json:"user_location,omitzero"`
	} `json:"web_search_options,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	// Didn't seem to increase token usage, unclear if it increases costs.
	c.ReturnImages = true
	c.ReturnRelatedQuestions = true
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
			if v.Seed != 0 {
				unsupported = append(unsupported, "Seed")
			}
			c.TopK = v.TopK
			if len(v.Stop) != 0 {
				errs = append(errs, errors.New("unsupported option Stop"))
			}
			if v.DecodeAs != nil {
				// Requires Tier 3 to work in practice.
				c.ResponseFormat.Type = "json_schema"
				c.ResponseFormat.JSONSchema.Schema = jsonschema.Reflect(v.DecodeAs)
			} else if v.ReplyAsJSON {
				errs = append(errs, errors.New("unsupported option ReplyAsJSON"))
			}
			if len(v.Tools) != 0 {
				errs = append(errs, errors.New("unsupported option Tools"))
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
		c.Messages[0].Content = sp
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
}

// https://docs.perplexity.ai/api-reference/chat-completions
type Message struct {
	Role    string `json:"role"` // "system", "assistant", "user"
	Content string `json:"content"`
}

func (m *Message) From(in *genai.Message) error {
	switch in.Role {
	case genai.User, genai.Assistant:
		m.Role = string(in.Role)
	default:
		return fmt.Errorf("unsupported role %q", in.Role)
	}
	if len(in.Contents) > 1 {
		return errors.New("perplexity doesn't support multiple content blocks; TODO split transparently")
	}
	if len(in.ToolCalls) != 0 {
		return errors.New("perplexity doesn't support tools")
	}
	if len(in.ToolCallResults) != 0 {
		return errors.New("perplexity doesn't support tools")
	}
	if in.Contents[0].Text != "" {
		m.Content = in.Contents[0].Text
	} else if in.Contents[0].Document != nil {
		// Check if this is a text/plain document
		mimeType, data, err := in.Contents[0].ReadDocument(10 * 1024 * 1024)
		if err != nil {
			return fmt.Errorf("failed to read document: %w", err)
		}
		if strings.HasPrefix(mimeType, "text/plain") {
			if in.Contents[0].URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			m.Content = string(data)
		} else {
			return fmt.Errorf("perplexity only supports text/plain documents, got %s", mimeType)
		}
	} else {
		return fmt.Errorf("unsupported content type %v", in.Contents[0])
	}
	return nil
}

// To converts the message to a genai.Message.
//
// Warning: it doesn't include the web search results, use ChatResponse.ToResult().
func (m *Message) To(out *genai.Message) error {
	switch role := m.Role; role {
	case "assistant":
		out.Role = genai.Role(role)
	default:
		return fmt.Errorf("unsupported role %q", role)
	}
	out.Contents = []genai.Content{{Text: m.Content}}
	return nil
}

type ChatResponse struct {
	ID        string    `json:"id"` // UUID
	Model     string    `json:"model"`
	Object    string    `json:"object"` // "chat.completion"
	Created   base.Time `json:"created"`
	Citations []string  `json:"citations"` // Same URLs from SearchResults in the same order.
	Images    []struct {
		Height    int64  `json:"height"`     // in pixels
		ImageURL  string `json:"image_url"`  // URL to the image
		OriginURL string `json:"origin_url"` // URL to the page that contains the image
		Width     int64  `json:"width"`      // in pixels
	} `json:"images"` // The images do not seem to have a direct relation with the citations.
	Choices []struct {
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
		Delta        struct {
			Content string `json:"content"`
			Role    string `json:"role"`
		} `json:"delta"`
	} `json:"choices"`
	RelatedQuestions []string `json:"related_questions"` // Questions related to the query
	SearchResults    []struct {
		Date  string `json:"date"` // RFC3339 date, or null
		Title string `json:"title"`
		URL   string `json:"url"` // URL to the search result
	} `json:"search_results"`
	Usage Usage `json:"usage"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Perplexity doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, errors.New("expected 1 choice")
	}
	out.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	if len(c.SearchResults) > 0 && len(out.Contents) > 0 {
		ct := genai.Citation{Type: "web", Sources: make([]genai.CitationSource, len(c.SearchResults))}
		for i := range c.SearchResults {
			ct.Sources[i].Type = "web"
			ct.Sources[i].Title = c.SearchResults[i].Title
			ct.Sources[i].URL = c.SearchResults[i].URL
			if c.SearchResults[i].Date != "" {
				ct.Sources[i].Metadata = map[string]any{"data": c.SearchResults[i].Date}
			}
		}
		out.Contents[0].Citations = append(out.Contents[0].Citations, ct)
		if len(c.Images) > 0 {
			ct := genai.Citation{Type: "document", Sources: make([]genai.CitationSource, len(c.Images))}
			for i := range c.Images {
				ct.Sources[i].Type = "image"
				ct.Sources[i].Title = c.Images[i].OriginURL
				ct.Sources[i].URL = c.Images[i].ImageURL
				ct.Sources[i].Metadata = map[string]any{
					"width":  c.Images[i].Width,
					"height": c.Images[i].Height,
				}
			}
			out.Contents[0].Citations = append(out.Contents[0].Citations, ct)
		}
	}
	return out, err
}

type FinishReason string

const (
	FinishStop   FinishReason = "stop"
	FinishLength FinishReason = "length"
)

func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishLength:
		return genai.FinishedLength
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

type Usage struct {
	PromptTokens      int64  `json:"prompt_tokens"`
	CompletionTokens  int64  `json:"completion_tokens"`
	TotalTokens       int64  `json:"total_tokens"`
	SearchContextSize string `json:"search_context_size"` // "low"
}

type ChatStreamChunkResponse = ChatResponse

//

type ErrorResponse struct {
	Detail string `json:"detail"`
	Error  struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    int    `json:"code"`
	} `json:"error"`
}

func (er *ErrorResponse) String() string {
	if er.Detail != "" {
		return "error " + er.Detail
	}
	return "error " + er.Error.Message
}

// Client implements genai.ProviderGen.
type Client struct {
	base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Perplexity platform API.
//
// If apiKey is not provided, it tries to load it from the PERPLEXITY_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://www.perplexity.ai/settings/api
//
// Models are listed at https://docs.perplexity.ai/guides/model-cards
//
// Pass model base.PreferredCheap to use a good cheap model, base.PreferredGood for a good model or
// base.PreferredSOTA to use its SOTA model. Keep in mind that as providers cycle through new models, it's
// possible the model is not available anymore.
//
// wrapper can be used to throttle outgoing requests, record calls, etc. It defaults to base.DefaultTransport.
func New(apiKey, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://www.perplexity.ai/settings/api"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("PERPLEXITY_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "PERPLEXITY_API_KEY", URL: apiKeyURL}
		}
	}
	switch model {
	case base.PreferredCheap:
		model = "sonar"
	case base.PreferredGood:
		model = "sonar-pro"
	case base.PreferredSOTA:
		model = "sonar-reasoning-pro"
	default:
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	return &Client{
		ProviderGen: base.ProviderGen[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			Model:                model,
			GenSyncURL:           "https://api.perplexity.ai/chat/completions",
			ProcessStreamPackets: processStreamPackets,
			Provider: base.Provider[*ErrorResponse]{
				ProviderName: "perplexity",
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
	}, err
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	defer func() {
		// We need to empty the channel to avoid blocking the goroutine.
		for range ch {
		}
	}()
	for pkt := range ch {
		if len(pkt.Choices) != 1 {
			continue
		}
		if pkt.Usage.PromptTokens != 0 {
			result.InputTokens = pkt.Usage.PromptTokens
			result.OutputTokens = pkt.Usage.CompletionTokens
		}
		if pkt.Choices[0].FinishReason != "" {
			result.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		f := genai.ContentFragment{TextFragment: pkt.Choices[0].Delta.Content}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		// We need to do one packet per citation type.
		if len(pkt.SearchResults) > 0 {
			f := genai.ContentFragment{Citation: genai.Citation{Type: "web", Sources: make([]genai.CitationSource, len(pkt.SearchResults))}}
			for i := range pkt.SearchResults {
				f.Citation.Sources[i].Type = "web"
				f.Citation.Sources[i].Title = pkt.SearchResults[i].Title
				f.Citation.Sources[i].URL = pkt.SearchResults[i].URL
				if pkt.SearchResults[i].Date != "" {
					f.Citation.Sources[i].Metadata = map[string]any{"data": pkt.SearchResults[i].Date}
				}
			}
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		if len(pkt.Images) > 0 {
			f := genai.ContentFragment{Citation: genai.Citation{Type: "document", Sources: make([]genai.CitationSource, len(pkt.Images))}}
			for i := range pkt.Images {
				f.Citation.Sources[i].Type = "image"
				f.Citation.Sources[i].Title = pkt.Images[i].OriginURL
				f.Citation.Sources[i].URL = pkt.Images[i].ImageURL
				f.Citation.Sources[i].Metadata = map[string]any{
					"width":  pkt.Images[i].Width,
					"height": pkt.Images[i].Height,
				}
			}
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
	_ genai.ProviderScoreboard = &Client{}
)
