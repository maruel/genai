// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package perplexity implements a client for the Perplexity API.
//
// It is described at https://docs.perplexity.ai/api-reference
package perplexity

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"reflect"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Perplexity.
//
// # Warnings
//
//   - Websearch, which is automatic for all models, is very expensive. Disable it manually when testing.
//   - No tool calling support.
var Scoreboard = scoreboard.Score{
	Country:      "US",
	DashboardURL: "https://www.perplexity.ai/settings/api",
	Scenarios: []scoreboard.Scenario{
		{
			Models:             []string{"sonar-reasoning", "sonar-reasoning-pro"},
			Thinking:           true,
			ThinkingTokenStart: "<think>",
			ThinkingTokenEnd:   "</think>",
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				JSONSchema:     true,
				Citations:      true,
				NoStopSequence: true,
			},
			GenStream: &scoreboard.FunctionalityText{
				JSONSchema:     true,
				Citations:      true,
				NoStopSequence: true,
			},
		},
		{
			Models: []string{"sonar", "sonar-pro"},
			In: map[genai.Modality]scoreboard.ModalCapability{
				genai.ModalityText: {Inline: true},
				genai.ModalityImage: {
					Inline:           true,
					URL:              true,
					SupportedFormats: []string{"image/gif", "image/jpeg", "image/png", "image/webp"},
				},
			},
			Out: map[genai.Modality]scoreboard.ModalCapability{genai.ModalityText: {Inline: true}},
			GenSync: &scoreboard.FunctionalityText{
				JSONSchema:     true,
				Citations:      true,
				NoStopSequence: true,
			},
			GenStream: &scoreboard.FunctionalityText{
				JSONSchema:     true,
				Citations:      true,
				NoStopSequence: true,
			},
		},
		// sonar-deep-research has the same capability as sonar-reasoning-pro but it is way too expensive, it cost
		// over 5$USD to run a scoreboard.
		{
			Models:             []string{"sonar-deep-research"},
			Thinking:           true,
			ThinkingTokenStart: "<think>",
			ThinkingTokenEnd:   "</think>",
		},
	},
}

// OptionsText includes Perplexity specific options.
type OptionsText struct {
	genai.OptionsText

	// DisableSearch disables websearch, which is automatic for all models.
	DisableSearch bool
	// DisableRelatedQuestions disabled related questions, to save on tokens and latency.
	DisableRelatedQuestions bool
}

// ChatRequest is documented at https://docs.perplexity.ai/api-reference/chat-completions-post
type ChatRequest struct {
	Model                   string    `json:"model"`
	Messages                []Message `json:"messages"`
	SearchMode              string    `json:"search_mode,omitzero"`      // "web", "academic"
	ReasoningEffort         string    `json:"reasoning_effort,omitzero"` // "low", "medium", "high" (model: sonar-deep-research)
	MaxTokens               int64     `json:"max_tokens,omitzero"`
	Temperature             float64   `json:"temperature,omitzero"`          // [0, 2.0]
	TopP                    float64   `json:"top_p,omitzero"`                // [0, 1.0]
	SearchDomainFilter      []string  `json:"search_domain_filter,omitzero"` // Max 10 items. Prefix "-" to exclude.
	ReturnImages            bool      `json:"return_images,omitzero"`
	ReturnRelatedQuestions  bool      `json:"return_related_questions,omitzero"`
	SearchRecencyFilter     string    `json:"search_recency_filter,omitzero"`      // "month", "week", "day", "hour"
	SearchAfterDateFilter   string    `json:"search_after_date_filter,omitzero"`   // RFC3339 date
	SearchBeforeDateFilter  string    `json:"search_before_date_filter,omitzero"`  // RFC3339 date
	LastUpdatedAfterFilter  string    `json:"last_updated_after_filter,omitzero"`  // RFC3339 date
	LastUpdatedBeforeFilter string    `json:"last_updated_before_filter,omitzero"` // RFC3339 date
	TopK                    int64     `json:"top_k,omitzero"`                      // [0, 2048]
	Stream                  bool      `json:"stream"`                              //
	PresencePenalty         float64   `json:"presence_penalty,omitzero"`           // [0, 2.0]
	FrequencyPenalty        float64   `json:"frequency_penalty,omitzero"`          // [0, 2.0]
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

	// These are "documented" at https://docs.perplexity.ai/guides/search-control-guide#curl-2
	DisableSearch          bool `json:"disable_search,omitzero"`
	EnableSearchClassifier bool `json:"enable_search_classifier,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	c.Model = model
	if err := msgs.Validate(); err != nil {
		return err
	}
	// Didn't seem to increase token usage, unclear if it increases costs.
	c.ReturnImages = true
	// This likely increase token usage.
	c.ReturnRelatedQuestions = true
	var errs []error
	var unsupported []string
	sp := ""
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return err
		}
		switch v := opts.(type) {
		case *genai.OptionsText:
			unsupported, errs = c.initOptions(v)
			sp = v.SystemPrompt
		case *OptionsText:
			unsupported, errs = c.initOptions(&v.OptionsText)
			sp = v.SystemPrompt
			c.DisableSearch = v.DisableSearch
			c.ReturnRelatedQuestions = !v.DisableRelatedQuestions
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
		c.Messages[0] = Message{Role: "system", Content: Contents{Content{Type: "text", Text: sp}}}
	}
	for i := range msgs {
		if err := c.Messages[i+offset].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
		}
	}
	// If we have unsupported features but no other errors, return a continuable error
	if len(unsupported) > 0 && len(errs) == 0 {
		return &genai.UnsupportedContinuableError{Unsupported: unsupported}
	}
	return errors.Join(errs...)
}

func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

func (c *ChatRequest) initOptions(v *genai.OptionsText) ([]string, []error) {
	var unsupported []string
	var errs []error
	c.MaxTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	if v.Seed != 0 {
		unsupported = append(unsupported, "Seed")
	}
	c.TopK = v.TopK
	if v.TopLogprobs > 0 {
		unsupported = append(unsupported, "TopLogprobs")
	}
	if len(v.Stop) != 0 {
		errs = append(errs, errors.New("unsupported option Stop"))
	}
	if v.DecodeAs != nil {
		// Requires Tier 3 to work in practice.
		c.ResponseFormat.Type = "json_schema"
		c.ResponseFormat.JSONSchema.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
	} else if v.ReplyAsJSON {
		errs = append(errs, errors.New("unsupported option ReplyAsJSON"))
	}
	if len(v.Tools) != 0 {
		errs = append(errs, errors.New("unsupported option Tools"))
	}
	return unsupported, errs
}

// Message is documented at https://docs.perplexity.ai/api-reference/chat-completions
type Message struct {
	Role    string   `json:"role"` // "system", "assistant", "user"
	Content Contents `json:"content"`
}

// From converts from a genai.Message to a Message.
func (m *Message) From(in *genai.Message) error {
	switch r := in.Role(); r {
	case "user", "assistant":
		m.Role = r
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	if len(in.ToolCallResults) != 0 {
		return errors.New("perplexity doesn't support tools")
	}
	for i := range in.Requests {
		if in.Requests[i].Text != "" {
			m.Content = append(m.Content, Content{Type: "text", Text: in.Requests[i].Text})
		} else if !in.Requests[i].Doc.IsZero() {
			// Check if this is a text/plain document
			mimeType, data, err := in.Requests[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("request #%d: failed to read document: %w", i, err)
			}
			switch {
			case strings.HasPrefix(mimeType, "text/plain"):
				if in.Requests[i].Doc.URL != "" {
					return fmt.Errorf("request #%d: text/plain documents must be provided inline, not as a URL", i)
				}
				m.Content = append(m.Content, Content{Type: "text", Text: string(data)})
			case strings.HasPrefix(mimeType, "image/"):
				c := Content{Type: "image_url"}
				if in.Requests[i].Doc.URL != "" {
					c.ImageURL.URL = in.Requests[i].Doc.URL
				} else {
					c.ImageURL.URL = "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)
				}
				m.Content = append(m.Content, c)
			default:
				return fmt.Errorf("request #%d: perplexity only supports text/plain documents, got %s", i, mimeType)
			}
		} else {
			return fmt.Errorf("request #%d: unknown Request type", i)
		}
	}
	for i := range in.Replies {
		if len(in.Replies[i].Opaque) != 0 {
			return fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)
		}
		if in.Replies[i].Text != "" {
			m.Content = append(m.Content, Content{Type: "text", Text: in.Replies[i].Text})
		} else if !in.Requests[i].Doc.IsZero() {
			// Check if this is a text/plain document
			mimeType, data, err := in.Replies[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("reply #%d: failed to read document: %w", i, err)
			}
			switch {
			case strings.HasPrefix(mimeType, "text/plain"):
				if in.Replies[i].Doc.URL != "" {
					return fmt.Errorf("reply #%d: text/plain documents must be provided inline, not as a URL", i)
				}
				m.Content = append(m.Content, Content{Type: "text", Text: string(data)})
			case strings.HasPrefix(mimeType, "image/"):
				c := Content{Type: "image_url"}
				if in.Replies[i].Doc.URL != "" {
					c.ImageURL.URL = in.Replies[i].Doc.URL
				} else {
					c.ImageURL.URL = "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)
				}
				m.Content = append(m.Content, c)
			default:
				return fmt.Errorf("reply #%d: perplexity only supports text/plain documents, got %s", i, mimeType)
			}
		} else {
			return fmt.Errorf("reply #%d: unknown Reply type", i)
		}
	}
	return nil
}

// To converts the message to a genai.Message.
//
// Warning: it doesn't include the web search results, use ChatResponse.ToResult().
func (m *Message) To(out *genai.Message) error {
	for i := range m.Content {
		if m.Content[i].Type == "text" {
			out.Replies = append(out.Replies, genai.Reply{Text: m.Content[i].Text})
		} else {
			return fmt.Errorf("unsupported content type %q", m.Content[i].Type)
		}
	}
	return nil
}

type Content struct {
	Type     string `json:"type"` // "text", "image_url"
	Text     string `json:"text,omitzero"`
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`
}

type Contents []Content

func (c *Contents) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		*c = nil
		return nil
	}
	if err := json.Unmarshal(b, (*[]Content)(c)); err == nil {
		return nil
	}

	v := Content{}
	if err := json.Unmarshal(b, &v); err == nil {
		*c = Contents{v}
		return nil
	}

	s := ""
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	*c = Contents{{Type: "text", Text: s}}
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
		Date        string `json:"date"` // RFC3339 date, or null
		Title       string `json:"title"`
		URL         string `json:"url"`          // URL to the search result
		LastUpdated string `json:"last_updated"` // YYYY-MM-DD
	} `json:"search_results"`
	Usage Usage `json:"usage"`
}

func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Perplexity doesn't support cached tokens.
		Usage: genai.Usage{
			InputTokens:     c.Usage.PromptTokens,
			ReasoningTokens: c.Usage.ReasoningTokens,
			OutputTokens:    c.Usage.CompletionTokens,
			TotalTokens:     c.Usage.TotalTokens,
		},
	}
	if len(c.Choices) != 1 {
		return out, errors.New("expected 1 choice")
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	if len(c.SearchResults) > 0 && len(out.Replies) > 0 {
		ct := genai.Citation{Type: "web", Sources: make([]genai.CitationSource, len(c.SearchResults))}
		for i := range c.SearchResults {
			ct.Sources[i].Type = "web"
			ct.Sources[i].Title = c.SearchResults[i].Title
			ct.Sources[i].URL = c.SearchResults[i].URL
			if c.SearchResults[i].Date != "" {
				ct.Sources[i].Metadata = map[string]any{"data": c.SearchResults[i].Date}
			}
		}
		out.Replies[0].Citations = append(out.Replies[0].Citations, ct)
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
			out.Replies[0].Citations = append(out.Replies[0].Citations, ct)
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
	ReasoningTokens   int64  `json:"reasoning_tokens"`
	CitationTokens    int64  `json:"citation_tokens"`
	NumSearchQueries  int64  `json:"num_search_queries"`
	Cost              struct {
		RequestCost         float64 `json:"request_cost"`
		InputTokensCost     float64 `json:"input_tokens_cost"`
		OutputTokensCost    float64 `json:"output_tokens_cost"`
		ReasoningTokensCost float64 `json:"reasoning_tokens_cost"`
		CitationTokensCost  float64 `json:"citation_tokens_cost"`
		SearchQueriesCost   float64 `json:"search_queries_cost"`
		TotalCost           float64 `json:"total_cost"`
	} `json:"cost"`
}

type ChatStreamChunkResponse = ChatResponse

//

type ErrorResponse struct {
	Detail   string `json:"detail"`
	ErrorVal struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    int    `json:"code"`
	} `json:"error"`
}

func (er *ErrorResponse) Error() string {
	if er.Detail != "" {
		return er.Detail
	}
	if er.ErrorVal.Code != 0 {
		return fmt.Sprintf("%s (%d): %s", er.ErrorVal.Type, er.ErrorVal.Code, er.ErrorVal.Message)
	}
	return er.ErrorVal.Message
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
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
// To use multiple models, create multiple clients.
// Models are listed at https://docs.perplexity.ai/guides/model-cards
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	if opts.Remote != "" {
		return nil, errors.New("unexpected option Remote")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://www.perplexity.ai/settings/api"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("PERPLEXITY_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "PERPLEXITY_API_KEY", URL: apiKeyURL}
		}
	}
	// Perplexity doesn't have a list model API.
	model := opts.Model
	switch model {
	case genai.ModelNone:
		model = ""
	case genai.ModelCheap, "":
		model = "sonar"
	case genai.ModelGood:
		model = "sonar-pro"
	case genai.ModelSOTA:
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
				APIKeyURL: apiKeyURL,
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

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "perplexity"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.Model
}

// Scoreboard implements scoreboard.ProviderScore.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard
}

// GenSync implements genai.ProviderGen.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	return c.ProviderGen.GenSync(ctx, msgs, opts)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.ProviderGen.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.ProviderGen.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ReplyFragment, opts genai.Options) (genai.Result, error) {
	return c.ProviderGen.GenStream(ctx, msgs, chunks, opts)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest, out chan<- ChatStreamChunkResponse) error {
	return c.ProviderGen.GenStreamRaw(ctx, in, out)
}

func processStreamPackets(ch <-chan ChatStreamChunkResponse, chunks chan<- genai.ReplyFragment, result *genai.Result) error {
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
			result.Usage.InputTokens = pkt.Usage.PromptTokens
			result.Usage.OutputTokens = pkt.Usage.CompletionTokens
		}
		if pkt.Choices[0].FinishReason != "" {
			result.Usage.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
		}
		switch role := pkt.Choices[0].Delta.Role; role {
		case "", "assistant":
		default:
			return fmt.Errorf("unexpected role %q", role)
		}
		f := genai.ReplyFragment{TextFragment: pkt.Choices[0].Delta.Content}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
		// We need to do one packet per citation type.
		if len(pkt.SearchResults) > 0 {
			f := genai.ReplyFragment{Citation: genai.Citation{Type: "web", Sources: make([]genai.CitationSource, len(pkt.SearchResults))}}
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
			f := genai.ReplyFragment{Citation: genai.Citation{Type: "document", Sources: make([]genai.CitationSource, len(pkt.Images))}}
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
	_ scoreboard.ProviderScore = &Client{}
)
