// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Perplexity chat completions API.
//
// Reference: https://docs.perplexity.ai/api-reference/chat-completions-post

package perplexity

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

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
			Schema genai.JSONSchema `json:"schema,omitzero"`
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
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
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
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *genai.GenOptionText:
			unsupported, errs = c.initOptionsText(v)
			sp = v.SystemPrompt
		case *genai.GenOptionTools:
			if len(v.Tools) != 0 {
				errs = append(errs, errors.New("unsupported options GenOptionTools.Tools"))
			}
			if v.Force == genai.ToolCallRequired {
				unsupported = append(unsupported, "GenOptionTools.Force")
			}
		case *genai.GenOptionWeb:
			c.DisableSearch = !v.Search
			if v.Fetch {
				errs = append(errs, errors.New("unsupported GenOptionWeb.Fetch"))
			}
		case *GenOption:
			c.ReturnRelatedQuestions = !v.DisableRelatedQuestions
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
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
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// SetStream sets the streaming mode.
func (c *ChatRequest) SetStream(stream bool) {
	c.Stream = stream
}

func (c *ChatRequest) initOptionsText(v *genai.GenOptionText) ([]string, []error) {
	var unsupported []string
	var errs []error
	c.MaxTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	c.TopK = v.TopK
	if v.TopLogprobs > 0 {
		unsupported = append(unsupported, "GenOptionText.TopLogprobs")
	}
	if len(v.Stop) != 0 {
		errs = append(errs, errors.New("unsupported option Stop"))
	}
	if v.DecodeAs != nil {
		// Requires Tier 3 to work in practice.
		c.ResponseFormat.Type = "json_schema"
		s, err := genai.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
		if err != nil {
			errs = append(errs, err)
		} else {
			c.ResponseFormat.JSONSchema.Schema = s
		}
	} else if v.ReplyAsJSON {
		errs = append(errs, errors.New("unsupported option ReplyAsJSON"))
	}
	return unsupported, errs
}

// Message is documented at https://docs.perplexity.ai/api-reference/chat-completions
type Message struct {
	Role    string   `json:"role"` // "system", "assistant", "user"
	Content Contents `json:"content,omitzero"`
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
		switch {
		case in.Requests[i].Text != "":
			m.Content = append(m.Content, Content{Type: "text", Text: in.Requests[i].Text})
		case !in.Requests[i].Doc.IsZero():
			// Check if this is a text document
			mimeType, data, err := in.Requests[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("request #%d: failed to read document: %w", i, err)
			}
			switch {
			// text/plain, text/markdown
			case strings.HasPrefix(mimeType, "text/"):
				if in.Requests[i].Doc.URL != "" {
					return fmt.Errorf("request #%d: %s documents must be provided inline, not as a URL", i, mimeType)
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
				return fmt.Errorf("request #%d: perplexity only supports text documents, got %s", i, mimeType)
			}
		default:
			return fmt.Errorf("request #%d: unknown Request type", i)
		}
	}
	for i := range in.Replies {
		if len(in.Replies[i].Opaque) != 0 {
			return &internal.BadError{Err: fmt.Errorf("reply #%d: field Reply.Opaque not supported", i)}
		}
		switch {
		case in.Replies[i].Text != "":
			m.Content = append(m.Content, Content{Type: "text", Text: in.Replies[i].Text})
		case !in.Requests[i].Doc.IsZero():
			// Check if this is a text document
			mimeType, data, err := in.Replies[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return &internal.BadError{Err: fmt.Errorf("reply #%d: failed to read document: %w", i, err)}
			}
			switch {
			// text/plain, text/markdown
			case strings.HasPrefix(mimeType, "text/"):
				if in.Replies[i].Doc.URL != "" {
					return &internal.BadError{Err: fmt.Errorf("reply #%d: %s documents must be provided inline, not as a URL", i, mimeType)}
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
				return &internal.BadError{Err: fmt.Errorf("reply #%d: perplexity only supports text documents, got %s", i, mimeType)}
			}
		case in.Replies[i].Reasoning != "":
			// Ignore
		default:
			return &internal.BadError{Err: errors.New("unknown Reply type")}
		}
	}
	return nil
}

// To converts the message to a genai.Message.
func (m *Message) To(search []SearchResult, images []Images, related []string, out *genai.Message) error {
	for i := range m.Content {
		if m.Content[i].Type == "text" {
			out.Replies = append(out.Replies, genai.Reply{Text: m.Content[i].Text})
		} else {
			return &internal.BadError{Err: fmt.Errorf("unsupported content type %q", m.Content[i].Type)}
		}
	}
	if len(search) > 0 {
		ct := genai.Citation{Sources: make([]genai.CitationSource, len(search))}
		for i := range search {
			ct.Sources[i].Type = genai.CitationWeb
			ct.Sources[i].Title = search[i].Title
			ct.Sources[i].URL = search[i].URL
			ct.Sources[i].Date = search[i].Date
		}
		out.Replies = append(out.Replies, genai.Reply{Citation: ct})
	}
	if len(images) > 0 {
		ct := genai.Citation{Sources: make([]genai.CitationSource, len(images))}
		for i := range images {
			ct.Sources[i].Type = genai.CitationDocument
			ct.Sources[i].Title = images[i].OriginURL
			ct.Sources[i].URL = images[i].ImageURL
			ct.Sources[i].Metadata = map[string]any{
				"width":  images[i].Width,
				"height": images[i].Height,
			}
		}
		out.Replies = append(out.Replies, genai.Reply{Citation: ct})
	}
	// if len(related) > 0 {
	// TODO: Figure out how to return this.
	// }
	return nil
}

// Content is a provider-specific content block.
type Content struct {
	Type     string `json:"type"` // "text", "image_url"
	Text     string `json:"text,omitzero"`
	ImageURL struct {
		URL string `json:"url,omitzero"`
	} `json:"image_url,omitzero"`
}

// Contents is a collection of content blocks.
type Contents []Content

// UnmarshalJSON implements json.Unmarshaler.
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

// ChatResponse is the provider-specific chat completion response.
type ChatResponse struct {
	ID        string    `json:"id"` // UUID
	Model     string    `json:"model"`
	Object    string    `json:"object"` // "chat.completion"
	Created   base.Time `json:"created"`
	Citations []string  `json:"citations"` // Same URLs from SearchResults in the same order.
	Images    []Images  `json:"images"`    // The images do not seem to have a direct relation with the citations.
	Choices   []struct {
		Index        int64        `json:"index"`
		FinishReason FinishReason `json:"finish_reason"`
		Message      Message      `json:"message"`
		Delta        struct {
			Content string `json:"content"`
			Role    string `json:"role"`
		} `json:"delta"`
	} `json:"choices"`
	RelatedQuestions []string       `json:"related_questions"` // Questions related to the query
	SearchResults    []SearchResult `json:"search_results"`
	Usage            Usage          `json:"usage"`
}

// ToResult converts the response to a genai.Result.
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
	err := c.Choices[0].Message.To(c.SearchResults, c.Images, c.RelatedQuestions, &out.Message)
	return out, err
}

// Images is a provider-specific images container.
type Images struct {
	Height    int64  `json:"height"`     // in pixels
	ImageURL  string `json:"image_url"`  // URL to the image
	OriginURL string `json:"origin_url"` // URL to the page that contains the image
	Width     int64  `json:"width"`      // in pixels
	Title     string `json:"title,omitzero"`
}

// SearchResult is a provider-specific search result.
type SearchResult struct {
	Date        string `json:"date"` // RFC3339 date, or null
	Title       string `json:"title"`
	URL         string `json:"url"`          // URL to the search result
	LastUpdated string `json:"last_updated"` // YYYY-MM-DD
	Snippet     string `json:"snippet"`      // TODO: Add!
	Source      string `json:"source,omitzero"`
}

// FinishReason is a provider-specific finish reason.
type FinishReason string

// Finish reason values.
const (
	FinishStop   FinishReason = "stop"
	FinishLength FinishReason = "length"
)

// ToFinishReason converts to a genai.FinishReason.
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

// Usage is the provider-specific token usage.
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

// ChatStreamChunkResponse is the provider-specific streaming chat chunk.
type ChatStreamChunkResponse = ChatResponse

// ErrorResponse is the provider-specific error response.
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

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
