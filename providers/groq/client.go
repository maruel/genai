// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package groq implements a client for the Groq API.
//
// It is described at https://console.groq.com/docs/api-reference
package groq

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"reflect"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Groq.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// GenOption is the Groq-specific options.
type GenOption struct {
	// ReasoningFormat requests Groq to process the stream on our behalf. It must only be used on reasoning
	// models. It is required for reasoning models to enable JSON structured output or tool calling.
	ReasoningFormat ReasoningFormat
	// ServiceTier specify the priority.
	ServiceTier ServiceTier
}

// Validate implements genai.Validatable.
func (o *GenOption) Validate() error {
	// TODO: validate ReasoningFormat and ServiceTier.
	return nil
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *ChatRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
	c.Model = model
	if err := msgs.Validate(); err != nil {
		return err
	}
	var errs []error
	var unsupported []string
	sp := ""
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *GenOption:
			c.ServiceTier = v.ServiceTier
			c.ReasoningFormat = v.ReasoningFormat
		case *genai.GenOptionText:
			unsupported = append(unsupported, c.initOptionsText(v)...)
			sp = v.SystemPrompt
		case *genai.GenOptionTools:
			c.initOptionsTools(v)
		case *genai.GenOptionWeb:
			if v.Search {
				// https://console.groq.com/docs/browser-search
				// TODO: Country and domains
				c.SearchSettings.IncludeImages = true
				c.Tools = append(c.Tools, Tool{Type: "browser_search"})
			}
			// Fetch (visit_website) is only available on compound models, not chat completions.
			// https://console.groq.com/docs/agentic-tooling
			if v.Fetch {
				unsupported = append(unsupported, "GenOptionWeb.Fetch")
			}
		case genai.GenOptionSeed:
			c.Seed = int64(v)
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}

	if sp != "" {
		c.Messages = append(c.Messages, Message{Role: "system", Content: []Content{{Type: ContentText, Text: sp}}})
	}
	for i := range msgs {
		if len(msgs[i].ToolCallResults) > 1 {
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: tool call results #%d: %w", i, j, err))
				} else {
					c.Messages = append(c.Messages, newMsg)
				}
			}
		} else {
			var newMsg Message
			if err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
			} else {
				c.Messages = append(c.Messages, newMsg)
			}
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

func (c *ChatRequest) initOptionsText(v *genai.GenOptionText) []string {
	var unsupported []string
	c.MaxChatTokens = v.MaxTokens
	c.Temperature = v.Temperature
	c.TopP = v.TopP
	if v.TopK != 0 {
		unsupported = append(unsupported, "GenOptionText.TopK")
	}
	if v.TopLogprobs != 0 {
		unsupported = append(unsupported, "GenOptionText.TopLogprobs")
	}
	c.Stop = v.Stop
	if v.DecodeAs != nil {
		c.ResponseFormat.Type = "json_schema"
		c.ResponseFormat.JSONSchema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
		c.ResponseFormat.JSONSchema.Extras = map[string]any{"name": "response"}
	} else if v.ReplyAsJSON {
		c.ResponseFormat.Type = "json_object"
	}
	return unsupported
}

func (c *ChatRequest) initOptionsTools(v *genai.GenOptionTools) {
	if len(v.Tools) != 0 {
		switch v.Force {
		case genai.ToolCallAny:
			c.ToolChoice = "auto"
		case genai.ToolCallRequired:
			c.ToolChoice = "required"
		case genai.ToolCallNone:
			c.ToolChoice = "none"
		}
		// Documentation states max is 128 tools.
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
}

// From must be called with at most one ToolCallResults.
func (m *Message) From(in *genai.Message) error {
	if len(in.ToolCallResults) > 1 {
		return errors.New("internal error")
	}
	switch r := in.Role(); r {
	case "user", "assistant":
		m.Role = r
	case "computer":
		m.Role = "tool"
	default:
		return fmt.Errorf("unsupported role %q", r)
	}
	m.Name = in.User
	if len(in.Requests) != 0 {
		m.Content = make(Contents, len(in.Requests))
		for i := range in.Requests {
			if err := m.Content[i].FromRequest(&in.Requests[i]); err != nil {
				return fmt.Errorf("request #%d: %w", i, err)
			}
		}
	}
	if len(in.Replies) != 0 {
		m.Content = make(Contents, 0, len(in.Replies))
		for i := range in.Replies {
			if in.Replies[i].Reasoning != "" {
				// DeepSeek and Qwen recommend against passing reasoning back.
				continue
			}
			if !in.Replies[i].ToolCall.IsZero() {
				m.ToolCalls = append(m.ToolCalls, ToolCall{})
				if err := m.ToolCalls[len(m.ToolCalls)-1].From(&in.Replies[i].ToolCall); err != nil {
					return fmt.Errorf("reply #%d: %w", i, err)
				}
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[i]); err != nil {
				return fmt.Errorf("reply #%d: %w", i, err)
			}
		}
	}
	if len(in.ToolCallResults) != 0 {
		// Process only the first tool call result in this method.
		// The Init method handles multiple tool call results by creating multiple messages.
		m.Content = Contents{{Type: ContentText, Text: in.ToolCallResults[0].Result}}
		m.ToolCallID = in.ToolCallResults[0].ID
	}
	return nil
}

// To converts to the genai equivalent.
func (m *Message) To(out *genai.Message) error {
	if m.Reasoning != "" {
		out.Replies = append(out.Replies, genai.Reply{Reasoning: m.Reasoning})
	}
	for _, c := range m.Content {
		switch c.Type {
		case ContentText:
			if c.Text == "" {
				return &internal.BadError{Err: errors.New("empty content text")}
			}
			out.Replies = append(out.Replies, genai.Reply{Text: c.Text})
		case ContentImageURL:
			return &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
		default:
			return &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
		}
	}
	for i := range m.ToolCalls {
		out.Replies = append(out.Replies, genai.Reply{})
		m.ToolCalls[i].To(&out.Replies[len(out.Replies)-1].ToolCall)
	}
	for _, t := range m.ExecutedTools {
		// If Name is empty, use Type instead
		toolName := t.Name
		if toolName == "" {
			toolName = t.Type
		}
		switch toolName {
		case "browser.search", "search":
			d := json.NewDecoder(strings.NewReader(t.Arguments))
			d.DisallowUnknownFields()
			args := BrowserSearchArguments{}
			if err := d.Decode(&args); err != nil {
				return &internal.BadError{Err: fmt.Errorf("failed to unmarshal arguments for executed tool %q: %w", toolName, err)}
			}
			c := genai.Citation{
				Sources: make([]genai.CitationSource, 0, len(t.SearchResults.Results)+1),
			}
			c.Sources = append(c.Sources, genai.CitationSource{
				Type:    genai.CitationWebQuery,
				Snippet: args.Query,
			})
			for _, r := range t.SearchResults.Results {
				c.Sources = append(c.Sources, genai.CitationSource{
					Type: genai.CitationWeb, Title: r.Title, URL: r.URL, Snippet: r.Content,
				})
			}
			out.Replies = append(out.Replies, genai.Reply{Citation: c})
		case "browser.open", "browser.find", "visit":
			// Ignore, it's really useless.
		case "python":
			// Ignore python execution tool results from model reasoning
		default:
			return &internal.BadError{Err: fmt.Errorf("implement executed tool %q", toolName)}
		}
	}
	return nil
}

// IsZero reports whether the value is zero.
func (c *Contents) IsZero() bool {
	return len(*c) == 0
}

// MarshalJSON implements json.Marshaler.
func (c *Contents) MarshalJSON() ([]byte, error) {
	if len(*c) == 0 {
		// It's important otherwise Qwen3 fails with:
		// ('messages.2.content' : value must be a string) OR ('messages.2.content' : minimum number of items is 1)
		return []byte("null"), nil
	}
	if len(*c) == 1 && (*c)[0].Type == ContentText {
		return json.Marshal((*c)[0].Text)
	}
	return json.Marshal([]Content(*c))
}

// UnmarshalJSON implements json.Unmarshaler.
func (c *Contents) UnmarshalJSON(b []byte) error {
	if bytes.Equal(b, []byte("null")) {
		// e.g. tool calls.
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
	if s != "" {
		*c = Contents{{Type: ContentText, Text: s}}
	} else {
		// Decode empty string as nil.
		*c = nil
	}
	return nil
}

// FromRequest converts from a genai request.
func (c *Content) FromRequest(in *genai.Request) error {
	// DeepSeek and Qwen recommend against passing reasoning back to the model.
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			return fmt.Errorf("unsupported mime type %s", mimeType)
		}
		return nil
	}
	return errors.New("unknown Request type")
}

// FromReply converts from a genai reply.
func (c *Content) FromReply(in *genai.Reply) error {
	if len(in.Opaque) != 0 {
		return &internal.BadError{Err: errors.New("field Reply.Opaque not supported")}
	}
	// DeepSeek and Qwen recommend against passing reasoning back to the model.
	if in.Text != "" {
		c.Type = ContentText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		switch {
		case (in.Doc.URL != "" && mimeType == "") || strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentImageURL
			if in.Doc.URL == "" {
				c.ImageURL.URL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL.URL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			c.Type = ContentText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			return &internal.BadError{Err: fmt.Errorf("unsupported mime type %s", mimeType)}
		}
		return nil
	}
	// TODO: ExecutedTools.
	return &internal.BadError{Err: errors.New("unknown Reply type")}
}

// From converts from the genai equivalent.
func (t *ToolCall) From(in *genai.ToolCall) error {
	if len(in.Opaque) != 0 {
		return errors.New("field ToolCall.Opaque not supported")
	}
	t.Type = "function"
	t.ID = in.ID
	t.Function.Name = in.Name
	t.Function.Arguments = in.Arguments
	return nil
}

// To converts to the genai equivalent.
func (t *ToolCall) To(out *genai.ToolCall) {
	out.ID = t.ID
	out.Name = t.Function.Name
	out.Arguments = t.Function.Arguments
}

// ToResult converts the response to a genai.Result.
func (c *ChatResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		// At the moment, Groq does not support cached tokens.
		Usage: genai.Usage{
			InputTokens:  c.Usage.PromptTokens,
			OutputTokens: c.Usage.CompletionTokens,
			TotalTokens:  c.Usage.TotalTokens,
			ServiceTier:  string(c.ServiceTier),
		},
	}
	if len(c.Choices) != 1 {
		return out, &internal.BadError{Err: fmt.Errorf("server returned an unexpected number of choices, expected 1, got %d", len(c.Choices))}
	}
	out.Usage.FinishReason = c.Choices[0].FinishReason.ToFinishReason()
	err := c.Choices[0].Message.To(&out.Message)
	return out, err
}

// ToFinishReason converts to a genai.FinishReason.
func (f FinishReason) ToFinishReason() genai.FinishReason {
	switch f {
	case FinishStop:
		return genai.FinishedStop
	case FinishLength:
		return genai.FinishedLength
	case FinishToolCalls:
		return genai.FinishedToolCalls
	case FinishContentFilter:
		return genai.FinishedContentFilter
	default:
		if !internal.BeLenient {
			panic(f)
		}
		return genai.FinishReason(f)
	}
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	suffix := ""
	if !m.Active {
		suffix = " (inactive)"
	}
	return fmt.Sprintf("%s (%s) Context: %d/%d%s", m.ID, m.Created.AsTime().Format("2006-01-02"), m.ContextWindow, m.MaxCompletionTokens, suffix)
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return m.ContextWindow
}

// ToModels converts Groq models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

//

func (er *ErrorResponse) Error() string {
	suffix := ""
	if er.ErrorVal.Param != "" {
		suffix += fmt.Sprintf(" (param: %q)", er.ErrorVal.Param)
	}
	if er.ErrorVal.FailedGeneration != "" {
		suffix += fmt.Sprintf("failed generation: %q", er.ErrorVal.FailedGeneration)
	}
	return fmt.Sprintf("%s (%s): %s%s", er.ErrorVal.Code, er.ErrorVal.Type, er.ErrorVal.Message, suffix)
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]
}

// New creates a new client to talk to the Groq platform API.
//
// If opts.APIKey is not provided, it tries to load it from the GROQ_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://console.groq.com/keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://console.groq.com/dashboard/limits or https://console.groq.com/docs/models
//
// Tool use requires the use of a model that supports it.
// https://console.groq.com/docs/tool-use
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
	if err := base.CheckDuplicateOptions(opts); err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionAPIKey:
			apiKey = string(v)
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			modalities = genai.Modalities(v)
		case genai.ProviderOptionPreloadedModels:
			preloadedModels = []genai.Model(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://console.groq.com/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("GROQ_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "GROQ_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityText}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only text is supported", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *ChatRequest, *ChatResponse, ChatStreamChunkResponse]{
			GenSyncURL:      "https://api.groq.com/openai/v1/chat/completions",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProcessHeaders:  processHeaders,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: apiKeyURL,
				Lenient:   internal.BeLenient,
				Client: http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"Authorization": {"Bearer " + apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
				return nil, err
			}
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	nb := regexp.MustCompile(`(\d+)`)
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Skip non-general-purpose models.
		if strings.Contains(m.ID, "safeguard") {
			continue
		}
		// This is meh.
		switch {
		case cheap:
			if strings.HasPrefix(m.ID, "openai/") {
				if selectedModel != "" {
					m1 := nb.FindStringSubmatch(m.ID)
					m2 := nb.FindStringSubmatch(selectedModel)
					i1, _ := strconv.Atoi(m1[1])
					i2, _ := strconv.Atoi(m2[1])
					if i1 > i2 {
						continue
					}
				}
				selectedModel = m.ID
			}
		case good:
			if strings.Contains(m.ID, "openai/") {
				if selectedModel != "" {
					m1 := nb.FindStringSubmatch(m.ID)
					m2 := nb.FindStringSubmatch(selectedModel)
					i1, _ := strconv.Atoi(m1[1])
					i2, _ := strconv.Atoi(m2[1])
					if i1 < i2 {
						continue
					}
				}
				selectedModel = m.ID
			}
		default:
			if strings.HasPrefix(m.ID, "moonshotai/") {
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "groq"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to fetch results (e.g. videos) generated by the provider.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in *ChatRequest, out *ChatResponse) error {
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return c.impl.GenStream(ctx, msgs, opts...)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *ChatRequest) (iter.Seq[ChatStreamChunkResponse], func() error) {
	return c.impl.GenStreamRaw(ctx, in)
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://console.groq.com/docs/api-reference#models-list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.groq.com/openai/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ChatStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}

	return func(yield func(genai.Reply) bool) {
			for pkt := range chunks {
				if len(pkt.Choices) != 1 {
					continue
				}
				if pkt.Xgroq.Usage.TotalTokens != 0 {
					u.InputTokens = pkt.Xgroq.Usage.PromptTokens
					u.OutputTokens = pkt.Xgroq.Usage.CompletionTokens
					u.TotalTokens = pkt.Xgroq.Usage.TotalTokens
					u.FinishReason = pkt.Choices[0].FinishReason.ToFinishReason()
				}
				switch role := pkt.Choices[0].Delta.Role; role {
				case "assistant", "":
				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("unexpected role %q", role)}
					return
				}
				if s := pkt.Choices[0].Delta.Reasoning; s != "" {
					if !yield(genai.Reply{Reasoning: s}) {
						return
					}
				}
				for _, c := range pkt.Choices[0].Delta.Content {
					switch c.Type {
					case ContentText:
						if !yield(genai.Reply{Text: c.Text}) {
							return
						}
					case ContentImageURL:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
						return
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
						return
					}
				}
				for _, t := range pkt.Choices[0].Delta.ToolCalls {
					f := genai.Reply{}
					t.To(&f.ToolCall)
					if !yield(f) {
						return
					}
				}
				for _, t := range pkt.Choices[0].Delta.ExecutedTools {
					// Investigate merging with Message.To.
					// If Name is empty, use Type instead
					toolName := t.Name
					if toolName == "" {
						toolName = t.Type
					}
					switch toolName {
					case "browser.search", "search":
						d := json.NewDecoder(strings.NewReader(t.Arguments))
						d.DisallowUnknownFields()
						args := BrowserSearchArguments{}
						if err := d.Decode(&args); err != nil {
							finalErr = &internal.BadError{
								Err: fmt.Errorf("failed to unmarshal arguments for executed tool %q: %w", toolName, err),
							}
							return
						}
						f := genai.Reply{}
						f.Citation.Sources = make([]genai.CitationSource, 0, len(t.SearchResults.Results)+1)
						f.Citation.Sources = append(f.Citation.Sources, genai.CitationSource{
							Type:    genai.CitationWebQuery,
							Snippet: args.Query,
						})
						for _, r := range t.SearchResults.Results {
							f.Citation.Sources = append(f.Citation.Sources, genai.CitationSource{
								Type: genai.CitationWeb, Title: r.Title, URL: r.URL, Snippet: r.Content,
							})
						}
						if !yield(f) {
							return
						}
					case "browser.open", "browser.find", "visit":
						// Ignore, it's really useless.
					case "python":
						// Ignore python execution tool results from model reasoning
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement executed tool %q", toolName)}
						return
					}
				}
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, nil, finalErr
		}
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests"), 10, 64)
	requestsReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Requests"))

	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)
	tokensReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Tokens"))

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(requestsReset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(tokensReset).Round(10 * time.Millisecond),
		})
	}
	return limits
}

var _ genai.Provider = &Client{}
