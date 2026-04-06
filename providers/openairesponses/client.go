// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openairesponses implements a client for the OpenAI Responses API.
//
// It is described at https://platform.openai.com/docs/api-reference/responses/create
package openairesponses

// See official client at http://pkg.go.dev/github.com/openai/openai-go/responses

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"mime"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
	"github.com/maruel/roundtrippers"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for OpenAI.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// GenOptionText defines OpenAI Responses specific options.
type GenOptionText struct {
	// ReasoningEffort is the amount of effort (number of tokens) the LLM can use to think about the answer.
	//
	// When unspecified, defaults to medium.
	ReasoningEffort ReasoningEffort
	// ServiceTier specify the priority.
	ServiceTier ServiceTier
	// Truncation controls automatic shortening of long conversations.
	Truncation Truncation
	// PreviousResponseID enables server-side conversation state, avoiding re-transmitting full history.
	PreviousResponseID string
}

// Validate implements genai.Validatable.
func (o *GenOptionText) Validate() error {
	if err := o.ReasoningEffort.Validate(); err != nil {
		return err
	}
	return o.ServiceTier.Validate()
}

// GenOptionImage defines OpenAI specific options.
type GenOptionImage struct {
	// Background is only supported on gpt-image-1.
	Background Background
}

// Validate implements genai.Validatable.
func (o *GenOptionImage) Validate() error {
	return nil
}

// Init implements base.InitializableRequest.
func (r *Response) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
	var unsupported []string
	var errs []error
	r.Model = model
	r.Reasoning.Summary = "auto"
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *GenOptionText:
			r.Reasoning.Effort = v.ReasoningEffort
			r.ServiceTier = v.ServiceTier
			r.Truncation = string(v.Truncation)
			r.PreviousResponseID = v.PreviousResponseID
		case *genai.GenOptionText:
			u, e := r.initOptionsText(v)
			unsupported = append(unsupported, u...)
			errs = append(errs, e...)
		case *genai.GenOptionTools:
			errs = append(errs, r.initOptionsTools(v)...)
		case *genai.GenOptionWeb:
			if v.Search {
				r.Tools = append(r.Tools, Tool{
					Type: "web_search",
					// SearchContextSize: "medium",
				})
				r.Include = []string{"web_search_call.action.sources"}
			}
			if v.Fetch {
				errs = append(errs, errors.New("unsupported GenOptionWeb.Fetch"))
			}
		default:
			return &base.ErrNotSupported{Options: []string{internal.TypeName(opt)}}
		}
	}
	if len(msgs) == 0 {
		return errors.New("no messages provided")
	}

	for i := range msgs {
		// Each "Message" in OpenAI responses API is a content.
		switch {
		case len(msgs[i].ToolCallResults) > 1:
			// Handle messages with multiple tool call results by creating multiple messages
			for j := range msgs[i].ToolCallResults {
				// Create a copy of the message with only one tool call result
				msgCopy := msgs[i]
				msgCopy.ToolCallResults = []genai.ToolCallResult{msgs[i].ToolCallResults[j]}
				var newMsg Message
				if skip, err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: tool call results #%d: %w", i, j, err))
				} else if !skip {
					r.Input = append(r.Input, newMsg)
				}
			}
		case len(msgs[i].Replies) > 1:
			// Goddam OpenAI. Handle messages with multiple tool calls by creating multiple messages.
			var txt []genai.Reply
			for j := range msgs[i].Replies {
				if !msgs[i].Replies[j].ToolCall.IsZero() {
					msgCopy := msgs[i]
					msgCopy.Replies = []genai.Reply{msgs[i].Replies[j]}
					var newMsg Message
					if skip, err := newMsg.From(&msgCopy); err != nil {
						errs = append(errs, fmt.Errorf("message #%d: tool call #%d: %w", i, j, err))
					} else if !skip {
						r.Input = append(r.Input, newMsg)
					}
				} else {
					txt = append(txt, msgs[i].Replies[j])
				}
			}
			if len(txt) != 0 {
				// Create a copy of the message with only the non-tool call messages.
				msgCopy := msgs[i]
				msgCopy.Replies = txt
				var newMsg Message
				if skip, err := newMsg.From(&msgCopy); err != nil {
					errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
				} else if !skip {
					r.Input = append(r.Input, newMsg)
				}
			}
		default:
			// It's a Request, send it as-is.
			var newMsg Message
			if skip, err := newMsg.From(&msgs[i]); err != nil {
				errs = append(errs, fmt.Errorf("message #%d: %w", i, err))
			} else if !skip {
				r.Input = append(r.Input, newMsg)
			}
		}
	}
	// If we have unsupported features but no other errors, return a structured error.
	if len(unsupported) > 0 && len(errs) == 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	return errors.Join(errs...)
}

// SetStream implements base.InitializableRequest.
func (r *Response) SetStream(stream bool) {
	r.Stream = stream
}

// ToResult implements base.ResultConverter.
func (r *Response) ToResult() (genai.Result, error) {
	res := genai.Result{
		Usage: genai.Usage{
			InputTokens:       r.Usage.InputTokens,
			InputCachedTokens: r.Usage.InputTokensDetails.CachedTokens,
			ReasoningTokens:   r.Usage.OutputTokensDetails.ReasoningTokens,
			OutputTokens:      r.Usage.OutputTokens,
			TotalTokens:       r.Usage.TotalTokens,
			ServiceTier:       string(r.ServiceTier),
		},
	}
	for oi := range r.Output {
		if err := r.Output[oi].To(&res.Message); err != nil {
			return res, err
		}
		for i := range r.Output[oi].Content {
			for j := range r.Output[oi].Content[i].Logprobs {
				res.Logprobs = append(res.Logprobs, r.Output[oi].Content[i].Logprobs[j].To())
			}
		}
	}
	var err error
	hasRefusal := false
	for oi := range r.Output {
		for i := range r.Output[oi].Content {
			if r.Output[oi].Content[i].Type == ContentRefusal {
				hasRefusal = true
			}
		}
	}
	switch {
	case r.IncompleteDetails.Reason != "":
		if r.IncompleteDetails.Reason == "max_output_tokens" {
			res.Usage.FinishReason = genai.FinishedLength
		}
		err = errors.New(r.IncompleteDetails.Reason)
	case hasRefusal:
		res.Usage.FinishReason = genai.FinishedContentFilter
	case slices.ContainsFunc(res.Replies, func(r genai.Reply) bool { return !r.ToolCall.IsZero() }):
		res.Usage.FinishReason = genai.FinishedToolCalls
	default:
		res.Usage.FinishReason = genai.FinishedStop
	}
	return res, err
}

func (r *Response) initOptionsText(v *genai.GenOptionText) ([]string, []error) {
	var unsupported []string
	var errs []error
	r.MaxOutputTokens = v.MaxTokens
	r.Temperature = v.Temperature
	r.TopP = v.TopP
	if v.SystemPrompt != "" {
		r.Instructions = v.SystemPrompt
	}
	if v.TopK != 0 {
		unsupported = append(unsupported, "GenOptionText.TopK")
	}
	if v.TopLogprobs > 0 {
		r.TopLogprobs = v.TopLogprobs
	}
	if len(v.Stop) != 0 {
		errs = append(errs, errors.New("unsupported option Stop"))
	}
	if v.DecodeAs != nil {
		r.Text.Format.Type = "json_schema"
		// OpenAI requires a name.
		r.Text.Format.Name = "response"
		r.Text.Format.Strict = true
		r.Text.Format.Schema = internal.JSONSchemaFor(reflect.TypeOf(v.DecodeAs))
	} else if v.ReplyAsJSON {
		r.Text.Format.Type = "json_object"
	}
	return unsupported, errs
}

func (r *Response) initOptionsTools(v *genai.GenOptionTools) []error {
	var errs []error
	if len(v.Tools) != 0 {
		r.ParallelToolCalls = true
		switch v.Force {
		case genai.ToolCallAny:
			r.ToolChoice = "auto"
		case genai.ToolCallRequired:
			r.ToolChoice = "required"
		case genai.ToolCallNone:
			r.ToolChoice = "none"
		}
		r.Tools = make([]Tool, len(v.Tools))
		for i, t := range v.Tools {
			if t.Name == "" {
				errs = append(errs, errors.New("tool name is required"))
			}
			r.Tools[i].Type = "function"
			r.Tools[i].Name = t.Name
			r.Tools[i].Description = t.Description
			if r.Tools[i].Parameters = t.InputSchemaOverride; r.Tools[i].Parameters == nil {
				r.Tools[i].Parameters = t.GetInputSchema()
			}
		}
	}
	return errs
}

// From must be called with at most one ToolCallResults.
func (m *Message) From(in *genai.Message) (bool, error) {
	if len(in.ToolCallResults) > 1 {
		return false, &internal.BadError{Err: errors.New("internal error")}
	}
	if len(in.ToolCallResults) != 0 {
		// Handle multiple tool call results by creating multiple messages
		// The caller (Init method) should handle this by creating separate messages
		m.Type = MessageFunctionCallOutput
		m.CallID = in.ToolCallResults[0].ID
		m.Output = in.ToolCallResults[0].Result
		return false, nil
	}
	if len(in.Requests) != 0 {
		m.Type = MessageMessage
		m.Role = "user"
		m.Content = make([]Content, len(in.Requests))
		for j := range in.Requests {
			if err := m.Content[j].FromRequest(&in.Requests[j]); err != nil {
				return false, fmt.Errorf("request #%d: %w", j, err)
			}
		}
		return len(m.Content) == 0, nil
	}
	if len(in.Replies) != 0 {
		// Handle multiple tool calls by creating multiple messages
		// The caller (Init method) should handle this by creating separate messages
		if !in.Replies[0].ToolCall.IsZero() {
			if len(in.Replies[0].ToolCall.Opaque) != 0 {
				return false, &internal.BadError{Err: errors.New("field ToolCall.Opaque not supported")}
			}
			m.Type = MessageFunctionCall
			m.CallID = in.Replies[0].ToolCall.ID
			m.Name = in.Replies[0].ToolCall.Name
			m.Arguments = in.Replies[0].ToolCall.Arguments
			return false, nil
		}
		m.Type = MessageMessage
		m.Role = "assistant"
		for j := range in.Replies {
			// TODO: should we send it back, at least the ID?
			if in.Replies[j].Reasoning != "" {
				continue
			}
			m.Content = append(m.Content, Content{})
			if err := m.Content[len(m.Content)-1].FromReply(&in.Replies[j]); err != nil {
				return false, fmt.Errorf("reply #%d: %w", j, err)
			}
		}
		return len(m.Content) == 0, nil
	}
	return false, &internal.BadError{Err: fmt.Errorf("implement message: %#v", in)}
}

// To is different here because it can be called multiple times on the same out.
//
// In the Responses API, Message is actually a mix of Message and Content.
func (m *Message) To(out *genai.Message) error {
	// We only need to implement the types that can be returned from the LLM.
	switch m.Type {
	case MessageMessage:
		for i := range m.Content {
			replies, err := m.Content[i].To()
			if err != nil {
				return fmt.Errorf("reply %d: %w", i, err)
			}
			out.Replies = append(out.Replies, replies...)
		}
	case MessageReasoning:
		for i := range m.Summary {
			if m.Summary[i].Type != "summary_text" {
				return &internal.BadError{Err: fmt.Errorf("implement summary type %q", m.Summary[i].Type)}
			}
			out.Replies = append(out.Replies, genai.Reply{Reasoning: m.Summary[i].Text})
		}
	case MessageFunctionCall:
		out.Replies = append(out.Replies, genai.Reply{ToolCall: genai.ToolCall{ID: m.CallID, Name: m.Name, Arguments: m.Arguments}})
	case MessageWebSearchCall:
		if m.Action.Type != "search" {
			return &internal.BadError{Err: fmt.Errorf("implement action type %q", m.Action.Type)}
		}
		c := genai.Citation{Sources: make([]genai.CitationSource, len(m.Action.Sources)+1)}
		c.Sources[0].Type = genai.CitationWebQuery
		c.Sources[0].Snippet = m.Action.Query
		for i, src := range m.Action.Sources {
			c.Sources[i+1].Type = genai.CitationWeb
			c.Sources[i+1].URL = src.URL
		}
		out.Replies = append(out.Replies, genai.Reply{Citation: c})
	case MessageFileSearchCall:
		for _, q := range m.Queries {
			out.Replies = append(out.Replies, genai.Reply{Citation: genai.Citation{
				Sources: []genai.CitationSource{{Type: genai.CitationWebQuery, Snippet: q}},
			}})
		}
		for _, r := range m.Results {
			out.Replies = append(out.Replies, genai.Reply{Citation: genai.Citation{
				CitedText: r.Text,
				Sources: []genai.CitationSource{{
					Type:  genai.CitationDocument,
					ID:    r.FileID,
					Title: r.Filename,
				}},
			}})
		}
	case MessageComputerCall, MessageImageGenerationCall, MessageCodeInterpreterCall, MessageLocalShellCall, MessageMcpListTools, MessageMcpApprovalRequest, MessageMcpCall, MessageComputerCallOutput, MessageFunctionCallOutput, MessageLocalShellCallOutput, MessageMcpApprovalResponse, MessageItemReference:
		return &internal.BadError{Err: fmt.Errorf("unsupported output type %q", m.Type)}
	default:
		return &internal.BadError{Err: fmt.Errorf("unsupported output type %q", m.Type)}
	}
	return nil
}

// To converts to the genai equivalent.
func (c *Content) To() ([]genai.Reply, error) {
	var out []genai.Reply
	for _, a := range c.Annotations {
		var ci genai.Citation
		switch a.Type {
		case "url_citation":
			ci = genai.Citation{
				StartIndex: a.StartIndex,
				EndIndex:   a.EndIndex,
				Sources:    []genai.CitationSource{{Type: genai.CitationWeb, URL: a.URL, Title: a.Title}},
			}
		case "file_citation", "container_file_citation":
			ci = genai.Citation{
				StartIndex: a.StartIndex,
				EndIndex:   a.EndIndex,
				Sources:    []genai.CitationSource{{Type: genai.CitationDocument, ID: a.FileID}},
			}
		case "file_path":
			ci = genai.Citation{
				Sources: []genai.CitationSource{{Type: genai.CitationDocument, ID: a.FileID}},
			}
		default:
			return out, &internal.BadError{Err: fmt.Errorf("unsupported annotation type %q", a.Type)}
		}
		out = append(out, genai.Reply{Citation: ci})
	}
	switch c.Type {
	case ContentOutputText:
		out = append(out, genai.Reply{Text: c.Text})
	case ContentRefusal:
		// Surface refusal as text so the caller can see the reason.
		out = append(out, genai.Reply{Text: c.Refusal})
	case ContentInputText, ContentInputImage, ContentInputFile:
		return out, &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
	default:
		return out, &internal.BadError{Err: fmt.Errorf("implement content type %q", c.Type)}
	}
	return out, nil
}

// FromRequest converts from a genai request.
func (c *Content) FromRequest(in *genai.Request) error {
	if in.Text != "" {
		c.Type = ContentInputText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		// OpenAI require a mime-type to determine if image, sound or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentInputImage
			c.Detail = "auto" // TODO: Make it configurable.
			if in.Doc.URL == "" {
				c.ImageURL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			// OpenAI responses API doesn't support text documents as attachment.
			c.Type = ContentInputText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			if in.Doc.URL != "" {
				return fmt.Errorf("URL to %s file not supported", mimeType)
			}
			filename := in.Doc.GetFilename()
			if filename == "" {
				exts, err := mime.ExtensionsByType(mimeType)
				if err != nil {
					return err
				}
				if len(exts) == 0 {
					return fmt.Errorf("unknown extension for mime type %s", mimeType)
				}
				filename = "content" + exts[0]
			}
			c.Type = ContentInputFile
			c.Filename = filename
			c.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
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
	if in.Text != "" {
		c.Type = ContentInputText
		c.Text = in.Text
		return nil
	}
	if !in.Doc.IsZero() {
		// https://platform.openai.com/docs/guides/images?api-mode=chat&format=base64-encoded#image-input-requirements
		mimeType, data, err := in.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return err
		}
		// OpenAI require a mime-type to determine if image, sound or PDF.
		if mimeType == "" {
			return fmt.Errorf("unspecified mime type for URL %q", in.Doc.URL)
		}
		switch {
		case strings.HasPrefix(mimeType, "image/"):
			c.Type = ContentInputImage
			c.Detail = "auto" // TODO: Make it configurable.
			if in.Doc.URL == "" {
				c.ImageURL = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
			} else {
				c.ImageURL = in.Doc.URL
			}
			// text/plain, text/markdown
		case strings.HasPrefix(mimeType, "text/"):
			// OpenAI responses API doesn't support text documents as attachment.
			c.Type = ContentInputText
			if in.Doc.URL != "" {
				return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
			}
			c.Text = string(data)
		default:
			if in.Doc.URL != "" {
				return fmt.Errorf("URL to %s file not supported", mimeType)
			}
			filename := in.Doc.GetFilename()
			if filename == "" {
				exts, err := mime.ExtensionsByType(mimeType)
				if err != nil {
					return err
				}
				if len(exts) == 0 {
					return fmt.Errorf("unknown extension for mime type %s", mimeType)
				}
				filename = "content" + exts[0]
			}
			c.Type = ContentInputFile
			c.Filename = filename
			c.FileData = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
		}
		return nil
	}
	return &internal.BadError{Err: errors.New("unknown Reply type")}
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// To converts to the genai equivalent.
func (l *Logprobs) To() []genai.Logprob {
	out := make([]genai.Logprob, 1, len(l.TopLogprobs)+1)
	// Intentionally discard Bytes.
	out[0] = genai.Logprob{Text: l.Token, Logprob: l.Logprob}
	for _, tlp := range l.TopLogprobs {
		out = append(out, genai.Logprob{Text: tlp.Token, Logprob: tlp.Logprob})
	}
	return out
}

func (e *ErrorResponse) Error() string {
	return fmt.Sprintf("%s (type: %s, code: %s)", e.ErrorVal.Message, e.ErrorVal.Type, e.ErrorVal.Code)
}

// IsAPIError implements base.ErrorResponseI.
func (e *ErrorResponse) IsAPIError() bool {
	return true
}

//

// Client is a client for the OpenAI Responses API.
type Client struct {
	base.NotImplemented
	impl base.Provider[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]
}

// New creates a new client to talk to the OpenAI Responses API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the OPENAI_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://platform.openai.com/settings/organization/api-keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://platform.openai.com/docs/models
//
// # Documents
//
// OpenAI supports many types of documents, listed at
// https://platform.openai.com/docs/assistants/tools/file-search#supported-files
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model string
	var modalities genai.Modalities
	var preloadedModels []genai.Model
	var wrapper func(http.RoundTripper) http.RoundTripper
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
	const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("OPENAI_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "OPENAI_API_KEY", URL: apiKeyURL}
		}
	}
	switch len(modalities) {
	case 0:
		// Auto-detect below.
	case 1:
		switch modalities[0] {
		case genai.ModalityAudio, genai.ModalityImage, genai.ModalityText, genai.ModalityVideo:
		case genai.ModalityDocument:
			return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image or text are supported", modalities)
		default:
			return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image or text are supported", modalities)
		}
	default:
		return nil, fmt.Errorf("unexpected option Modalities %s, only audio, image or text are supported", modalities)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		impl: base.Provider[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]{
			GenSyncURL:      "https://api.openai.com/v1/responses",
			GenStreamURL:    "https://api.openai.com/v1/responses",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProcessHeaders:  processHeaders,
			ProviderBase: base.ProviderBase[*ErrorResponse]{
				APIKeyURL: "", // OpenAI error message prints the api key URL already.
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
			var mod genai.Modality
			switch len(modalities) {
			case 0:
				mod = genai.ModalityText
			case 1:
				mod = modalities[0]
			default:
				// TODO: Maybe it's possible, need to double check.
				return nil, fmt.Errorf("can't use model %s with option Modalities %s", model, modalities)
			}
			switch mod {
			case genai.ModalityText:
				if c.impl.Model, err = c.selectBestTextModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityImage:
				if c.impl.Model, err = c.selectBestImageModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityVideo:
				if c.impl.Model, err = c.selectBestVideoModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityAudio:
				return nil, errors.New("OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio")
			case genai.ModalityDocument:
				// TODO: Implement document modality model selection.
				return nil, fmt.Errorf("automatic model selection is not implemented yet for modality %s (send PR to add support)", modalities)
			default:
				return nil, fmt.Errorf("automatic model selection is not implemented yet for modality %s (send PR to add support)", modalities)
			}
		default:
			c.impl.Model = model
			switch len(modalities) {
			case 0:
				c.impl.OutputModalities, err = c.detectModelModalities(ctx, model)
			case 1:
				c.impl.OutputModalities = modalities
			default:
				// TODO: Maybe it's possible, need to double check.
				return nil, fmt.Errorf("can't use model %s with option Modalities %s", model, modalities)
			}
		}
	}
	return c, err
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "openairesponses"
}

// GenSyncRaw provides access to the raw API.
func (c *Client) GenSyncRaw(ctx context.Context, in, out *Response) error {
	// Check if audio output was requested
	if len(c.impl.OutputModalities) > 0 && c.impl.OutputModalities[0] == genai.ModalityAudio {
		return errors.New("OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio")
	}
	return c.impl.GenSyncRaw(ctx, in, out)
}

// GenStreamRaw provides access to the raw API.
func (c *Client) GenStreamRaw(ctx context.Context, in *Response) (iter.Seq[ResponseStreamChunkResponse], func() error) {
	// Check if audio output was requested
	if len(c.impl.OutputModalities) > 0 && c.impl.OutputModalities[0] == genai.ModalityAudio {
		return func(yield func(ResponseStreamChunkResponse) bool) {}, func() error {
			return errors.New("OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio")
		}
	}
	return c.impl.GenStreamRaw(ctx, in)
}

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{GenAsync: true}
}

// GenAsync implements genai.Provider.
//
// It uses the OpenAI Responses API background mode to submit a request that is processed asynchronously.
// The returned Job is the response ID that can be polled with PokeResult.
//
// https://platform.openai.com/docs/api-reference/responses/create
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Job, error) {
	if err := c.impl.Validate(); err != nil {
		return "", err
	}
	req := Response{}
	if err := req.Init(msgs, c.impl.Model, opts...); err != nil {
		return "", err
	}
	req.Background = true
	resp := Response{}
	if err := c.impl.DoRequest(ctx, "POST", c.impl.GenSyncURL, &req, &resp); err != nil {
		return "", err
	}
	if resp.ID == "" {
		return "", errors.New("no response ID returned")
	}
	return genai.Job(resp.ID), nil
}

// PokeResult implements genai.Provider.
//
// It polls the status of a background response by its ID.
//
// https://platform.openai.com/docs/api-reference/responses/get
func (c *Client) PokeResult(ctx context.Context, job genai.Job) (genai.Result, error) {
	resp := Response{}
	url := c.impl.GenSyncURL + "/" + string(job)
	if err := c.impl.DoRequest(ctx, "GET", url, nil, &resp); err != nil {
		return genai.Result{}, err
	}
	switch resp.Status {
	case "queued", "in_progress":
		return genai.Result{Usage: genai.Usage{FinishReason: genai.Pending}}, nil
	case "incomplete":
		res, err := resp.ToResult()
		if err != nil {
			return res, err
		}
		return res, errors.New(resp.IncompleteDetails.Reason)
	case "failed":
		return genai.Result{}, &resp.Error
	case "completed":
		return resp.ToResult()
	default:
		return genai.Result{}, fmt.Errorf("unexpected response status %q", resp.Status)
	}
}

// PokeResultRaw provides raw access to poll a background response.
func (c *Client) PokeResultRaw(ctx context.Context, job genai.Job) (*Response, error) {
	resp := &Response{}
	url := c.impl.GenSyncURL + "/" + string(job)
	if err := c.impl.DoRequest(ctx, "GET", url, nil, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

// ProcessStream converts the raw packets from the streaming API into Reply fragments.
func ProcessStream(chunks iter.Seq[ResponseStreamChunkResponse]) (iter.Seq[genai.Reply], func() (genai.Usage, [][]genai.Logprob, error)) {
	var finalErr error
	u := genai.Usage{}
	var l [][]genai.Logprob

	return func(yield func(genai.Reply) bool) {
			refused := false
			pendingToolCall := genai.ToolCall{}
			for pkt := range chunks {
				f := genai.Reply{}
				for _, lp := range pkt.Logprobs {
					l = append(l, lp.To())
				}
				switch pkt.Type {
				case ResponseCreated, ResponseInProgress:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/created
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/in_progress
					// Both are useless.
				case ResponseCompleted:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/completed
					// This message contains all the data duplicated. :( It's clear they never thought about efficiency.
					u.InputTokens = pkt.Response.Usage.InputTokens
					u.InputCachedTokens = pkt.Response.Usage.InputTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Response.Usage.OutputTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Response.Usage.OutputTokens
					u.ServiceTier = string(pkt.Response.ServiceTier)
					if len(pkt.Response.Output) == 0 {
						// TODO: Likely failed.
						finalErr = &internal.BadError{Err: fmt.Errorf("no output: %#v", pkt)}
						return
					}
					u.FinishReason = genai.FinishedStop
					for i := range pkt.Response.Output {
						msg := pkt.Response.Output[i]
						switch msg.Status {
						case "":
						case "completed":
							if msg.Type == MessageFunctionCall {
								u.FinishReason = genai.FinishedToolCalls
							}
						case "in_progress":
						case "incomplete":
							u.FinishReason = genai.FinishedLength
						case "failed":
							finalErr = fmt.Errorf("failed: %#v", pkt)
							return
						default:
							finalErr = &internal.BadError{Err: fmt.Errorf("unknown status %q: %#v", msg.Status, pkt)}
							return
						}
					}

				case ResponseFailed:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/failed
					finalErr = &internal.BadError{Err: &pkt.Response.Error}
					return
				case ResponseError:
					// https://platform.openai.com/docs/api-reference/responses_streaming/error
					// This happens fairly consistently with gpt-5-nano_thinking/GenStream-Tools-ToolBias-Canada.yaml.
					// Normally this should be a BadError but that would break the smoke test.
					finalErr = &pkt.ErrorResponse
					return
				case ResponseIncomplete:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/incomplete
					u.InputTokens = pkt.Response.Usage.InputTokens
					u.InputCachedTokens = pkt.Response.Usage.InputTokensDetails.CachedTokens
					u.ReasoningTokens = pkt.Response.Usage.OutputTokensDetails.ReasoningTokens
					u.OutputTokens = pkt.Response.Usage.OutputTokens
					if pkt.Response.IncompleteDetails.Reason == "max_output_tokens" {
						u.FinishReason = genai.FinishedLength
					}
					finalErr = errors.New(pkt.Response.IncompleteDetails.Reason)
					return

				case ResponseOutputItemAdded:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/output_item/added
					switch pkt.Item.Type {
					case MessageMessage:
						// Unnecessary.
					case MessageFunctionCall:
						pendingToolCall.Name = pkt.Item.Name
						pendingToolCall.ID = pkt.Item.CallID
					case MessageReasoning:
						var bits []string
						for i := range pkt.Item.Summary {
							bits = append(bits, pkt.Item.Summary[i].Text)
						}
						f.Reasoning = strings.Join(bits, "")
					case MessageWebSearchCall:
						// TODO: Send a fragment to tell the user. It's a server-side tool call, we don't have infrastructure
						// to surface that to the user yet.
					case MessageFileSearchCall:
						// Server-side file search; data arrives in ResponseOutputItemDone.
					case MessageComputerCall, MessageImageGenerationCall, MessageCodeInterpreterCall, MessageLocalShellCall, MessageMcpListTools, MessageMcpApprovalRequest, MessageMcpCall, MessageComputerCallOutput, MessageFunctionCallOutput, MessageLocalShellCallOutput, MessageMcpApprovalResponse, MessageItemReference:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement item: %q", pkt.Item.Type)}
						return
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement item: %q", pkt.Item.Type)}
						return
					}
				case ResponseOutputItemDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/output_item/done
					// Perfect place to handle web search, since the "pending" has no data.
					switch pkt.Item.Type {
					case MessageWebSearchCall:
						// TODO: Check for pkt.Item.Status == "completed"
						switch pkt.Item.Action.Type {
						case "search":
							f.Citation.Sources = make([]genai.CitationSource, len(pkt.Item.Action.Sources)+1)
							f.Citation.Sources[0].Type = genai.CitationWebQuery
							f.Citation.Sources[0].Snippet = pkt.Item.Action.Query
							for i, src := range pkt.Item.Action.Sources {
								f.Citation.Sources[i+1].Type = genai.CitationWeb
								f.Citation.Sources[i+1].URL = src.URL
							}
						default:
							finalErr = &internal.BadError{Err: fmt.Errorf("implement action type %q", pkt.Item.Action.Type)}
							return
						}
					case MessageFileSearchCall:
						// File search completed; yield results as citations.
						for _, r := range pkt.Item.Results {
							if !yield(genai.Reply{Citation: genai.Citation{
								CitedText: r.Text,
								Sources: []genai.CitationSource{{
									Type:  genai.CitationDocument,
									ID:    r.FileID,
									Title: r.Filename,
								}},
							}}) {
								return
							}
						}
					case MessageMessage, MessageComputerCall, MessageFunctionCall, MessageReasoning, MessageImageGenerationCall, MessageCodeInterpreterCall, MessageLocalShellCall, MessageMcpListTools, MessageMcpApprovalRequest, MessageMcpCall, MessageComputerCallOutput, MessageFunctionCallOutput, MessageLocalShellCallOutput, MessageMcpApprovalResponse, MessageItemReference:
					default:
						// The default stance is to ignore this event since it's generally duplicate information.
					}
				case ResponseContentPartAdded:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/content_part/added
					switch pkt.Part.Type {
					case ContentOutputText:
						if len(pkt.Part.Annotations) > 0 {
							finalErr = &internal.BadError{Err: fmt.Errorf("implement citations: %#v", pkt.Part.Annotations)}
							return
						}
						if pkt.Part.Text != "" {
							finalErr = &internal.BadError{Err: fmt.Errorf("unexpected text: %q", pkt.Part.Text)}
							return
						}
					case ContentRefusal:
						// Refusal content part; the actual text arrives via ResponseRefusalDelta.
						refused = true
					case ContentInputText, ContentInputImage, ContentInputFile:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement part: %q", pkt.Part.Type)}
						return
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement part: %q", pkt.Part.Type)}
						return
					}
				case ResponseContentPartDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/content_part/done
					// Unnecessary, as we already streamed the content in ResponseContentPartAdded.
				case ResponseOutputTextDelta:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/output_text/delta
					f.Text = pkt.Delta
				case ResponseOutputTextDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/output_text/done
					// Unnecessary, we captured the text via ResponseOutputTextDelta.

				case ResponseRefusalDelta:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/refusal/delta
					// Surface refusal text so the caller can see the reason.
					f.Text = pkt.Delta
					refused = true
				case ResponseRefusalDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/refusal/done
					refused = true

				case ResponseFunctionCallArgumentsDelta:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/function_call_arguments/delta
					// Unnecessary. The content is sent in ResponseFunctionCallArgumentsDone.
				case ResponseFunctionCallArgumentsDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/function_call_arguments/done
					pendingToolCall.Arguments = pkt.Arguments
					f.ToolCall = pendingToolCall
					pendingToolCall = genai.ToolCall{}

				case ResponseFileSearchCallInProgress, ResponseFileSearchCallSearching:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/file_search_call/in_progress
					// Data is sent in ResponseOutputItemDone.
				case ResponseFileSearchCallCompleted:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/file_search_call/completed
					// Data is sent in ResponseOutputItemDone.

				case ResponseWebSearchCallInProgress:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/web_search_call/in_progress
					// Data is sent in ResponseOutputItemDone.
				case ResponseWebSearchCallSearching:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/web_search_call/searching
					// Data is sent in ResponseOutputItemDone.
				case ResponseWebSearchCallCompleted:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/web_search_call/completed
					// Data is sent in ResponseOutputItemDone.

				case ResponseReasoningSummaryPartAdded:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/reasoning_summary_part/added
				case ResponseReasoningSummaryPartDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/reasoning_summary_part/done
				case ResponseReasoningSummaryTextDelta:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/reasoning_summary_text/delta
					f.Reasoning = pkt.Delta
				case ResponseReasoningSummaryTextDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/reasoning_summary_text/done
				case ResponseReasoningTextDelta:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/reasoning_text/delta
					// I'm not sure it will ever happen.
					f.Reasoning = pkt.Delta
				case ResponseReasoningTextDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/reasoning_text/done

				case ResponseImageGenerationCallCompleted, ResponseImageGenerationCallGenerating, ResponseImageGenerationCallInProgress, ResponseImageGenerationCallPartialImage:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/image_generation_call/completed
					finalErr = &internal.BadError{Err: fmt.Errorf("implement packet: %q", pkt.Type)}
					return

				case ResponseMCPCallArgumentsDelta, ResponseMCPCallArgumentsDone, ResponseMCPCallCompleted, ResponseMCPCallFailed, ResponseMCPCallInProgress, ResponseMCPListToolsCompleted, ResponseMCPListToolsFailed, ResponseMCPListToolsInProgress:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/mcp_call_arguments/delta
					finalErr = &internal.BadError{Err: fmt.Errorf("implement packet: %q", pkt.Type)}
					return

				case ResponseCodeInterpreterCallInterpreting, ResponseCodeInterpreterCallCompleted, ResponseCodeInterpreterCallDelta, ResponseCodeInterpreterCallDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/code_interpreter_call/in_progress
					finalErr = &internal.BadError{Err: fmt.Errorf("implement packet: %q", pkt.Type)}
					return

				case ResponseOutputTextAnnotationAdded:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/output_text/annotation/added
					switch pkt.Annotation.Type {
					case "url_citation":
						f.Citation.StartIndex = pkt.Annotation.StartIndex
						f.Citation.EndIndex = pkt.Annotation.EndIndex
						f.Citation.Sources = []genai.CitationSource{
							{Type: genai.CitationWeb, URL: pkt.Annotation.URL, Title: pkt.Annotation.Title},
						}
					case "file_citation", "container_file_citation":
						f.Citation.StartIndex = pkt.Annotation.StartIndex
						f.Citation.EndIndex = pkt.Annotation.EndIndex
						f.Citation.Sources = []genai.CitationSource{
							{Type: genai.CitationDocument, ID: pkt.Annotation.FileID},
						}
					case "file_path":
						f.Citation.Sources = []genai.CitationSource{
							{Type: genai.CitationDocument, ID: pkt.Annotation.FileID},
						}
					default:
						finalErr = &internal.BadError{Err: fmt.Errorf("implement annotation type: %q", pkt.Annotation.Type)}
						return
					}
				case ResponseQueued:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/queued
					// Ignore.

				case ResponseCustomToolCallInputDelta, ResponseCustomToolCallInputDone:
					// https://platform.openai.com/docs/api-reference/responses_streaming/response/custom_tool_call_input/delta
					finalErr = &internal.BadError{Err: fmt.Errorf("implement packet: %q", pkt.Type)}
					return

				default:
					finalErr = &internal.BadError{Err: fmt.Errorf("implement packet: %q", pkt.Type)}
					return
				}
				if !yield(f) {
					return
				}
			}
			if !pendingToolCall.IsZero() {
				finalErr = &internal.BadError{Err: errors.New("unexpected pending tool call")}
				return
			}
			if refused {
				u.FinishReason = genai.FinishedContentFilter
			}
		}, func() (genai.Usage, [][]genai.Logprob, error) {
			return u, l, finalErr
		}
}

var _ genai.Provider = &Client{}
