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
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"os"
	"strings"

	"github.com/maruel/roundtrippers"
	"golang.org/x/net/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/providers/openaibase"
	"github.com/maruel/genai/scoreboard"
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

// Client is a client for the OpenAI Responses API.
type Client struct {
	base.NotImplemented
	impl   base.Provider[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]
	shared openaibase.Client

	baseURL string
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
	var apiKey, model, remote string
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
		case genai.ProviderOptionRemote:
			remote = string(v)
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://platform.openai.com/settings/organization/api-keys"
	var err error
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_API_KEY")
		if apiKey == "" && wrapper == nil {
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
	baseURL := "https://api.openai.com/v1"
	if remote != "" {
		baseURL = strings.TrimRight(remote, "/")
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	c := &Client{
		baseURL: baseURL,
		impl: base.Provider[*ErrorResponse, *Response, *Response, ResponseStreamChunkResponse]{
			GenSyncURL:      baseURL + "/responses",
			GenStreamURL:    baseURL + "/responses",
			ProcessStream:   ProcessStream,
			PreloadedModels: preloadedModels,
			ProcessHeaders:  openaibase.ProcessHeaders,
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
	c.shared = openaibase.Client{
		Impl:            &c.impl.ProviderBase,
		BaseURL:         baseURL,
		PreloadedModels: preloadedModels,
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
				if c.impl.Model, err = c.shared.SelectBestTextModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityImage:
				if c.impl.Model, err = c.shared.SelectBestImageModel(ctx, model); err != nil {
					return nil, err
				}
				c.impl.OutputModalities = genai.Modalities{mod}
			case genai.ModalityVideo:
				if c.impl.Model, err = c.shared.SelectBestVideoModel(ctx, model); err != nil {
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
				c.impl.OutputModalities, err = c.shared.DetectModelModalities(ctx, model)
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
					u.FinishReason = genai.FinishedStop
					if len(pkt.Response.Output) == 0 {
						// Some backends (e.g. ChatGPT) don't include output
						// in the completed event; content was already streamed
						// via delta events.
						break
					}
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
				case ResponseKeepalive:
					// Heartbeat sent during long reasoning to keep the connection alive.
					// Ignore.
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

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	return c.shared.ListModels(ctx)
}

// GenSync implements genai.Provider.
//
// It handles delta detection: if msgs contains metadata from a prior call (via Reply.Opaque),
// only new messages are sent. The response ID is captured and emitted as metadata for the next call.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	if c.shared.IsAudio() {
		return genai.Result{}, errors.New("OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio")
	}
	if c.shared.IsImage() || c.shared.IsVideo() {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.shared.GenDoc(ctx, &msgs[0], opts...)
	}
	cleaned, prevRespID := c.prepareDelta(msgs, opts)
	in := &Response{}
	if err := in.Init(cleaned, c.impl.Model, opts...); err != nil {
		return genai.Result{}, err
	}
	in.PreviousResponseID = prevRespID
	out := &Response{}
	if err := c.GenSyncRaw(ctx, in, out); err != nil {
		return genai.Result{}, err
	}
	lastResp := c.impl.LastResponseHeaders()
	res, err := out.ToResult()
	if err != nil {
		return res, err
	}
	if err := res.Validate(); err != nil {
		return res, &internal.BadError{Err: err}
	}
	if c.impl.ProcessHeaders != nil && lastResp != nil {
		res.Usage.Limits = c.impl.ProcessHeaders(lastResp)
	}
	if out.ID != "" {
		res.Replies = append(res.Replies, emitMeta(out.ID, len(msgs)))
	}
	return res, nil
}

// GenStream implements genai.Provider.
//
// It handles delta detection: if msgs contains metadata from a prior call (via Reply.Opaque),
// only new messages are sent. The response ID is captured and emitted as metadata for the next call.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if c.shared.IsAudio() {
		return func(yield func(genai.Reply) bool) {}, func() (genai.Result, error) {
			return genai.Result{}, errors.New("OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio")
		}
	}
	if c.shared.IsImage() || c.shared.IsVideo() {
		return base.SimulateStream(ctx, c, msgs, opts...)
	}
	cleaned, prevRespID := c.prepareDelta(msgs, opts)
	in := &Response{}
	if err := in.Init(cleaned, c.impl.Model, opts...); err != nil {
		return func(yield func(genai.Reply) bool) {}, func() (genai.Result, error) {
			return genai.Result{}, err
		}
	}
	in.PreviousResponseID = prevRespID
	chunks, finish := c.GenStreamRaw(ctx, in)
	// Capture headers immediately after the HTTP call, before iterating.
	lastResp := c.impl.LastResponseHeaders()
	var respID string
	filtered := streamWithRespID(chunks, &respID)
	fragments, finish2 := ProcessStream(filtered)
	res := genai.Result{}
	var finalErr error
	msgCount := len(msgs)

	fnFragments := func(yield func(genai.Reply) bool) {
		sent := false
		for f := range fragments {
			if f.IsZero() {
				continue
			}
			if err := f.Validate(); err != nil {
				finalErr = &internal.BadError{Err: err}
				break
			}
			if err := res.Accumulate(&f); err != nil {
				finalErr = &internal.BadError{Err: err}
				break
			}
			sent = true
			if !yield(f) {
				break
			}
		}
		if err := finish(); finalErr == nil {
			finalErr = err
		}
		usage, _, err := finish2()
		res.Usage = usage
		if finalErr == nil {
			finalErr = err
		}
		if !sent && finalErr == nil {
			finalErr = errors.New("model sent no reply")
		}
	}
	fnFinish := func() (genai.Result, error) {
		if finalErr != nil {
			return res, finalErr
		}
		if err := res.Validate(); err != nil {
			return res, &internal.BadError{Err: err}
		}
		if c.impl.ProcessHeaders != nil && lastResp != nil {
			res.Usage.Limits = c.impl.ProcessHeaders(lastResp)
		}
		if respID != "" {
			res.Replies = append(res.Replies, emitMeta(respID, msgCount))
		}
		return res, nil
	}
	return fnFragments, fnFinish
}

// WebSocket opens a persistent WebSocket connection to the OpenAI Responses API.
//
// The returned connection inherits the client's model, API key, and base URL.
// Call Close() when done.
func (c *Client) WebSocket(ctx context.Context) (*WebSocketConn, error) {
	if c.impl.Model == "" {
		return nil, errors.New("a model is required")
	}
	// Derive WebSocket URL from the client's base URL.
	wsURL := strings.Replace(c.baseURL, "https://", "wss://", 1)
	wsURL = strings.Replace(wsURL, "http://", "ws://", 1)
	wsURL += "/responses"

	wsCfg, err := websocket.NewConfig(wsURL, wsURL)
	if err != nil {
		return nil, fmt.Errorf("failed to create websocket config: %w", err)
	}
	// Extract auth headers from the HTTP client's transport chain.
	wsCfg.Header = http.Header{}
	wsCfg.Header.Set("OpenAI-Beta", "responses=v1")
	if h, ok := c.impl.Client.Transport.(*roundtrippers.Header); ok {
		for k, vs := range h.Header {
			for _, v := range vs {
				wsCfg.Header.Set(k, v)
			}
		}
	}
	raw, err := wsCfg.DialContext(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to websocket %s: %w", wsURL, err)
	}
	return &WebSocketConn{
		client: c,
		ws:     &websocketConn{raw},
	}, nil
}

// Opaque keys for session metadata stored in Reply.Opaque between calls.
const (
	opaqueResponseID = "response_id"
	opaqueSentMsgs   = "sent_msgs"
)

// prepareDelta resolves the delta messages and previous_response_id for GenSync/GenStream.
//
// It inspects msgs for session metadata (via Reply.Opaque) and GenOptionText for an explicit
// PreviousResponseID (which takes precedence). When metadata is found, only unsent messages are
// returned and Opaque-only bookkeeping replies are stripped.
func (c *Client) prepareDelta(msgs genai.Messages, opts []genai.GenOption) (cleaned genai.Messages, prevRespID string) {
	sentMsgs, respID := findPrevMeta(msgs)
	// An explicit PreviousResponseID in opts takes precedence over Opaque metadata.
	if explicitID := getExplicitPrevID(opts); explicitID != "" {
		respID = explicitID
	}
	if respID != "" {
		cleaned = deltaMessages(msgs, sentMsgs)
		return cleaned, respID
	}
	return msgs, ""
}

// getExplicitPrevID extracts an explicit PreviousResponseID from GenOptionText options.
func getExplicitPrevID(opts []genai.GenOption) string {
	for _, opt := range opts {
		if t, ok := opt.(*GenOptionText); ok && t.PreviousResponseID != "" {
			return t.PreviousResponseID
		}
	}
	return ""
}

// findPrevMeta searches msgs for session metadata stored directly in Reply.Opaque.
func findPrevMeta(msgs genai.Messages) (sentMsgs int, respID string) {
	for i := len(msgs) - 1; i >= 0; i-- {
		for j := range msgs[i].Replies {
			if msgs[i].Replies[j].Opaque != nil {
				if id, ok := msgs[i].Replies[j].Opaque[opaqueResponseID].(string); ok && id != "" {
					respID = id
				}
				if n, ok := msgs[i].Replies[j].Opaque[opaqueSentMsgs].(float64); ok {
					sentMsgs = int(n)
				}
				if respID != "" {
					return
				}
			}
		}
	}
	return
}

// deltaMessages computes the delta from msgs based on sentMsgs, and strips
// Opaque-only replies that Init/From rejects.
func deltaMessages(msgs genai.Messages, sentMsgs int) genai.Messages {
	var delta genai.Messages
	if sentMsgs <= len(msgs) {
		delta = msgs[sentMsgs:]
	} else {
		delta = msgs
	}
	cleaned := make(genai.Messages, 0, len(delta))
	for i := range delta {
		m := delta[i]
		if len(m.Replies) > 0 {
			filtered := make([]genai.Reply, 0, len(m.Replies))
			for j := range m.Replies {
				r := m.Replies[j]
				if len(r.Opaque) > 0 {
					if r.Text == "" && r.ToolCall.IsZero() && r.Reasoning == "" && r.Doc.IsZero() && r.Citation.IsZero() {
						// Drop Opaque-only bookkeeping replies.
						continue
					}
					// Strip Opaque metadata so FromReply does not reject it.
					r.Opaque = nil
				}
				filtered = append(filtered, r)
			}
			if len(filtered) == 0 && len(m.Requests) == 0 && len(m.ToolCallResults) == 0 {
				continue
			}
			m.Replies = filtered
		}
		cleaned = append(cleaned, m)
	}
	return cleaned
}

// emitMeta returns a Reply carrying session metadata for delta tracking.
func emitMeta(respID string, msgCount int) genai.Reply {
	return genai.Reply{Opaque: map[string]any{
		opaqueResponseID: respID,
		opaqueSentMsgs:   float64(msgCount),
	}}
}

// streamWithRespID returns a filtered iterator that passes through all events from src
// while capturing the response ID from terminal events (ResponseCompleted, ResponseFailed,
// ResponseIncomplete).
func streamWithRespID(src iter.Seq[ResponseStreamChunkResponse], respID *string) iter.Seq[ResponseStreamChunkResponse] {
	return func(yield func(ResponseStreamChunkResponse) bool) {
		for pkt := range src {
			switch pkt.Type {
			case ResponseCompleted, ResponseFailed, ResponseIncomplete:
				if pkt.Response.ID != "" {
					*respID = pkt.Response.ID
				}
			default:
			}
			if !yield(pkt) {
				return
			}
		}
	}
}

var _ genai.Provider = &Client{}
