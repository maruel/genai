// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package acp implements a client for the Agent Client Protocol (ACP).
//
// The CLI providers that launch an ACP agent subprocess use it, and other ACP
// clients can reuse it.
//
// It performs the JSON-RPC 2.0 handshake (initialize → session/new or
// session/resume), applies session configuration options, sends
// session/prompt, and reads session/update notifications until the prompt
// response arrives. Providers own the subprocess and pass its stdin and stdout
// to Handshake; they translate their model IDs to configuration values and
// decide which options to expose.
//
// See https://agentclientprotocol.com for the specification.
package acp

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/url"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/msgutil"
)

// SessionIDKey is the key used in Reply.Opaque to carry session IDs.
const SessionIDKey = "session_id"

// Options configures the ACP client for one provider.
type Options struct {
	// ClientName identifies the client in the initialize request.
	ClientName string
	// ClientVersion is the initialize request client version.
	ClientVersion string
	// Cwd is the absolute working directory for the session. Handshake rejects
	// an empty value.
	Cwd string
	// ResumeMethod creates the session from a previous session ID. Use
	// MethodSessionLoad or MethodSessionResume.
	ResumeMethod Method
	// SetModelMethod selects the model when the session exposes no model
	// configuration option. Leave empty to reject the request.
	SetModelMethod Method
	// ModelID, EffortID, and ModeID name the session configuration options used
	// to apply SessionRequest. An empty ID disables that option.
	ModelID  ConfigOptionID
	EffortID ConfigOptionID
	ModeID   ConfigOptionID
	// AllowUnlistedModel accepts a model value the server did not advertise.
	// The server remains responsible for validating it.
	AllowUnlistedModel bool
}

// SessionRequest holds the settings to apply when the session starts.
type SessionRequest struct {
	// ResumeSessionID resumes a previous session instead of creating one.
	ResumeSessionID string
	// Model, Effort, and Mode are opaque ACP configuration values. An empty
	// value keeps the server's current selection.
	Model  string
	Effort string
	Mode   string
}

// ModelChoice is one selectable model value.
type ModelChoice struct {
	Value string
	Name  string
}

// Session is an initialized ACP session.
type Session struct {
	// ID is the server-assigned session ID.
	ID string
	// InputModalities lists the prompt kinds SendPrompt accepts. Text is always
	// present; Image is added when the agent advertises prompt image support.
	InputModalities []genai.Modality

	o         Options
	stdin     io.Writer
	sc        *bufio.Scanner
	nextID    int64
	options   []SessionConfigOption
	fallbacks []ModelInfo
}

// Handshake performs initialize and creates or resumes the session. It then
// applies the requested model, effort, and mode through session configuration
// options.
func Handshake(stdin io.Writer, stdout io.Reader, o *Options, req SessionRequest) (*Session, error) {
	if o.Cwd == "" {
		return nil, errors.New("cwd is required")
	}
	s := &Session{o: *o, stdin: stdin, sc: newScanner(stdout)}

	s.nextID++
	params, err := marshalJSONRaw(InitializeParams{
		ProtocolVersion:    1,
		ClientCapabilities: ClientCapabilities{Terminal: false},
		ClientInfo:         ClientInfo{Name: o.ClientName, Title: o.ClientName, Version: o.ClientVersion},
	})
	if err != nil {
		return nil, fmt.Errorf("marshal initialize params: %w", err)
	}
	if err := msgutil.WriteNDJSON(s.stdin, JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      s.nextID,
		Method:  MethodInitialize,
		Params:  params,
	}); err != nil {
		return nil, fmt.Errorf("write initialize: %w", err)
	}
	initData, err := s.readResponse(s.nextID)
	if err != nil {
		return nil, fmt.Errorf("read initialize response: %w", err)
	}
	var initResult InitializeResult
	if err := internal.UnmarshalJSON(initData, &initResult); err != nil {
		return nil, fmt.Errorf("parse initialize response: %w", err)
	}
	s.InputModalities = []genai.Modality{genai.ModalityText}
	if initResult.AgentCapabilities.PromptCapabilities.Image {
		s.InputModalities = append(s.InputModalities, genai.ModalityImage)
	}

	s.nextID++
	var sessionReq JSONRPCRequest
	if req.ResumeSessionID != "" {
		var params json.RawMessage
		var err error
		if o.ResumeMethod == MethodSessionLoad {
			params, err = marshalJSONRaw(SessionLoadParams{SessionID: req.ResumeSessionID, Cwd: o.Cwd, McpServers: []MCPServer{}})
		} else {
			params, err = marshalJSONRaw(SessionResumeParams{SessionID: req.ResumeSessionID, Cwd: o.Cwd, McpServers: []MCPServer{}})
		}
		if err != nil {
			return nil, fmt.Errorf("marshal %s params: %w", o.ResumeMethod, err)
		}
		sessionReq = JSONRPCRequest{JSONRPC: "2.0", ID: s.nextID, Method: o.ResumeMethod, Params: params}
	} else {
		params, err := marshalJSONRaw(SessionNewParams{Cwd: o.Cwd, McpServers: []MCPServer{}})
		if err != nil {
			return nil, fmt.Errorf("marshal session/new params: %w", err)
		}
		sessionReq = JSONRPCRequest{JSONRPC: "2.0", ID: s.nextID, Method: MethodSessionNew, Params: params}
	}
	if err := msgutil.WriteNDJSON(s.stdin, sessionReq); err != nil {
		return nil, fmt.Errorf("write session request: %w", err)
	}
	sessionData, err := s.readResponse(s.nextID)
	if err != nil {
		return nil, fmt.Errorf("read session response: %w", err)
	}
	var snResult SessionNewResult
	if err := internal.UnmarshalJSON(sessionData, &snResult); err != nil {
		return nil, fmt.Errorf("parse session result: %w", err)
	}
	if snResult.SessionID != "" {
		s.ID = snResult.SessionID
	} else if req.ResumeSessionID != "" {
		s.ID = req.ResumeSessionID
	}
	if s.ID == "" {
		return nil, errors.New("session response missing sessionId")
	}
	s.fallbacks = snResult.Models.AvailableModels
	s.options = snResult.ConfigOptions

	// Some agents do not expose a model configuration option. Model selection
	// falls back to the legacy session/set_model method.
	if req.Model != "" {
		if err := s.setModel(req.Model); err != nil {
			return nil, err
		}
	}
	if req.Effort != "" {
		if err := s.setConfigOption(o.EffortID, req.Effort); err != nil {
			return nil, err
		}
	}
	if req.Mode != "" {
		if err := s.setConfigOption(o.ModeID, req.Mode); err != nil {
			return nil, err
		}
	}
	return s, nil
}

// ModelChoices returns the values of the model selector. It falls back to the
// session's available-model list when the server exposes no selector.
func (s *Session) ModelChoices() []ModelChoice {
	if option := s.configOption(s.o.ModelID); option != nil {
		var out []ModelChoice
		flattenConfigValues(option.Options, &out)
		return out
	}
	out := make([]ModelChoice, 0, len(s.fallbacks))
	for _, m := range s.fallbacks {
		out = append(out, ModelChoice{Value: m.ModelID, Name: m.Name})
	}
	return out
}

// SendPrompt sends a session/prompt request with the user message and returns
// its JSON-RPC request ID.
func (s *Session) SendPrompt(msg *genai.Message) (int64, error) {
	content, err := msgToPromptContent(msg, s.InputModalities)
	if err != nil {
		return 0, err
	}
	s.nextID++
	id := s.nextID
	params, err := marshalJSONRaw(SessionPromptParams{SessionID: s.ID, Prompt: content})
	if err != nil {
		return 0, fmt.Errorf("marshal session/prompt params: %w", err)
	}
	return id, msgutil.WriteNDJSON(s.stdin, JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  MethodSessionPrompt,
		Params:  params,
	})
}

// ReadTurn reads session/update notifications until the session/prompt
// response arrives. For each text or reasoning delta, onDelta is called;
// returning false stops the read loop early and cancels the session.
//
// allow approves permission requests; otherwise they are rejected.
func (s *Session) ReadTurn(promptID int64, allow bool, onDelta func(text, reasoning string) bool) (genai.Result, error) {
	var textBuf, thinkBuf strings.Builder
	for s.sc.Scan() {
		line := s.sc.Bytes()
		var msg JSONRPCMessage
		if err := internal.UnmarshalJSON(line, &msg); err != nil {
			return genai.Result{}, fmt.Errorf("unmarshal JSON-RPC message: %w", err)
		}

		// Response to our prompt request → turn complete.
		if msg.IsResponse() {
			var id int64
			if err := json.Unmarshal(msg.ID, &id); err != nil {
				return genai.Result{}, fmt.Errorf("invalid JSON-RPC response id %s: %w", msg.ID, err)
			}
			if id != promptID {
				return genai.Result{}, fmt.Errorf("unexpected JSON-RPC response id %d, want %d", id, promptID)
			}
			return buildPromptResult(line, textBuf.String(), thinkBuf.String(), s.ID)
		}

		if msg.IsAgentRequest() {
			if err := handleAgentRequest(s.stdin, line, allow); err != nil {
				return genai.Result{}, fmt.Errorf("handle agent request: %w", err)
			}
			continue
		}

		if msg.Method != MethodSessionUpdate {
			continue
		}

		text, reasoning, err := parseSessionUpdateDelta(msg.Params)
		if err != nil {
			return genai.Result{}, err
		}
		textBuf.WriteString(text)
		thinkBuf.WriteString(reasoning)
		if (text != "" || reasoning != "") && !onDelta(text, reasoning) {
			if err := s.cancel(); err != nil {
				return genai.Result{}, fmt.Errorf("cancel session: %w", err)
			}
			return genai.Result{}, nil
		}
	}
	if err := s.sc.Err(); err != nil {
		return genai.Result{}, fmt.Errorf("read stdout: %w", err)
	}
	return genai.Result{}, errors.New("exited without prompt response")
}

func (s *Session) configOption(id ConfigOptionID) *SessionConfigOption {
	for i := range s.options {
		if s.options[i].ID == id {
			return &s.options[i]
		}
	}
	return nil
}

func (s *Session) setConfigOption(id ConfigOptionID, value string) error {
	option := s.configOption(id)
	if option == nil {
		return fmt.Errorf("ACP does not expose %q for the selected model", id)
	}
	if option.Type != ConfigOptionTypeSelect {
		return fmt.Errorf("ACP %q option has unsupported type %q", id, option.Type)
	}
	// Model selections may append a valid server-side variant to the advertised
	// base model ID, so the server must validate them. Other options are exact.
	valid := id == s.o.ModelID && s.o.AllowUnlistedModel
	var choices []ModelChoice
	flattenConfigValues(option.Options, &choices)
	if !valid {
		for i := range choices {
			if choices[i].Value == value {
				valid = true
				break
			}
		}
	}
	if !valid {
		available := make([]string, len(choices))
		for i := range choices {
			available[i] = choices[i].Value
		}
		return fmt.Errorf("ACP %q %q is unavailable; supported values: %s", id, value, strings.Join(available, ", "))
	}

	s.nextID++
	params, err := marshalJSONRaw(SetSessionConfigOptionParams{SessionID: s.ID, ConfigID: id, Value: value})
	if err != nil {
		return fmt.Errorf("marshal session/set_config_option params: %w", err)
	}
	if err := msgutil.WriteNDJSON(s.stdin, JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      s.nextID,
		Method:  MethodSessionSetConfigOption,
		Params:  params,
	}); err != nil {
		return fmt.Errorf("write session/set_config_option: %w", err)
	}
	data, err := s.readResponse(s.nextID)
	if err != nil {
		return fmt.Errorf("read session/set_config_option response: %w", err)
	}
	var result SetSessionConfigOptionResult
	if err := internal.UnmarshalJSON(data, &result); err != nil {
		return fmt.Errorf("parse session/set_config_option response: %w", err)
	}
	if len(result.ConfigOptions) == 0 {
		return errors.New("session/set_config_option response missing configOptions")
	}
	s.options = result.ConfigOptions
	return nil
}

func (s *Session) setModel(model string) error {
	if s.configOption(s.o.ModelID) != nil {
		return s.setConfigOption(s.o.ModelID, model)
	}
	if s.o.SetModelMethod == "" {
		return fmt.Errorf("ACP does not expose %q for the selected model", s.o.ModelID)
	}

	s.nextID++
	params, err := marshalJSONRaw(SetSessionModelParams{SessionID: s.ID, ModelID: model})
	if err != nil {
		return fmt.Errorf("marshal %s params: %w", s.o.SetModelMethod, err)
	}
	if err := msgutil.WriteNDJSON(s.stdin, JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      s.nextID,
		Method:  s.o.SetModelMethod,
		Params:  params,
	}); err != nil {
		return fmt.Errorf("write %s: %w", s.o.SetModelMethod, err)
	}
	_, err = s.readResponse(s.nextID)
	return err
}

func newScanner(r io.Reader) *bufio.Scanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 32<<20) // 1 MB initial, 32 MB max
	return sc
}

func flattenConfigValues(in []ConfigOptionValue, out *[]ModelChoice) {
	for i := range in {
		if len(in[i].Options) != 0 {
			flattenConfigValues(in[i].Options, out)
			continue
		}
		*out = append(*out, ModelChoice{Value: in[i].Value, Name: in[i].Name})
	}
}

func (s *Session) cancel() error {
	params, err := marshalJSONRaw(SessionCancelParams{SessionID: s.ID})
	if err != nil {
		return fmt.Errorf("marshal session/cancel params: %w", err)
	}
	return msgutil.WriteNDJSON(s.stdin, JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  MethodSessionCancel,
		Params:  params,
	})
}

// msgToPromptContent converts a genai.Message to ACP prompt content blocks.
func msgToPromptContent(msg *genai.Message, modalities []genai.Modality) ([]PromptContent, error) {
	var items []PromptContent
	for i := range msg.Requests {
		req := &msg.Requests[i]
		if req.Text != "" {
			items = append(items, PromptContent{Type: ContentText, Text: req.Text})
		}
		if !req.Doc.IsZero() {
			blk, err := docToPromptContent(req.Doc, modalities)
			if err != nil {
				return nil, err
			}
			items = append(items, blk)
		}
	}
	if len(items) == 0 {
		return nil, errors.New("user message has no content")
	}
	return items, nil
}

// docToPromptContent converts a genai.Doc to a promptContent block.
func docToPromptContent(doc genai.Doc, modalities []genai.Modality) (PromptContent, error) {
	if doc.URL != "" {
		mimeType, _, err := doc.Read(10 * 1024 * 1024)
		if err != nil {
			return PromptContent{}, fmt.Errorf("read URL document: %w", err)
		}
		if !slices.Contains(modalities, genai.ModalityImage) {
			return PromptContent{}, errors.New("agent does not support image input")
		}
		if !strings.HasPrefix(mimeType, "image/") {
			return PromptContent{}, fmt.Errorf("unsupported URL document MIME type %q (images only)", mimeType)
		}
		u, err := url.ParseRequestURI(doc.URL)
		if err != nil {
			return PromptContent{}, fmt.Errorf("parse image URL: %w", err)
		}
		if u.Scheme != "http" && u.Scheme != "https" {
			return PromptContent{}, fmt.Errorf("unsupported image URL scheme %q", u.Scheme)
		}
		return PromptContent{Type: ContentImage, URI: doc.URL, MimeType: mimeType}, nil
	}
	mimeType, data, err := doc.Read(10 * 1024 * 1024)
	if err != nil {
		return PromptContent{}, fmt.Errorf("read doc: %w", err)
	}
	if strings.HasPrefix(mimeType, "text/") {
		return PromptContent{Type: ContentText, Text: string(data)}, nil
	}
	if !slices.Contains(modalities, genai.ModalityImage) {
		return PromptContent{}, errors.New("agent does not support image input")
	}
	if !strings.HasPrefix(mimeType, "image/") {
		return PromptContent{}, fmt.Errorf("unsupported doc MIME type %q (text and images only)", mimeType)
	}
	return PromptContent{Type: ContentImage, Data: base64.StdEncoding.EncodeToString(data), MimeType: mimeType}, nil
}

// readResponse reads lines from the session scanner until the expected JSON-RPC
// response is found. Notifications are skipped and agent requests are answered.
func (s *Session) readResponse(expectedID int64) (json.RawMessage, error) {
	for s.sc.Scan() {
		line := s.sc.Bytes()
		var msg JSONRPCMessage
		if err := internal.UnmarshalJSON(line, &msg); err != nil {
			return nil, fmt.Errorf("unmarshal JSON-RPC message: %w", err)
		}
		if msg.IsAgentRequest() {
			if err := handleAgentRequest(s.stdin, line, false); err != nil {
				return nil, fmt.Errorf("handle agent request during handshake: %w", err)
			}
			continue
		}
		if !msg.IsResponse() {
			continue
		}
		var id int64
		if err := json.Unmarshal(msg.ID, &id); err != nil {
			return nil, fmt.Errorf("invalid JSON-RPC response id %s: %w", msg.ID, err)
		}
		if id != expectedID {
			return nil, fmt.Errorf("unexpected JSON-RPC response id %d, want %d", id, expectedID)
		}
		if msg.Error != nil {
			if len(msg.Error.Data) != 0 {
				return nil, fmt.Errorf("JSON-RPC error %d: %s: %s", msg.Error.Code, msg.Error.Message, msg.Error.Data)
			}
			return nil, fmt.Errorf("JSON-RPC error %d: %s", msg.Error.Code, msg.Error.Message)
		}
		return msg.Result, nil
	}
	if err := s.sc.Err(); err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	return nil, errors.New("exited during handshake")
}

// parseSessionUpdateDelta extracts text and reasoning deltas from a
// session/update notification's params.
func parseSessionUpdateDelta(params json.RawMessage) (text, reasoning string, err error) {
	var sup SessionUpdateParams
	if err := internal.UnmarshalJSON(params, &sup); err != nil {
		return "", "", fmt.Errorf("unmarshal session/update params: %w", err)
	}
	// The probe reads only the discriminator; the matching update type is
	// decoded strictly below. Unknown fields belong to that type, not the probe.
	var probe UpdateProbe
	if err := json.Unmarshal(sup.Update, &probe); err != nil {
		return "", "", fmt.Errorf("unmarshal session/update discriminator: %w", err)
	}
	switch probe.SessionUpdate {
	case UpdateAgentMessageChunk:
		var u AgentMessageChunkUpdate
		if err := internal.UnmarshalJSON(sup.Update, &u); err != nil {
			return "", "", fmt.Errorf("unmarshal agent_message_chunk: %w", err)
		}
		return u.Content.Text, "", nil
	case UpdateAgentThoughtChunk:
		var u AgentThoughtChunkUpdate
		if err := internal.UnmarshalJSON(sup.Update, &u); err != nil {
			return "", "", fmt.Errorf("unmarshal agent_thought_chunk: %w", err)
		}
		return "", u.Content.Text, nil
	case UpdateUserMessageChunk:
		return "", "", decodeSessionUpdate[UserMessageChunkUpdate](sup.Update, probe.SessionUpdate)
	case UpdateToolCall:
		return "", "", decodeSessionUpdate[ToolCallUpdate](sup.Update, probe.SessionUpdate)
	case UpdateToolCallUpdate:
		return "", "", decodeSessionUpdate[ToolCallUpdateUpdate](sup.Update, probe.SessionUpdate)
	case UpdatePlan:
		return "", "", decodeSessionUpdate[PlanUpdate](sup.Update, probe.SessionUpdate)
	case UpdateUsageUpdate:
		return "", "", decodeSessionUpdate[UsageUpdateUpdate](sup.Update, probe.SessionUpdate)
	case UpdateCurrentModeUpdate:
		return "", "", decodeSessionUpdate[CurrentModeUpdate](sup.Update, probe.SessionUpdate)
	case UpdateSessionInfoUpdate:
		return "", "", decodeSessionUpdate[SessionInfoUpdate](sup.Update, probe.SessionUpdate)
	case UpdateAvailableCommandsUpdate:
		return "", "", decodeSessionUpdate[AvailableCommandsUpdate](sup.Update, probe.SessionUpdate)
	case UpdateConfigOptionUpdate:
		return "", "", decodeSessionUpdate[ConfigOptionUpdate](sup.Update, probe.SessionUpdate)
	default:
		return "", "", nil
	}
}

func decodeSessionUpdate[T any](data json.RawMessage, updateType UpdateType) error {
	var update T
	if err := internal.UnmarshalJSON(data, &update); err != nil {
		return fmt.Errorf("unmarshal %s: %w", updateType, err)
	}
	return nil
}

// handleAgentRequest responds to JSON-RPC requests from the agent. A permission
// request selects the first allow option when allow is set and the first
// reject option otherwise, and is cancelled when no such option exists.
func handleAgentRequest(stdin io.Writer, line []byte, allow bool) error {
	var msg JSONRPCMessage
	if err := internal.UnmarshalJSON(line, &msg); err != nil {
		return fmt.Errorf("unmarshal agent request: %w", err)
	}
	if msg.Method != MethodSessionRequestPermission {
		data, err := marshalJSONRaw(MethodNotFoundData{Method: msg.Method})
		if err != nil {
			return fmt.Errorf("marshal method-not-found data: %w", err)
		}
		return msgutil.WriteNDJSON(stdin, JSONRPCResponse{
			JSONRPC: "2.0",
			ID:      msg.ID,
			Error: JSONRPCError{
				Code:    -32601,
				Message: fmt.Sprintf("method not found: %s", msg.Method),
				Data:    data,
			},
		})
	}
	var params PermissionRequestParams
	if err := internal.UnmarshalJSON(msg.Params, &params); err != nil {
		return fmt.Errorf("unmarshal permission request: %w", err)
	}
	optionID := ""
	for _, o := range params.Options {
		isAllow := o.Kind == PermissionAllowOnce || o.Kind == PermissionAllowAlways
		isReject := o.Kind == PermissionRejectOnce || o.Kind == PermissionRejectAlways
		if allow && isAllow || !allow && isReject {
			optionID = o.OptionID
			break
		}
	}
	outcome := PermissionOutcome{Outcome: PermissionOutcomeCancelled}
	if optionID != "" {
		outcome = PermissionOutcome{Outcome: PermissionOutcomeSelected, OptionID: optionID}
	}
	result, err := marshalJSONRaw(PermissionResponseResult{Outcome: outcome})
	if err != nil {
		return fmt.Errorf("marshal permission response result: %w", err)
	}
	return msgutil.WriteNDJSON(stdin, JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      msg.ID,
		Result:  result,
	})
}

// buildPromptResult constructs a genai.Result from the session/prompt response.
// Returns an error if the JSON-RPC response is an error.
func buildPromptResult(line []byte, text, thinking, sessionID string) (genai.Result, error) {
	r := genai.Result{}

	var msg JSONRPCMessage
	if err := internal.UnmarshalJSON(line, &msg); err != nil {
		return r, fmt.Errorf("unmarshal session/prompt response: %w", err)
	}
	if msg.Error != nil {
		if len(msg.Error.Data) != 0 {
			return r, fmt.Errorf("JSON-RPC error %d: %s: %s", msg.Error.Code, msg.Error.Message, msg.Error.Data)
		}
		return r, fmt.Errorf("JSON-RPC error %d: %s", msg.Error.Code, msg.Error.Message)
	}
	if msg.Result != nil {
		var pr PromptResult
		if err := internal.UnmarshalJSON(msg.Result, &pr); err != nil {
			return r, fmt.Errorf("unmarshal session/prompt result: %w", err)
		}
		r.Usage.FinishReason = stopReasonToFinishReason(pr.StopReason)
		r.Usage.InputTokens = int64(pr.Usage.InputTokens)
		r.Usage.OutputTokens = int64(pr.Usage.OutputTokens)
		r.Usage.ReasoningTokens = int64(pr.Usage.ThoughtTokens)
		r.Usage.InputCachedTokens = int64(pr.Usage.CachedReadTokens)
		// ACP totals include reasoning and both cache token categories.
		r.Usage.TotalTokens = int64(pr.Usage.TotalTokens)
	}

	if thinking != "" {
		r.Replies = append(r.Replies, genai.Reply{Reasoning: thinking})
	}
	if text != "" {
		r.Replies = append(r.Replies, genai.Reply{Text: text})
	}
	if sessionID != "" {
		r.Replies = append(r.Replies, genai.Reply{
			Opaque: map[string]any{SessionIDKey: sessionID},
		})
	}
	return r, nil
}

// stopReasonToFinishReason maps an ACP stop reason string to a
// genai.FinishReason.
func stopReasonToFinishReason(reason StopReason) genai.FinishReason {
	switch reason {
	case StopReasonMaxTokens, StopReasonMaxTurnRequests:
		return genai.FinishedLength
	case StopReasonCancelled, StopReasonRefusal:
		return genai.FinishedContentFilter
	default:
		return genai.FinishedStop
	}
}

func marshalJSONRaw(v any) (json.RawMessage, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return json.RawMessage(b), nil
}
