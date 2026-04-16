// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package opencode implements a genai provider backed by the OpenCode CLI.
//
// Instead of making HTTP requests directly, it launches `opencode acp` as a
// subprocess and communicates over stdin/stdout using the ACP (Agent Client
// Protocol) JSON-RPC 2.0 protocol.
//
// See https://opencode.ai for OpenCode documentation and
// https://agentclientprotocol.com for the ACP specification.
//
// # Protocol
//
// The provider performs a JSON-RPC handshake (initialize → session/new) on each
// call, then sends a session/prompt request and reads session/update
// notifications until the session/prompt response arrives.
//
// # Session / multi-turn
//
// Each GenSync or GenStream call launches a fresh subprocess. The session ID is
// always returned inside Reply.Opaque["session_id"]. When the message history
// contains a previous session ID, it is automatically picked up: session/load
// is used instead of session/new and only the last user message is sent.
package opencode

import (
	"bufio"
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/msgutil"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for the OpenCode CLI provider.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// sessionIDKey is the key used in Reply.Opaque to carry session IDs.
const sessionIDKey = "session_id"

// cmdExecutor is the production executor backed by exec.Cmd.
type cmdExecutor struct{ bin string }

func (e *cmdExecutor) start(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	dir, err := os.MkdirTemp("", "genai-opencode-*")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("create temp dir: %w", err)
	}
	cmd := exec.CommandContext(ctx, e.bin, args...)
	cmd.Dir = dir
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, nil, errors.Join(fmt.Errorf("stdin pipe: %w", err), os.RemoveAll(dir))
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, errors.Join(fmt.Errorf("stdout pipe: %w", err), os.RemoveAll(dir))
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, nil, errors.Join(fmt.Errorf("start opencode: %w", err), os.RemoveAll(dir))
	}
	return stdin, stdout, func() error {
		return errors.Join(cmd.Wait(), os.RemoveAll(dir))
	}, nil
}

// newScanner wraps an io.Reader in a bufio.Scanner sized for large JSON-RPC
// messages (up to 32 MB).
func newScanner(r io.Reader) *bufio.Scanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 32<<20) // 1 MB initial, 32 MB max
	return sc
}

// Client is a genai provider that delegates to the local `opencode` CLI.
type Client struct {
	base.NotImplemented
	exec           genai.Starter
	starterWrapper genai.ProviderOptionStarterWrapper
	bin            string
	model          string
	binOnce        sync.Once
	binErr         error
}

// New creates a Client for the `opencode` CLI.
//
// The binary is located lazily on the first call to GenSync, GenStream, or
// Ping, so New succeeds even when the CLI is not installed.
//
// Supported ProviderOptions:
//   - genai.ProviderOptionModel — model ID (e.g. "opencode/big-pickle").
//     Use genai.ModelCheap, genai.ModelGood, or genai.ModelSOTA for automatic selection.
func New(opts ...genai.ProviderOption) (*Client, error) {
	c := &Client{}
	if err := base.CheckDuplicateOptions(opts); err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionModel:
			switch v {
			case genai.ModelCheap:
				c.model = "opencode/gpt-5-nano"
			case genai.ModelGood:
				c.model = "opencode/big-pickle"
			case genai.ModelSOTA:
				c.model = "openai/gpt-5.4/xhigh"
			default:
				c.model = string(v)
			}
		case genai.ProviderOptionStarterWrapper:
			c.starterWrapper = v
		default:
			return nil, fmt.Errorf("unsupported provider option %T", opt)
		}
	}
	return c, nil
}

// ensureBin locates the opencode binary on first call. Safe for concurrent use.
func (c *Client) ensureBin() error {
	c.binOnce.Do(func() {
		bin, err := exec.LookPath("opencode")
		if err != nil {
			if c.starterWrapper == nil {
				c.binErr = fmt.Errorf("opencode CLI not found on PATH: %w", err)
				return
			}
			bin = "opencode"
		}
		c.bin = bin
		s := genai.Starter((&cmdExecutor{bin: bin}).start)
		if c.starterWrapper != nil {
			s = c.starterWrapper(s)
		}
		c.exec = s
	})
	return c.binErr
}

// Name implements genai.Provider.
func (c *Client) Name() string { return "opencode" }

// ModelID implements genai.Provider.
func (c *Client) ModelID() string { return c.model }

// OutputModalities implements genai.Provider.
func (c *Client) OutputModalities() genai.Modalities {
	return genai.Modalities{genai.ModalityText}
}

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{}
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score { return Scoreboard() }

// HTTPClient implements genai.Provider. The CLI provider does not use HTTP.
func (c *Client) HTTPClient() *http.Client { return nil }

// Ping implements genai.ProviderPing by running `opencode --version`.
func (c *Client) Ping(ctx context.Context) error {
	if err := c.ensureBin(); err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, c.bin, "--version")
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("opencode --version: %w", err)
	}
	if !strings.Contains(strings.ToLower(string(out)), "opencode") {
		return fmt.Errorf("unexpected opencode --version output: %q", string(out))
	}
	return nil
}

// ListModels implements genai.Provider by querying the OpenCode ACP server's
// available models. It launches a subprocess, performs the handshake, and
// returns the model list from session/new.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if err := c.ensureBin(); err != nil {
		return nil, err
	}
	stdin, stdout, wait, err := c.exec(ctx, []string{"acp"})
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = stdin.Close()
		_ = wait()
	}()

	sc := newScanner(stdout)
	hs, err := handshake(stdin, sc, "", "")
	if err != nil {
		return nil, err
	}
	out := make([]genai.Model, 0, len(hs.availableModels))
	for _, m := range hs.availableModels {
		name := m.Name
		if name == "" {
			name = m.ModelID
		}
		out = append(out, &model{id: m.ModelID, displayName: name})
	}
	return out, nil
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (r genai.Result, err error) {
	if err := c.ensureBin(); err != nil {
		return genai.Result{}, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return genai.Result{}, err
		}
	}
	if len(opts) > 0 {
		defer func() {
			if err == nil {
				err = &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", opts[0])}}
			}
		}()
	}
	userMsg, err := msgutil.LastUserMsg(msgs)
	if err != nil {
		return genai.Result{}, err
	}
	resumeSessionID := msgutil.ExtractOpaqueID(msgs, sessionIDKey)

	stdin, stdout, wait, err := c.exec(ctx, []string{"acp"})
	if err != nil {
		return genai.Result{}, err
	}
	defer func() {
		_ = stdin.Close()
		err = errors.Join(err, wait())
	}()

	sc := newScanner(stdout)
	hs, err := handshake(stdin, sc, c.model, resumeSessionID)
	if err != nil {
		return genai.Result{}, err
	}

	promptID, err := sendUserPrompt(stdin, hs, &userMsg)
	if err != nil {
		return genai.Result{}, err
	}

	return readTurn(sc, stdin, hs.sessionID, promptID, func(string, string) bool { return true })
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if err := c.ensureBin(); err != nil {
		return yieldNothing, errFinish(err)
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return yieldNothing, errFinish(err)
		}
	}
	var optsErr error
	if len(opts) > 0 {
		optsErr = &base.ErrNotSupported{Options: []string{fmt.Sprintf("%T", opts[0])}}
	}
	userMsg, err := msgutil.LastUserMsg(msgs)
	if err != nil {
		return yieldNothing, errFinish(err)
	}
	resumeSessionID := msgutil.ExtractOpaqueID(msgs, sessionIDKey)

	var (
		result   genai.Result
		finalErr error
	)
	seq := func(yield func(genai.Reply) bool) {
		stdin, stdout, wait, startErr := c.exec(ctx, []string{"acp"})
		if startErr != nil {
			finalErr = startErr
			return
		}
		defer func() {
			_ = stdin.Close()
			finalErr = errors.Join(finalErr, wait())
		}()

		sc := newScanner(stdout)
		hs, hsErr := handshake(stdin, sc, c.model, resumeSessionID)
		if hsErr != nil {
			finalErr = hsErr
			return
		}
		promptID, err := sendUserPrompt(stdin, hs, &userMsg)
		if err != nil {
			finalErr = err
			return
		}

		result, finalErr = readTurn(sc, stdin, hs.sessionID, promptID, func(text, reasoning string) bool {
			if text != "" && !yield(genai.Reply{Text: text}) {
				return false
			}
			if reasoning != "" && !yield(genai.Reply{Reasoning: reasoning}) {
				return false
			}
			return true
		})
	}
	return seq, func() (genai.Result, error) {
		if finalErr == nil {
			finalErr = optsErr
		}
		return result, finalErr
	}
}

// handshakeResult bundles everything returned by a successful handshake.
type handshakeResult struct {
	sessionID       string
	supportsImage   bool
	nextID          int64
	availableModels []ModelInfo
}

// handshake performs the ACP initialize → session/new sequence.
func handshake(stdin io.Writer, sc *bufio.Scanner, mdl, resumeSessionID string) (*handshakeResult, error) {
	hs := &handshakeResult{}

	// 1. Send initialize request.
	hs.nextID++
	if err := msgutil.WriteNDJSON(stdin, JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      hs.nextID,
		Method:  MethodInitialize,
		Params: InitializeParams{
			ProtocolVersion:    1,
			ClientCapabilities: ClientCapabilities{Terminal: false},
			ClientInfo:         ClientInfo{Name: "genai-opencode", Title: "genai-opencode", Version: "1.0.0"},
		},
	}); err != nil {
		return nil, fmt.Errorf("write initialize: %w", err)
	}
	initData, err := readResponse(sc)
	if err != nil {
		return nil, fmt.Errorf("read initialize response: %w", err)
	}
	var initResult InitializeResult
	if internal.UnmarshalJSON(initData, &initResult) == nil {
		hs.supportsImage = initResult.AgentCapabilities.PromptCapabilities.Image
	}

	// 2. Create or resume session.
	hs.nextID++
	var sessionReq JSONRPCRequest
	if resumeSessionID != "" {
		sessionReq = JSONRPCRequest{
			JSONRPC: "2.0",
			ID:      hs.nextID,
			Method:  MethodSessionLoad,
			Params:  SessionLoadParams{SessionID: resumeSessionID, Cwd: os.TempDir(), McpServers: []MCPServer{}},
		}
	} else {
		sessionReq = JSONRPCRequest{
			JSONRPC: "2.0",
			ID:      hs.nextID,
			Method:  MethodSessionNew,
			Params:  SessionNewParams{Cwd: os.TempDir(), McpServers: []MCPServer{}},
		}
	}
	if err := msgutil.WriteNDJSON(stdin, sessionReq); err != nil {
		return nil, fmt.Errorf("write session request: %w", err)
	}
	sessionData, err := readResponse(sc)
	if err != nil {
		return nil, fmt.Errorf("read session response: %w", err)
	}
	var snResult SessionNewResult
	if err := internal.UnmarshalJSON(sessionData, &snResult); err != nil {
		return nil, fmt.Errorf("parse session result: %w", err)
	}
	if snResult.SessionID != "" {
		hs.sessionID = snResult.SessionID
	} else if resumeSessionID != "" {
		hs.sessionID = resumeSessionID
	}
	if hs.sessionID == "" {
		return nil, errors.New("session response missing sessionId")
	}
	hs.availableModels = snResult.Models.AvailableModels

	// 3. Switch model if requested (best-effort).
	if mdl != "" {
		hs.nextID++
		if err := msgutil.WriteNDJSON(stdin, JSONRPCRequest{
			JSONRPC: "2.0",
			ID:      hs.nextID,
			Method:  MethodUnstableSetSessionModel,
			Params:  SetSessionModelParams{SessionID: hs.sessionID, ModelID: mdl},
		}); err != nil {
			return nil, fmt.Errorf("write setSessionModel: %w", err)
		}
		// Best-effort: ignore errors from the unstable method.
		_, _ = readResponse(sc)
	}

	return hs, nil
}

// sendUserPrompt sends a session/prompt JSON-RPC request with the user message.
func sendUserPrompt(stdin io.Writer, hs *handshakeResult, msg *genai.Message) (int64, error) {
	content, err := msgToPromptContent(msg, hs.supportsImage)
	if err != nil {
		return 0, err
	}
	hs.nextID++
	id := hs.nextID
	return id, msgutil.WriteNDJSON(stdin, JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  MethodSessionPrompt,
		Params:  SessionPromptParams{SessionID: hs.sessionID, Prompt: content},
	})
}

// msgToPromptContent converts a genai.Message to ACP prompt content blocks.
func msgToPromptContent(msg *genai.Message, supportsImage bool) ([]PromptContent, error) {
	var items []PromptContent
	for i := range msg.Requests {
		req := &msg.Requests[i]
		if req.Text != "" {
			items = append(items, PromptContent{Type: ContentText, Text: req.Text})
		}
		if !req.Doc.IsZero() {
			blk, err := docToPromptContent(req.Doc, supportsImage)
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
//
// URL images are not supported: the ACP agent.ts handler only accepts http:
// (not https:) URLs, so we reject all URLs and require inline base64.
// See packages/opencode/src/acp/agent.ts case "image" (~line 1327).
func docToPromptContent(doc genai.Doc, supportsImage bool) (PromptContent, error) {
	if doc.URL != "" {
		return PromptContent{}, fmt.Errorf("URL documents not supported by opencode ACP; inline the content")
	}
	mimeType, data, err := doc.Read(10 * 1024 * 1024)
	if err != nil {
		return PromptContent{}, fmt.Errorf("read doc: %w", err)
	}
	if strings.HasPrefix(mimeType, "text/") {
		return PromptContent{Type: ContentText, Text: string(data)}, nil
	}
	if !supportsImage {
		return PromptContent{}, fmt.Errorf("opencode agent does not support image input")
	}
	if !strings.HasPrefix(mimeType, "image/") {
		return PromptContent{}, fmt.Errorf("unsupported doc MIME type %q for opencode (text and images only)", mimeType)
	}
	return PromptContent{
		Type:     ContentImage,
		Data:     base64.StdEncoding.EncodeToString(data),
		MimeType: mimeType,
	}, nil
}

// readResponse reads lines from the scanner until a JSON-RPC response (has
// "id" field, no "method" field) is found. Notifications are skipped.
func readResponse(sc *bufio.Scanner) (json.RawMessage, error) {
	for sc.Scan() {
		var msg JSONRPCMessage
		if json.Unmarshal(sc.Bytes(), &msg) != nil {
			continue
		}
		if !msg.IsResponse() || msg.Method != "" {
			continue
		}
		if msg.Error != nil {
			return nil, fmt.Errorf("JSON-RPC error %d: %s", msg.Error.Code, msg.Error.Message)
		}
		return msg.Result, nil
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	return nil, errors.New("opencode exited during handshake")
}

// readTurn reads session/update notifications until the session/prompt response
// arrives. For each text or reasoning delta, onDelta is called; returning false
// stops the read loop early (used by GenStream when the caller breaks).
func readTurn(sc *bufio.Scanner, stdin io.Writer, sessionID string, promptID int64, onDelta func(text, reasoning string) bool) (genai.Result, error) {
	var textBuf, thinkBuf strings.Builder
	for sc.Scan() {
		line := sc.Bytes()
		var probe MessageProbe
		if json.Unmarshal(line, &probe) != nil {
			continue
		}

		// Response to our prompt request → turn complete.
		if len(probe.ID) > 0 && probe.Method == "" {
			var id int64
			if json.Unmarshal(probe.ID, &id) == nil && id == promptID {
				return buildPromptResult(line, textBuf.String(), thinkBuf.String(), sessionID)
			}
			continue
		}

		// Request from agent (permission) → auto-approve.
		if len(probe.ID) > 0 && probe.Method != "" {
			if err := handleAgentRequest(stdin, line); err != nil {
				return genai.Result{}, fmt.Errorf("handle agent request: %w", err)
			}
			continue
		}

		if probe.Method != MethodSessionUpdate {
			continue
		}

		var msg JSONRPCMessage
		if json.Unmarshal(line, &msg) != nil {
			continue
		}
		text, reasoning, err := parseSessionUpdateDelta(msg.Params)
		if err != nil {
			return genai.Result{}, err
		}
		textBuf.WriteString(text)
		thinkBuf.WriteString(reasoning)
		if (text != "" || reasoning != "") && !onDelta(text, reasoning) {
			return genai.Result{}, nil
		}
	}
	if err := sc.Err(); err != nil {
		return genai.Result{}, fmt.Errorf("read stdout: %w", err)
	}
	return genai.Result{}, errors.New("opencode exited without prompt response")
}

// parseSessionUpdateDelta extracts text and reasoning deltas from a
// session/update notification's params.
func parseSessionUpdateDelta(params json.RawMessage) (text, reasoning string, err error) {
	var sup SessionUpdateParams
	if err := internal.UnmarshalJSON(params, &sup); err != nil {
		return "", "", fmt.Errorf("unmarshal session/update params: %w", err)
	}
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
	default:
		return "", "", nil
	}
}

// handleAgentRequest responds to JSON-RPC requests from the agent (e.g.
// permission requests) by auto-approving with the first "allow" option.
func handleAgentRequest(stdin io.Writer, line []byte) error {
	var msg JSONRPCMessage
	if err := internal.UnmarshalJSON(line, &msg); err != nil {
		return fmt.Errorf("unmarshal agent request: %w", err)
	}
	var id int64
	if err := json.Unmarshal(msg.ID, &id); err != nil {
		return fmt.Errorf("unmarshal request id: %w", err)
	}
	if msg.Method != MethodSessionRequestPermission {
		return msgutil.WriteNDJSON(stdin, JSONRPCResponse{JSONRPC: "2.0", ID: id, Result: struct{}{}})
	}
	var params PermissionRequestParams
	if err := internal.UnmarshalJSON(msg.Params, &params); err != nil {
		return fmt.Errorf("unmarshal permission request: %w", err)
	}
	// Find the first allow option.
	optionID := ""
	for _, o := range params.Options {
		if o.Kind == "allow_once" || o.Kind == "allow_always" {
			optionID = o.OptionID
			break
		}
	}
	if optionID == "" && len(params.Options) > 0 {
		optionID = params.Options[0].OptionID
	}
	return msgutil.WriteNDJSON(stdin, JSONRPCResponse{
		JSONRPC: "2.0",
		ID:      id,
		Result:  PermissionResponseResult{OptionID: optionID},
	})
}

// buildPromptResult constructs a genai.Result from the session/prompt response.
// Returns an error if the JSON-RPC response is an error.
func buildPromptResult(line []byte, text, thinking, sessionID string) (genai.Result, error) {
	r := genai.Result{}

	var msg JSONRPCMessage
	if internal.UnmarshalJSON(line, &msg) == nil {
		if msg.Error != nil {
			return r, fmt.Errorf("JSON-RPC error %d: %s", msg.Error.Code, msg.Error.Message)
		}
		if msg.Result != nil {
			var pr PromptResult
			if internal.UnmarshalJSON(msg.Result, &pr) == nil {
				r.Usage.FinishReason = stopReasonToFinishReason(pr.StopReason)
				r.Usage.InputTokens = int64(pr.Usage.InputTokens)
				r.Usage.OutputTokens = int64(pr.Usage.OutputTokens)
				r.Usage.ReasoningTokens = int64(pr.Usage.ThoughtTokens)
				r.Usage.InputCachedTokens = int64(pr.Usage.CachedReadTokens)
				r.Usage.TotalTokens = int64(pr.Usage.InputTokens + pr.Usage.OutputTokens)
			}
		}
	}

	if thinking != "" {
		r.Replies = append(r.Replies, genai.Reply{Reasoning: thinking})
	}
	if text != "" {
		r.Replies = append(r.Replies, genai.Reply{Text: text})
	}
	if sessionID != "" {
		r.Replies = append(r.Replies, genai.Reply{
			Opaque: map[string]any{sessionIDKey: sessionID},
		})
	}
	return r, nil
}

// yieldNothing is an empty iterator used for early error returns in GenStream.
func yieldNothing(func(genai.Reply) bool) {}

// errFinish returns a finish function that always returns the given error.
func errFinish(err error) func() (genai.Result, error) {
	return func() (genai.Result, error) { return genai.Result{}, err }
}

// model is the genai.Model implementation for OpenCode models.
type model struct {
	id          string
	displayName string
}

func (m *model) GetID() string  { return m.id }
func (m *model) String() string { return m.displayName }
func (m *model) Context() int64 { return 200_000 }

// stopReasonToFinishReason maps an ACP stop reason string to a genai.FinishReason.
func stopReasonToFinishReason(reason string) genai.FinishReason {
	switch reason {
	case "max_tokens":
		return genai.FinishedLength
	case "cancelled", "refusal":
		return genai.FinishedContentFilter
	default:
		return genai.FinishedStop
	}
}

// Compile-time interface checks.
var (
	_ genai.Provider     = (*Client)(nil)
	_ genai.ProviderPing = (*Client)(nil)
)
