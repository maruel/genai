// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package codex implements a genai provider backed by the Codex CLI.
//
// Instead of making HTTP requests directly, it launches `codex app-server` as
// a subprocess and communicates over stdin/stdout using the JSON-RPC 2.0
// protocol.
//
// See https://codex.openai.com/ for Codex CLI documentation.
//
// # Protocol
//
// The provider performs a JSON-RPC handshake (initialize → initialized →
// model/list → thread/start) on each call, then sends a turn/start request
// and reads notifications until turn/completed.
//
// # Session / multi-turn
//
// Each GenSync or GenStream call launches a fresh subprocess. The thread ID is
// always returned inside Reply.Opaque["thread_id"]. When the message history
// contains a previous thread ID, it is automatically picked up: thread/resume
// is used instead of thread/start and only the last user message is sent.
package codex

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
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for the Codex CLI provider.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// threadIDKey is the key used in Reply.Opaque to carry thread IDs.
const threadIDKey = "thread_id"

// optOutMethods are notification methods we opt out of during initialization.
// These reduce noise for a text-generation-only use case.
var optOutMethods = []string{
	"item/commandExecution/terminalInteraction",
	"item/fileChange/outputDelta",
	"item/reasoning/summaryPartAdded",
	"item/reasoning/textDelta",
	"item/plan/delta",
	"turn/diff/updated",
	"turn/plan/updated",
	"thread/name/updated",
}

// executor abstracts subprocess creation so tests can inject a recording or
// fake implementation.
type executor interface {
	start(ctx context.Context, args []string) (stdin io.WriteCloser, stdout io.ReadCloser, wait func() error, err error)
}

// cmdExecutor is the production executor backed by exec.Cmd.
type cmdExecutor struct{ bin string }

func (e *cmdExecutor) start(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	dir, err := os.MkdirTemp("", "genai-codex-*")
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
		return nil, nil, nil, errors.Join(fmt.Errorf("start codex: %w", err), os.RemoveAll(dir))
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

// ReasoningEffort controls how much reasoning the model performs.
//
// Use the ReasoningEffort* constants. Defaults to ReasoningEffortMedium if unset.
type ReasoningEffort string

// Reasoning effort levels, from least to most compute.
const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// Validate implements genai.ProviderOption.
func (p ReasoningEffort) Validate() error {
	switch p {
	case ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh:
		return nil
	default:
		return fmt.Errorf("invalid reasoning effort %q; use one of the ReasoningEffort* constants", string(p))
	}
}

// Client is a genai provider that delegates to the local `codex` CLI.
type Client struct {
	base.NotImplemented
	exec    executor
	bin     string
	model   string
	effort  ReasoningEffort
	binOnce sync.Once
	binErr  error
}

// New creates a Client for the `codex` CLI.
//
// The binary is located lazily on the first call to GenSync, GenStream, or
// Ping, so New succeeds even when the CLI is not installed.
//
// Supported ProviderOptions:
//   - genai.ProviderOptionModel — model ID (e.g. "gpt-5.4", "gpt-5.3-codex").
//     Use genai.ModelCheap, genai.ModelGood, or genai.ModelSOTA for automatic selection.
//   - ReasoningEffort — reasoning depth ("none", "minimal", "low",
//     "medium", "high", "xhigh"). Defaults to "medium".
func New(opts ...genai.ProviderOption) (*Client, error) {
	c := &Client{effort: ReasoningEffortMedium}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionModel:
			switch v {
			case genai.ModelCheap:
				c.model = "gpt-5.1-codex-mini"
			case genai.ModelGood:
				c.model = "gpt-5.3-codex"
			case genai.ModelSOTA:
				c.model = "gpt-5.4"
			default:
				c.model = string(v)
			}
		case ReasoningEffort:
			c.effort = v
		default:
			return nil, fmt.Errorf("unsupported provider option %T", opt)
		}
	}
	return c, nil
}

// ensureBin locates the codex binary on first call. Safe for concurrent use.
func (c *Client) ensureBin() error {
	c.binOnce.Do(func() {
		if c.exec != nil {
			return
		}
		bin, err := exec.LookPath("codex")
		if err != nil {
			c.binErr = fmt.Errorf("codex CLI not found on PATH: %w", err)
			return
		}
		c.bin = bin
		c.exec = &cmdExecutor{bin: bin}
	})
	return c.binErr
}

// Name implements genai.Provider.
func (c *Client) Name() string { return "codex" }

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

// Ping implements genai.ProviderPing by running `codex --version`.
func (c *Client) Ping(ctx context.Context) error {
	if err := c.ensureBin(); err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, c.bin, "--version")
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("codex --version: %w", err)
	}
	if !strings.Contains(strings.ToLower(string(out)), "codex") {
		return fmt.Errorf("unexpected codex --version output: %q", string(out))
	}
	return nil
}

// ListModels implements genai.Provider by querying the codex app-server's
// model/list RPC. It launches a subprocess, performs the JSON-RPC handshake
// (initialize → initialized → model/list), and returns the live model list.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if err := c.ensureBin(); err != nil {
		return nil, err
	}
	stdin, stdout, wait, err := c.exec.start(ctx, []string{"app-server"})
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = stdin.Close()
		_ = wait()
	}()

	sc := newScanner(stdout)
	models, err := initAndListModels(stdin, sc)
	if err != nil {
		return nil, err
	}
	out := make([]genai.Model, len(models))
	for i, m := range models {
		name := m.DisplayName
		if name == "" {
			name = m.ID
		}
		out[i] = &model{id: m.ID, displayName: name}
	}
	return out, nil
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (r genai.Result, err error) {
	if err := c.ensureBin(); err != nil {
		return genai.Result{}, err
	}
	_, optsErr := parseOpts(opts)
	if optsErr != nil {
		var uerr *base.ErrNotSupported
		if !errors.As(optsErr, &uerr) {
			return genai.Result{}, optsErr
		}
		defer func() {
			if err == nil {
				err = optsErr
			}
		}()
	}
	userMsg, err := lastUserMsg(msgs)
	if err != nil {
		return genai.Result{}, err
	}
	threadID := extractThreadID(msgs)

	stdin, stdout, wait, err := c.exec.start(ctx, []string{"app-server"})
	if err != nil {
		return genai.Result{}, err
	}
	defer func() {
		_ = stdin.Close()
		err = errors.Join(err, wait())
	}()

	sc := newScanner(stdout)
	newThreadID, err := handshake(stdin, sc, c.model, threadID)
	if err != nil {
		return genai.Result{}, err
	}

	if err := sendTurnStart(stdin, newThreadID, c.effort, &userMsg); err != nil {
		return genai.Result{}, err
	}

	return readTurnSync(sc, newThreadID)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if err := c.ensureBin(); err != nil {
		return yieldNothing, errFinish(err)
	}
	_, optsErr := parseOpts(opts)
	if optsErr != nil {
		var uerr *base.ErrNotSupported
		if !errors.As(optsErr, &uerr) {
			return yieldNothing, errFinish(optsErr)
		}
	}
	userMsg, err := lastUserMsg(msgs)
	if err != nil {
		return yieldNothing, errFinish(err)
	}
	threadID := extractThreadID(msgs)

	var (
		result   genai.Result
		finalErr error
	)
	seq := func(yield func(genai.Reply) bool) {
		stdin, stdout, wait, startErr := c.exec.start(ctx, []string{"app-server"})
		if startErr != nil {
			finalErr = startErr
			return
		}
		defer func() {
			_ = stdin.Close()
			finalErr = errors.Join(finalErr, wait())
		}()

		sc := newScanner(stdout)
		newThreadID, hsErr := handshake(stdin, sc, c.model, threadID)
		if hsErr != nil {
			finalErr = hsErr
			return
		}
		if err := sendTurnStart(stdin, newThreadID, c.effort, &userMsg); err != nil {
			finalErr = err
			return
		}

		var (
			replies []genai.Reply
			usage   genai.Usage
		)
		for sc.Scan() {
			line := sc.Bytes()
			var probe lineProbe
			if json.Unmarshal(line, &probe) != nil {
				continue
			}
			// Skip JSON-RPC responses (have id, no method).
			if probe.ID != nil && probe.Method == "" {
				continue
			}
			if probe.Method == "" {
				continue
			}

			var msg jsonrpcMessage
			if json.Unmarshal(line, &msg) != nil {
				continue
			}

			switch msg.Method {
			case methodItemDelta:
				var p itemDeltaParams
				if json.Unmarshal(msg.Params, &p) == nil && p.Delta != "" {
					if !yield(genai.Reply{Text: p.Delta}) {
						return
					}
				}
			case methodReasoningSummaryTextDelta:
				var p reasoningSummaryTextDeltaParams
				if json.Unmarshal(msg.Params, &p) == nil && p.Delta != "" {
					if !yield(genai.Reply{Reasoning: p.Delta}) {
						return
					}
				}
			case methodItemCompleted:
				r := parseCompletedItem(msg.Params)
				if r != nil {
					replies = append(replies, *r)
				}
			case methodTokenUsageUpdated:
				var p tokenUsageUpdatedParams
				if json.Unmarshal(msg.Params, &p) == nil {
					accumulateUsage(&usage, &p.TokenUsage)
				}
			case methodTurnCompleted:
				var p turnCompletedParams
				if json.Unmarshal(msg.Params, &p) != nil {
					continue
				}
				if p.Turn.Status == "failed" || p.Turn.Status == "interrupted" {
					errMsg := "codex turn " + p.Turn.Status
					if p.Turn.Error != nil && p.Turn.Error.Message != "" {
						errMsg = p.Turn.Error.Message
					}
					finalErr = errors.New(errMsg)
					return
				}
				result = buildResult(replies, usage, newThreadID)
				return
			case methodErrorNotification:
				var p errorNotificationParams
				if json.Unmarshal(msg.Params, &p) == nil && !p.WillRetry && p.Error != nil {
					finalErr = fmt.Errorf("codex error: %s", p.Error.Message)
					return
				}
			}
		}
		if err := sc.Err(); err != nil {
			finalErr = fmt.Errorf("read stdout: %w", err)
		} else {
			finalErr = errors.New("codex exited without turn/completed")
		}
	}
	return seq, func() (genai.Result, error) {
		if finalErr == nil {
			finalErr = optsErr
		}
		return result, finalErr
	}
}

// initAndListModels performs the JSON-RPC initialize → initialized →
// model/list sequence and returns the model list. nextID is set to the last
// used request ID so the caller can continue numbering.
func initAndListModels(stdin io.Writer, sc *bufio.Scanner) ([]modelInfo, error) {
	// 1. Send initialize request.
	if err := writeJSON(stdin, jsonrpcRequest{
		JSONRPC: "2.0",
		ID:      1,
		Method:  "initialize",
		Params: initializeParams{
			ClientInfo:   clientInfo{Name: "genai-codex", Title: "genai-codex", Version: "1.0.0"},
			Capabilities: capabilities{OptOutNotificationMethods: optOutMethods},
		},
	}); err != nil {
		return nil, fmt.Errorf("write initialize: %w", err)
	}
	if _, err := readResponse(sc); err != nil {
		return nil, fmt.Errorf("read initialize response: %w", err)
	}

	// 2. Send initialized notification.
	if err := writeJSON(stdin, jsonrpcNotification{JSONRPC: "2.0", Method: "initialized"}); err != nil {
		return nil, fmt.Errorf("write initialized: %w", err)
	}

	// 3. Fetch model list.
	if err := writeJSON(stdin, jsonrpcRequest{JSONRPC: "2.0", ID: 2, Method: "model/list", Params: struct{}{}}); err != nil {
		return nil, fmt.Errorf("write model/list: %w", err)
	}
	mlData, err := readResponse(sc)
	if err != nil {
		return nil, fmt.Errorf("read model/list response: %w", err)
	}
	var mlResult modelListResult
	if err := json.Unmarshal(mlData, &mlResult); err != nil {
		return nil, fmt.Errorf("parse model/list result: %w", err)
	}
	return mlResult.Data, nil
}

// handshake performs the JSON-RPC initialize → initialized → model/list →
// thread/start (or thread/resume) sequence. Returns the thread ID.
func handshake(stdin io.Writer, sc *bufio.Scanner, mdl, resumeThreadID string) (string, error) {
	if _, err := initAndListModels(stdin, sc); err != nil {
		return "", err
	}

	// Send thread/start or thread/resume.
	var threadReq jsonrpcRequest
	if resumeThreadID != "" {
		threadReq = jsonrpcRequest{
			JSONRPC: "2.0",
			ID:      3,
			Method:  "thread/resume",
			Params:  threadResumeParams{ThreadID: resumeThreadID},
		}
	} else {
		threadReq = jsonrpcRequest{
			JSONRPC: "2.0",
			ID:      3,
			Method:  "thread/start",
			Params:  threadStartParams{Model: mdl},
		}
	}
	if err := writeJSON(stdin, threadReq); err != nil {
		return "", fmt.Errorf("write thread/start: %w", err)
	}
	respData, err := readResponse(sc)
	if err != nil {
		return "", fmt.Errorf("read thread/start response: %w", err)
	}

	var result threadStartResult
	if err := json.Unmarshal(respData, &result); err != nil {
		return "", fmt.Errorf("parse thread/start result: %w", err)
	}
	if result.Thread.ID == "" {
		if resumeThreadID != "" {
			return resumeThreadID, nil
		}
		return "", errors.New("thread/start response missing thread.id")
	}
	return result.Thread.ID, nil
}

// readResponse reads lines from the scanner until a JSON-RPC response (has
// "id" field) is found. Notifications are skipped.
func readResponse(sc *bufio.Scanner) (json.RawMessage, error) {
	for sc.Scan() {
		var msg jsonrpcMessage
		if json.Unmarshal(sc.Bytes(), &msg) != nil {
			continue
		}
		if !msg.isResponse() {
			continue // Skip notifications during handshake.
		}
		if msg.Error != nil {
			return nil, fmt.Errorf("JSON-RPC error %d: %s", msg.Error.Code, msg.Error.Message)
		}
		return msg.Result, nil
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	return nil, errors.New("codex exited during handshake")
}

// sendTurnStart sends a turn/start JSON-RPC request with the user message.
func sendTurnStart(stdin io.Writer, threadID string, effort ReasoningEffort, msg *genai.Message) error {
	input, err := msgToTurnInput(msg)
	if err != nil {
		return err
	}
	return writeJSON(stdin, jsonrpcRequest{
		JSONRPC: "2.0",
		ID:      100, // Arbitrary; we don't wait for this response.
		Method:  "turn/start",
		Params:  turnStartParams{ThreadID: threadID, Input: input, Effort: effort},
	})
}

// msgToTurnInput converts a genai.Message to turn/start input items.
func msgToTurnInput(msg *genai.Message) ([]turnInput, error) {
	var items []turnInput
	for i := range msg.Requests {
		req := &msg.Requests[i]
		if req.Text != "" {
			items = append(items, turnInput{Type: "text", Text: req.Text})
		}
		if !req.Doc.IsZero() {
			blk, err := docToTurnInput(req.Doc)
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

// docToTurnInput converts a genai.Doc to a turnInput for the codex CLI.
// Text documents are sent inline; images are sent as data URLs.
func docToTurnInput(doc genai.Doc) (turnInput, error) {
	if doc.URL != "" {
		return turnInput{Type: "image", URL: doc.URL}, nil
	}
	mimeType, data, err := doc.Read(10 * 1024 * 1024)
	if err != nil {
		return turnInput{}, fmt.Errorf("read doc: %w", err)
	}
	if strings.HasPrefix(mimeType, "text/") {
		return turnInput{Type: "text", Text: string(data)}, nil
	}
	if !strings.HasPrefix(mimeType, "image/") {
		return turnInput{}, fmt.Errorf("unsupported doc MIME type %q for codex (text and images only)", mimeType)
	}
	dataURL := "data:" + mimeType + ";base64," + base64.StdEncoding.EncodeToString(data)
	return turnInput{Type: "image", URL: dataURL}, nil
}

// readTurnSync reads notifications from the scanner until turn/completed.
func readTurnSync(sc *bufio.Scanner, threadID string) (genai.Result, error) {
	var (
		replies []genai.Reply
		usage   genai.Usage
	)
	for sc.Scan() {
		line := sc.Bytes()
		var probe lineProbe
		if json.Unmarshal(line, &probe) != nil {
			continue
		}
		// Skip JSON-RPC responses.
		if probe.ID != nil && probe.Method == "" {
			continue
		}
		if probe.Method == "" {
			continue
		}

		var msg jsonrpcMessage
		if json.Unmarshal(line, &msg) != nil {
			continue
		}

		switch msg.Method {
		case methodItemCompleted:
			r := parseCompletedItem(msg.Params)
			if r != nil {
				replies = append(replies, *r)
			}
		case methodTokenUsageUpdated:
			var p tokenUsageUpdatedParams
			if json.Unmarshal(msg.Params, &p) == nil {
				accumulateUsage(&usage, &p.TokenUsage)
			}
		case methodTurnCompleted:
			var p turnCompletedParams
			if json.Unmarshal(msg.Params, &p) != nil {
				continue
			}
			if p.Turn.Status == "failed" || p.Turn.Status == "interrupted" {
				errMsg := "codex turn " + p.Turn.Status
				if p.Turn.Error != nil && p.Turn.Error.Message != "" {
					errMsg = p.Turn.Error.Message
				}
				return genai.Result{}, errors.New(errMsg)
			}
			return buildResult(replies, usage, threadID), nil
		case methodErrorNotification:
			var p errorNotificationParams
			if json.Unmarshal(msg.Params, &p) == nil && !p.WillRetry && p.Error != nil {
				return genai.Result{}, fmt.Errorf("codex error: %s", p.Error.Message)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return genai.Result{}, fmt.Errorf("read stdout: %w", err)
	}
	return genai.Result{}, errors.New("codex exited without turn/completed")
}

// parseCompletedItem extracts a Reply from an item/completed notification if
// the item is an agentMessage or reasoning. Returns nil for other item types.
func parseCompletedItem(params json.RawMessage) *genai.Reply {
	var p itemParams
	if json.Unmarshal(params, &p) != nil {
		return nil
	}
	var h itemHeader
	if json.Unmarshal(p.Item, &h) != nil {
		return nil
	}
	switch h.Type {
	case "agentMessage":
		var item agentMessageItem
		if json.Unmarshal(p.Item, &item) != nil || item.Text == "" {
			return nil
		}
		return &genai.Reply{Text: item.Text}
	case "reasoning":
		var item reasoningItem
		if json.Unmarshal(p.Item, &item) != nil || len(item.Summary) == 0 {
			return nil
		}
		return &genai.Reply{Reasoning: strings.Join(item.Summary, "\n")}
	}
	return nil
}

// accumulateUsage adds the incremental (Last) token usage into the running total.
func accumulateUsage(usage *genai.Usage, tu *threadTokenUsage) {
	usage.InputTokens += tu.Last.InputTokens
	usage.InputCachedTokens += tu.Last.CachedInputTokens
	usage.OutputTokens += tu.Last.OutputTokens
	usage.ReasoningTokens += tu.Last.ReasoningOutputTokens
	usage.TotalTokens = usage.InputTokens + usage.OutputTokens
}

// buildResult constructs a genai.Result from collected replies, usage, and
// thread ID.
func buildResult(replies []genai.Reply, usage genai.Usage, threadID string) genai.Result {
	usage.FinishReason = genai.FinishedStop
	r := genai.Result{Usage: usage}
	r.Replies = append(r.Replies, replies...)
	if threadID != "" {
		r.Replies = append(r.Replies, genai.Reply{
			Opaque: map[string]any{threadIDKey: threadID},
		})
	}
	return r
}

// extractThreadID scans message history for a thread ID stored in
// Reply.Opaque by a previous codex call.
func extractThreadID(msgs genai.Messages) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		for j := range msgs[i].Replies {
			if id, ok := msgs[i].Replies[j].Opaque[threadIDKey].(string); ok && id != "" {
				return id
			}
		}
	}
	return ""
}

// lastUserMsg returns the last user message in msgs.
func lastUserMsg(msgs genai.Messages) (genai.Message, error) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role() == "user" {
			if len(msgs[i].Requests) == 0 {
				return genai.Message{}, errors.New("last user message has no content")
			}
			return msgs[i], nil
		}
	}
	return genai.Message{}, errors.New("no user message found in msgs")
}

// writeJSON marshals v as JSON and writes it followed by a newline.
func writeJSON(w io.Writer, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = w.Write(data)
	return err
}

// yieldNothing is an empty iterator used for early error returns in GenStream.
func yieldNothing(func(genai.Reply) bool) {}

// errFinish returns a finish function that always returns the given error.
func errFinish(err error) func() (genai.Result, error) {
	return func() (genai.Result, error) { return genai.Result{}, err }
}

// model is the genai.Model implementation for Codex models.
type model struct {
	id          string
	displayName string
}

func (m *model) GetID() string  { return m.id }
func (m *model) String() string { return m.displayName }
func (m *model) Context() int64 { return 200_000 }

// Compile-time interface checks.
var (
	_ genai.Provider     = (*Client)(nil)
	_ genai.ProviderPing = (*Client)(nil)
)
