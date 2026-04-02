// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package claudecode implements a genai provider backed by the Claude Code CLI.
//
// Instead of making HTTP requests directly, it launches the `claude` binary as
// a subprocess and communicates over stdin/stdout using the NDJSON stream-json
// protocol (--input-format stream-json --output-format stream-json).
//
// See https://docs.anthropic.com/en/docs/claude-code/cli-reference for CLI
// reference documentation.
//
// Anthropic is fairly prompt to ban you from using their CLI, so this provider
// is not recommended for production use, especially if used 24/7. It's meant for
// one off use, like "Generate a git commit description for this git diff".
//
// # Safe defaults
//
// To minimize side-effects the provider disables all tools, slash-command
// skills, and project/local CLAUDE.md loading by default. Individual
// capabilities can be unlocked per-call with the GenOption* types in this
// package.
//
// # Session / multi-turn
//
// Each GenSync or GenStream call launches a fresh subprocess. The session ID is
// always returned inside Reply.Opaque["session_id"]. When the message history
// contains a previous session ID, it is automatically picked up: --resume <id>
// is passed and only the last user message is sent.
package claudecode

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
	"strconv"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for the Claude Code CLI provider.
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

// executor abstracts subprocess creation so tests can inject a recording or
// fake implementation.
//
// start launches the claude subprocess and returns:
//   - stdin: write the NDJSON user message, then close to signal end-of-input
//   - stdout: raw reader over the subprocess stdout; caller wraps in a Scanner
//   - wait: call after stdout is fully consumed to release process resources
type executor interface {
	start(ctx context.Context, args []string) (stdin io.WriteCloser, stdout io.ReadCloser, wait func() error, err error)
}

// cmdExecutor is the production executor backed by exec.Cmd.
type cmdExecutor struct{ bin string }

func (e *cmdExecutor) start(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	// Fresh empty dir: claude sees no project files, CLAUDE.md, or history.
	// Cleaned up by the wait closure below.
	dir, err := os.MkdirTemp("", "genai-claudecode-*")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("create temp dir: %w", err)
	}
	cmd := exec.CommandContext(ctx, e.bin, args...)
	cmd.Dir = dir
	// Unset CLAUDECODE so that claude can be launched even when the caller is
	// itself running inside a Claude Code session (nested session guard).
	// Set CLAUDE_CODE_SIMPLE=1 to skip hooks, auto-memory, plugin sync, LSP,
	// and other auto-discovery that is not relevant for programmatic use.
	// This is the same env var that --bare sets internally, but without the
	// --bare auth restriction (which disallows OAuth).
	cmd.Env = append(environWithout(os.Environ(), "CLAUDECODE"), "CLAUDE_CODE_SIMPLE=1")
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, nil, errors.Join(fmt.Errorf("stdin pipe: %w", err), os.RemoveAll(dir))
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, nil, errors.Join(fmt.Errorf("stdout pipe: %w", err), os.RemoveAll(dir))
	}
	if err := cmd.Start(); err != nil {
		return nil, nil, nil, errors.Join(fmt.Errorf("start claude: %w", err), os.RemoveAll(dir))
	}
	return stdin, stdout, func() error {
		return errors.Join(cmd.Wait(), os.RemoveAll(dir))
	}, nil
}

// newScanner wraps an io.ReadCloser in a bufio.Scanner sized for the
// stream-json protocol (lines up to 32 MB for base64-encoded images).
func newScanner(r io.Reader) *bufio.Scanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 32<<20) // 1 MB initial, 32 MB max
	return sc
}

// Client is a genai provider that delegates to the local `claude` CLI.
type Client struct {
	base.NotImplemented
	exec    executor
	bin     string // resolved lazily by ensureBin; empty when exec is a test fake
	model   string
	binOnce sync.Once
	binErr  error
}

// New creates a Client for the `claude` CLI.
//
// The binary is located lazily on the first call to GenSync, GenStream, or
// Ping, so New succeeds even when the CLI is not installed. This allows
// introspection methods (Name, Scoreboard, ListModels, …) to work without
// requiring the binary.
//
// Supported ProviderOptions:
//   - genai.ProviderOptionModel — model alias ("opus", "sonnet", "haiku") or full ID.
//     Use genai.ModelCheap, genai.ModelGood, or genai.ModelSOTA for automatic selection.
func New(opts ...genai.ProviderOption) (*Client, error) {
	c := &Client{}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionModel:
			switch v {
			case genai.ModelCheap:
				c.model = "haiku"
			case genai.ModelGood:
				c.model = "sonnet"
			case genai.ModelSOTA:
				c.model = "opus"
			default:
				c.model = string(v)
			}
		default:
			return nil, fmt.Errorf("unsupported provider option %T", opt)
		}
	}
	return c, nil
}

// ensureBin locates the claude binary on first call. It is safe for
// concurrent use. When exec is already set (e.g. by test code), it is a
// no-op.
func (c *Client) ensureBin() error {
	c.binOnce.Do(func() {
		if c.exec != nil {
			return
		}
		bin, err := exec.LookPath("claude")
		if err != nil {
			c.binErr = fmt.Errorf("claude CLI not found on PATH: %w", err)
			return
		}
		c.bin = bin
		c.exec = &cmdExecutor{bin: bin}
	})
	return c.binErr
}

// Name implements genai.Provider.
func (c *Client) Name() string { return "claudecode" }

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

// HTTPClient implements genai.Provider.  The CLI provider does not use HTTP.
func (c *Client) HTTPClient() *http.Client { return nil }

// Ping implements genai.ProviderPing by running `claude --version`.
func (c *Client) Ping(ctx context.Context) error {
	if err := c.ensureBin(); err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, c.bin, "--version")
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("claude --version: %w", err)
	}
	if !strings.HasPrefix(string(out), "claude") && !strings.Contains(string(out), "Claude Code") {
		return fmt.Errorf("unexpected claude --version output: %q", string(out))
	}
	return nil
}

// ListModels implements genai.Provider with the unversioned model aliases
// supported by the claude CLI --model flag.
func (c *Client) ListModels(_ context.Context) ([]genai.Model, error) {
	// TODO: we can't fetch the API since the model available does not always match.
	return []genai.Model{
		&model{id: "opus", displayName: "Claude Opus (latest)"},
		&model{id: "sonnet", displayName: "Claude Sonnet (latest)"},
		&model{id: "haiku", displayName: "Claude Haiku (latest)"},
	}, nil
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (r genai.Result, err error) {
	if err := c.ensureBin(); err != nil {
		return genai.Result{}, err
	}
	co, optsErr := parseOpts(opts)
	if optsErr != nil {
		var uerr *base.ErrNotSupported
		if !errors.As(optsErr, &uerr) {
			return genai.Result{}, optsErr
		}
		// Proceed with the call; return ErrNotSupported after getting results.
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
	sessionID := extractSessionID(msgs)
	args := c.buildArgs(co, sessionID, false)

	stdin, stdout, wait, err := c.exec.start(ctx, args)
	if err != nil {
		return genai.Result{}, err
	}
	defer func() {
		// Always close stdin so the subprocess sees EOF and exits.
		_ = stdin.Close()
		err = errors.Join(err, wait())
	}()

	if err := writeUserMsg(stdin, &userMsg); err != nil {
		return genai.Result{}, err
	}

	sc := newScanner(stdout)
	var initSessionID string
	var lastAsst outputAssistant
	for sc.Scan() {
		line := sc.Bytes()
		var b outputTypeProbe
		if json.Unmarshal(line, &b) != nil {
			continue
		}
		switch b.Type {
		case outputTypeSystem:
			if b.Subtype == string(systemInit) {
				var m outputInit
				if json.Unmarshal(line, &m) == nil {
					initSessionID = m.SessionID
				}
			}
		case outputTypeAssistant:
			if err := json.Unmarshal(line, &lastAsst); err != nil {
				return genai.Result{}, fmt.Errorf("parse assistant: %w", err)
			}
		case outputTypeResult:
			var res outputResult
			if err := json.Unmarshal(line, &res); err != nil {
				return genai.Result{}, fmt.Errorf("parse result: %w", err)
			}
			if res.IsError {
				errMsg := res.Result
				if errMsg == "" && len(res.Errors) > 0 {
					errMsg = strings.Join(res.Errors, "; ")
				}
				return genai.Result{}, fmt.Errorf("claude error (%s): %s", res.Subtype, errMsg)
			}
			return buildResult(&res, &lastAsst, initSessionID), nil
		}
	}
	if err := sc.Err(); err != nil {
		return genai.Result{}, fmt.Errorf("read stdout: %w", err)
	}
	return genai.Result{}, errors.New("claude exited without a result message")
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if err := c.ensureBin(); err != nil {
		return yieldNothing, errFinish(err)
	}
	co, optsErr := parseOpts(opts)
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
	sessionID := extractSessionID(msgs)
	args := c.buildArgs(co, sessionID, true)

	var (
		result   genai.Result
		finalErr error
	)
	seq := func(yield func(genai.Reply) bool) {
		stdin, stdout, wait, startErr := c.exec.start(ctx, args)
		if startErr != nil {
			finalErr = startErr
			return
		}
		defer func() {
			_ = stdin.Close()
			finalErr = errors.Join(finalErr, wait())
		}()

		if err := writeUserMsg(stdin, &userMsg); err != nil {
			finalErr = err
			return
		}

		sc := newScanner(stdout)
		var initSessionID string
		var lastAsst outputAssistant
		for sc.Scan() {
			line := sc.Bytes()
			var b outputTypeProbe
			if json.Unmarshal(line, &b) != nil {
				continue
			}
			switch b.Type {
			case outputTypeSystem:
				if b.Subtype == string(systemInit) {
					var m outputInit
					if json.Unmarshal(line, &m) == nil {
						initSessionID = m.SessionID
					}
				}
			case outputTypeStreamEvent:
				var ev outputStreamEvent
				if json.Unmarshal(line, &ev) != nil {
					continue
				}
				if ev.Event.Type == "error" {
					finalErr = errors.New("claude stream error")
					return
				}
				if ev.Event.Delta != nil {
					switch ev.Event.Delta.Type {
					case "text_delta":
						if ev.Event.Delta.Text != "" {
							if !yield(genai.Reply{Text: ev.Event.Delta.Text}) {
								return
							}
						}
					case "thinking_delta":
						if ev.Event.Delta.Thinking != "" {
							if !yield(genai.Reply{Reasoning: ev.Event.Delta.Thinking}) {
								return
							}
						}
					}
				}
			case outputTypeAssistant:
				if err := json.Unmarshal(line, &lastAsst); err != nil {
					finalErr = fmt.Errorf("parse assistant: %w", err)
					return
				}
			case outputTypeResult:
				var res outputResult
				if err := json.Unmarshal(line, &res); err != nil {
					finalErr = fmt.Errorf("parse result: %w", err)
					return
				}
				if res.IsError {
					errMsg := res.Result
					if errMsg == "" && len(res.Errors) > 0 {
						errMsg = strings.Join(res.Errors, "; ")
					}
					finalErr = fmt.Errorf("claude error (%s): %s", res.Subtype, errMsg)
					return
				}
				result = buildResult(&res, &lastAsst, initSessionID)
			}
		}
		if err := sc.Err(); err != nil {
			finalErr = fmt.Errorf("read stdout: %w", err)
		}
	}
	return seq, func() (genai.Result, error) {
		if finalErr == nil {
			finalErr = optsErr
		}
		return result, finalErr
	}
}

// buildArgs constructs the claude CLI argument list for a call.
func (c *Client) buildArgs(co callOpts, sessionID string, stream bool) []string {
	args := []string{
		"-p",
		"--verbose",
		"--input-format", "stream-json",
		"--output-format", "stream-json",
		// Ignore user-configured MCP servers; this provider only needs the
		// built-in tools. Without this flag, MCP servers from the user's
		// settings would silently load into the subprocess.
		"--strict-mcp-config",
	}

	// Tools: disabled by default.
	if co.tools == nil {
		args = append(args, "--tools", "")
	} else {
		args = append(args, "--tools", strings.Join(co.tools, ","))
	}

	// Skills: disabled by default.
	if !co.skills {
		args = append(args, "--disable-slash-commands")
	}

	// Settings: no user settings by default (no ~/.claude/CLAUDE.md or settings.json).
	// We always run in a fresh empty temp dir so project and local settings never
	// exist there either, making this effectively zero settings.
	// GenOptionProjectSettings opts back in.
	if !co.projSettings {
		args = append(args, "--setting-sources", "project,local")
	}

	// Session persistence: enabled automatically when resuming a previous session.
	if sessionID != "" {
		args = append(args, "--resume", sessionID)
	} else {
		args = append(args, "--no-session-persistence")
	}

	// Partial streaming for GenStream.
	if stream {
		args = append(args, "--include-partial-messages")
	}

	// Optional model.
	if c.model != "" {
		args = append(args, "--model", c.model)
	}

	// Optional budget cap.
	if co.maxBudgetUSD > 0 {
		args = append(args, "--max-budget-usd", strconv.FormatFloat(co.maxBudgetUSD, 'f', 4, 64))
	}

	// Optional permission mode.
	if co.permissionMode != "" {
		args = append(args, "--permission-mode", co.permissionMode)
	}

	// Optional system prompt.
	if co.systemPrompt != "" {
		args = append(args, "--system-prompt", co.systemPrompt)
	}

	return args
}

// extractSessionID scans message history for a session ID stored in
// Reply.Opaque by a previous claudecode call.
func extractSessionID(msgs genai.Messages) string {
	for i := len(msgs) - 1; i >= 0; i-- {
		for j := range msgs[i].Replies {
			if id, ok := msgs[i].Replies[j].Opaque[sessionIDKey].(string); ok && id != "" {
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

// writeUserMsg encodes a genai.Message as NDJSON and writes it to stdin.
// Text-only messages use a plain string for content; messages that include
// images use an array of content blocks (matching the Anthropic content-block
// wire format that the Claude Code CLI proxies).
func writeUserMsg(w io.Writer, msg *genai.Message) error {
	var content any

	// Determine whether any request carries a document (image).
	hasDoc := false
	for i := range msg.Requests {
		if !msg.Requests[i].Doc.IsZero() {
			hasDoc = true
			break
		}
	}

	if !hasDoc {
		// Text-only: concatenate all text into a plain string.
		var sb strings.Builder
		for i := range msg.Requests {
			sb.WriteString(msg.Requests[i].Text)
		}
		if sb.Len() == 0 {
			return errors.New("user message has no content")
		}
		content = sb.String()
	} else {
		// Multi-modal: build an array of typed content blocks.
		blocks := make([]inputContentBlock, 0, len(msg.Requests))
		for i := range msg.Requests {
			req := &msg.Requests[i]
			if !req.Doc.IsZero() {
				blk, err := docToInputBlock(req.Doc)
				if err != nil {
					return err
				}
				blocks = append(blocks, blk)
			} else if req.Text != "" {
				blocks = append(blocks, inputContentBlock{Type: "text", Text: req.Text})
			}
		}
		if len(blocks) == 0 {
			return errors.New("user message has no content")
		}
		content = blocks
	}

	m := inputUser{
		Type:    inputTypeUser,
		Message: inputUserContent{Role: "user", Content: content},
	}
	data, err := json.Marshal(m)
	if err != nil {
		return fmt.Errorf("marshal user message: %w", err)
	}
	_, err = fmt.Fprintf(w, "%s\n", data)
	return err
}

// docToInputBlock converts a genai.Doc to an inputContentBlock for the claude CLI.
// Text documents are inlined as text blocks; images are sent as base64 or URL.
func docToInputBlock(doc genai.Doc) (inputContentBlock, error) {
	if doc.URL != "" {
		return inputContentBlock{
			Type:   "image",
			Source: &inputImageSource{Type: "url", URL: doc.URL},
		}, nil
	}
	mimeType, data, err := doc.Read(10 * 1024 * 1024)
	if err != nil {
		return inputContentBlock{}, fmt.Errorf("read doc: %w", err)
	}
	switch {
	case strings.HasPrefix(mimeType, "text/"):
		return inputContentBlock{Type: "text", Text: string(data)}, nil
	case strings.HasPrefix(mimeType, "image/"):
		return inputContentBlock{
			Type: "image",
			Source: &inputImageSource{
				Type:      "base64",
				MediaType: mimeType,
				Data:      base64.StdEncoding.EncodeToString(data),
			},
		}, nil
	default:
		return inputContentBlock{}, fmt.Errorf("unsupported doc MIME type %q", mimeType)
	}
}

// buildResult converts a resultMsg and the last assistantMsg into a genai.Result.
func buildResult(res *outputResult, asst *outputAssistant, sessionID string) genai.Result {
	r := genai.Result{
		Usage: genai.Usage{
			InputTokens:       res.Usage.InputTokens,
			InputCachedTokens: res.Usage.CacheReadInputTokens,
			OutputTokens:      res.Usage.OutputTokens,
			TotalTokens:       res.Usage.InputTokens + res.Usage.OutputTokens,
			FinishReason:      mapFinishReason(res.StopReason),
			ServiceTier:       res.Usage.ServiceTier,
		},
	}

	// Prefer text content from the assistant message (present in both sync and
	// stream modes) over the plain-text result field.
	if asst != nil {
		for _, blk := range asst.Message.Content {
			switch blk.Type {
			case "text":
				if blk.Text != "" {
					r.Replies = append(r.Replies, genai.Reply{Text: blk.Text})
				}
			case "thinking":
				if blk.Thinking != "" {
					r.Replies = append(r.Replies, genai.Reply{Reasoning: blk.Thinking})
				}
			}
		}
	}

	// Fall back to the result text field if no content was found.
	if len(r.Replies) == 0 && res.Result != "" {
		r.Replies = append(r.Replies, genai.Reply{Text: res.Result})
	}

	// Always store session ID and metadata so the caller can resume and inspect.
	opaque := map[string]any{}
	if sessionID != "" {
		opaque[sessionIDKey] = sessionID
	}
	if res.TotalCostUSD > 0 {
		opaque["total_cost_usd"] = res.TotalCostUSD
	}
	if res.DurationMs > 0 {
		opaque["duration_ms"] = res.DurationMs
	}
	if res.DurationAPIMs > 0 {
		opaque["duration_api_ms"] = res.DurationAPIMs
	}
	if res.NumTurns > 0 {
		opaque["num_turns"] = res.NumTurns
	}
	if len(opaque) > 0 {
		r.Replies = append(r.Replies, genai.Reply{
			Opaque: opaque,
		})
	}

	return r
}

// mapFinishReason converts a claude stop_reason string to a genai.FinishReason.
func mapFinishReason(r string) genai.FinishReason {
	switch r {
	case "max_tokens":
		return genai.FinishedLength
	case "tool_use":
		return genai.FinishedToolCalls
	default:
		return genai.FinishedStop
	}
}

// environWithout returns os.Environ() with all entries whose key equals name
// removed, so the subprocess inherits the full environment minus that variable.
func environWithout(env []string, name string) []string {
	prefix := name + "="
	out := make([]string, 0, len(env))
	for _, e := range env {
		if !strings.HasPrefix(e, prefix) {
			out = append(out, e)
		}
	}
	return out
}

// yieldNothing is an empty iterator used for early error returns in GenStream.
func yieldNothing(func(genai.Reply) bool) {}

// errFinish returns a finish function that always returns the given error.
func errFinish(err error) func() (genai.Result, error) {
	return func() (genai.Result, error) { return genai.Result{}, err }
}

// model is the genai.Model implementation for Claude Code models.
type model struct {
	id          string
	displayName string
}

func (m *model) GetID() string  { return m.id }
func (m *model) String() string { return m.displayName }
func (m *model) Context() int64 { return 0 }

// Compile-time interface checks.
var (
	_ genai.Provider     = (*Client)(nil)
	_ genai.ProviderPing = (*Client)(nil)
)
