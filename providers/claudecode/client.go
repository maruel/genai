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
	"path/filepath"
	"strconv"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/msgutil"
	"github.com/maruel/genai/providers/anthropic"
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

// cmdExecutor is the production executor backed by exec.Cmd.
type cmdExecutor struct {
	bin        string
	apiKeyAuth bool // keep ANTHROPIC_API_KEY in subprocess environment
}

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
	// Unset ANTHROPIC_API_KEY so that Claude Code uses OAuth (subscription)
	// instead of consuming API key credits. The key is typically set in the
	// parent process for HTTP-based providers (e.g. anthropic), not for the CLI.
	// Use GenOption.APIKeyAuth = true to keep it.
	strip := []string{"CLAUDECODE"}
	useOAuth := !e.apiKeyAuth && hasOAuth()
	if useOAuth {
		strip = append(strip, "ANTHROPIC_API_KEY")
	}
	env := environWithout(os.Environ(), strip...)
	// CLAUDE_CODE_SIMPLE=1 skips hooks, auto-memory, plugin sync, LSP, and
	// other auto-discovery not needed for programmatic use. However it also
	// disables OAuth, so only set it when authenticating via API key.
	if !useOAuth {
		env = append(env, "CLAUDE_CODE_SIMPLE=1")
	}
	cmd.Env = env
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
	exec           genai.Starter
	starterWrapper genai.ProviderOptionStarterWrapper
	bin            string
	model          string
	apiKeyAuth     bool // keep ANTHROPIC_API_KEY in subprocess environment

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
//   - ProviderOptionAPIKeyAuth — keep ANTHROPIC_API_KEY in the subprocess
//     environment. By default the key is stripped so Claude Code uses OAuth.
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
				c.model = "haiku"
			case genai.ModelGood:
				c.model = "sonnet"
			case genai.ModelSOTA:
				c.model = "opus"
			default:
				c.model = string(v)
			}
		case ProviderOptionAPIKeyAuth:
			c.apiKeyAuth = bool(v)
		case genai.ProviderOptionStarterWrapper:
			c.starterWrapper = v
		default:
			return nil, fmt.Errorf("unsupported provider option %T", opt)
		}
	}
	return c, nil
}

// ensureBin locates the claude binary on first call. It is safe for
// concurrent use.
func (c *Client) ensureBin() error {
	c.binOnce.Do(func() {
		bin, err := exec.LookPath("claude")
		if err != nil {
			if c.starterWrapper == nil {
				c.binErr = fmt.Errorf("claude CLI not found on PATH: %w", err)
				return
			}
			// Binary not found but a wrapper is set (e.g. test replay).
			// Use a placeholder; the wrapper may never call the inner starter.
			bin = "claude"
		}
		c.bin = bin
		s := genai.Starter((&cmdExecutor{bin: bin, apiKeyAuth: c.apiKeyAuth}).start)
		if c.starterWrapper != nil {
			s = c.starterWrapper(s)
		}
		c.exec = s
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
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	records, err := c.GenSyncRaw(ctx, msgs, opts...)
	if len(records) == 0 {
		return genai.Result{}, err
	}
	var initSessionID string
	var asstBlocks []OutputContentBlock
	var summaries []string
	for _, line := range records {
		var b OutputTypeProbe
		if json.Unmarshal(line, &b) != nil {
			continue
		}
		switch b.Type {
		case OutputSystem:
			if b.Subtype == string(SystemInit) {
				var m OutputInitMsg
				if json.Unmarshal(line, &m) == nil {
					initSessionID = m.SessionID
				}
			} else if b.Subtype == string(SystemPostTurnSummary) {
				var m OutputPostTurnSummaryMsg
				if json.Unmarshal(line, &m) == nil {
					if s := m.Reasoning(); s != "" {
						summaries = append(summaries, s)
					}
				}
			}
		case OutputAssistant:
			var asst OutputAssistantMsg
			if e := internal.UnmarshalJSON(line, &asst); e != nil {
				return genai.Result{}, errors.Join(fmt.Errorf("parse assistant: %w", e), err)
			}
			asstBlocks = append(asstBlocks, asst.Message.Content...)
		case OutputResult:
			var res OutputResultMsg
			if e := internal.UnmarshalJSON(line, &res); e != nil {
				return genai.Result{}, errors.Join(fmt.Errorf("parse result: %w", e), err)
			}
			if e := res.AsError(); e != nil {
				return genai.Result{}, errors.Join(e, err)
			}
			return buildResult(&res, asstBlocks, summaries, initSessionID), err
		default:
		}
	}
	return genai.Result{}, errors.Join(errors.New("claude exited without a result message"), err)
}

// GenSyncRaw returns the raw Claude Code stdout records for one synchronous turn.
func (c *Client) GenSyncRaw(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (records []json.RawMessage, err error) {
	if err := c.ensureBin(); err != nil {
		return nil, err
	}
	co, optsErr := parseOpts(opts)
	if optsErr != nil {
		var uerr *base.ErrNotSupported
		if !errors.As(optsErr, &uerr) {
			return nil, optsErr
		}
		// Proceed with the call; return ErrNotSupported after getting results.
		defer func() {
			if err == nil {
				err = optsErr
			}
		}()
	}
	userMsg, err := msgutil.LastUserMsg(msgs)
	if err != nil {
		return nil, err
	}
	sessionID := msgutil.ExtractOpaqueID(msgs, sessionIDKey)
	args := c.buildArgs(&co, sessionID, false)

	stdin, stdout, wait, err := c.exec(ctx, args)
	if err != nil {
		return nil, err
	}
	defer func() {
		// Always close stdin so the subprocess sees EOF and exits.
		_ = stdin.Close()
		err = errors.Join(err, wait())
	}()

	if err := writeInitialize(stdin, &co); err != nil {
		return nil, err
	}
	if err := writeUserMsg(stdin, &userMsg); err != nil {
		return nil, err
	}

	sc := newScanner(stdout)
	for sc.Scan() {
		line := sc.Bytes()
		records = append(records, append(json.RawMessage(nil), line...))
		var b OutputTypeProbe
		if json.Unmarshal(line, &b) != nil {
			continue
		}
		switch b.Type {
		case OutputControlRequest:
			if err := handleControlRequest(ctx, stdin, co.controlHandler, line); err != nil {
				return records, err
			}
		case OutputResult:
			return records, nil
		}
	}
	if err := sc.Err(); err != nil {
		return records, fmt.Errorf("read stdout: %w", err)
	}
	return records, errors.New("claude exited without a result message")
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
	userMsg, err := msgutil.LastUserMsg(msgs)
	if err != nil {
		return yieldNothing, errFinish(err)
	}
	sessionID := msgutil.ExtractOpaqueID(msgs, sessionIDKey)
	args := c.buildArgs(&co, sessionID, true)

	var (
		result   genai.Result
		finalErr error
	)
	seq := func(yield func(genai.Reply) bool) {
		stdin, stdout, wait, startErr := c.exec(ctx, args)
		if startErr != nil {
			finalErr = startErr
			return
		}
		defer func() {
			_ = stdin.Close()
			finalErr = errors.Join(finalErr, wait())
		}()

		if err := writeInitialize(stdin, &co); err != nil {
			finalErr = err
			return
		}
		if err := writeUserMsg(stdin, &userMsg); err != nil {
			finalErr = err
			return
		}

		sc := newScanner(stdout)
		var initSessionID string
		var asstBlocks []OutputContentBlock
		var summaries []string
		var streamUsage MsgUsage
		for sc.Scan() {
			line := sc.Bytes()
			var b OutputTypeProbe
			if json.Unmarshal(line, &b) != nil {
				continue
			}
			switch b.Type {
			case OutputSystem:
				if b.Subtype == string(SystemInit) {
					var m OutputInitMsg
					if json.Unmarshal(line, &m) == nil {
						initSessionID = m.SessionID
					}
				} else if b.Subtype == string(SystemPostTurnSummary) {
					var m OutputPostTurnSummaryMsg
					if json.Unmarshal(line, &m) == nil {
						if s := m.Reasoning(); s != "" {
							summaries = append(summaries, s)
							if !yield(genai.Reply{Reasoning: s}) {
								return
							}
						}
					}
				}
			case OutputStreamEvent:
				var ev OutputStreamEventMsg
				if json.Unmarshal(line, &ev) != nil {
					continue
				}
				if ev.Event.Type == "error" {
					finalErr = errors.New("claude stream error")
					return
				}
				if ev.Event.Type == "message_delta" && !ev.Event.Usage.IsZero() {
					streamUsage = ev.Event.Usage
				}
				if ev.Event.Delta.Type != "" {
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
			case OutputAssistant:
				var asst OutputAssistantMsg
				if err := internal.UnmarshalJSON(line, &asst); err != nil {
					finalErr = fmt.Errorf("parse assistant: %w", err)
					return
				}
				asstBlocks = append(asstBlocks, asst.Message.Content...)
			case OutputResult:
				var res OutputResultMsg
				if err := internal.UnmarshalJSON(line, &res); err != nil {
					finalErr = fmt.Errorf("parse result: %w", err)
					return
				}
				if err := res.AsError(); err != nil {
					finalErr = err
					return
				}
				result = buildResult(&res, asstBlocks, summaries, initSessionID)
				if result.Usage.ReasoningTokens == 0 {
					result.Usage.ReasoningTokens = streamUsage.OutputTokensDetails.ThinkingTokens
				}
				return
			case OutputControlRequest:
				if err := handleControlRequest(ctx, stdin, co.controlHandler, line); err != nil {
					finalErr = err
					return
				}
			default:
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
func (c *Client) buildArgs(co *callOpts, sessionID string, stream bool) []string {
	args := []string{
		"-p",
		"--verbose",
		"--input-format", "stream-json",
		"--output-format", "stream-json",
		// Ignore user-configured MCP servers; this provider only needs the
		// built-in tools. Without this flag, MCP servers from the user's
		// settings would silently load into the subprocess.
		"--strict-mcp-config",
		// Disable Chrome integration; the subprocess is headless.
		"--no-chrome",
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

	if co.controlHandler != nil {
		args = append(args, "--permission-prompt-tool", "stdio")
	}

	// Permission mode: default to "bypassPermissions" when tools are enabled
	// and the caller has not installed a control handler. With a handler, let
	// Claude Code emit can_use_tool requests over stdio.
	pm := co.permissionMode
	if pm == "" && len(co.tools) > 0 && co.controlHandler == nil {
		pm = "bypassPermissions"
	}
	if pm != "" {
		args = append(args, "--permission-mode", pm)
	}

	// Optional reasoning effort.
	if co.effort != "" {
		args = append(args, "--effort", co.effort)
	}

	// Optional system prompt.
	if co.systemPrompt != "" {
		args = append(args, "--system-prompt", co.systemPrompt)
	}

	return args
}

func handleControlRequest(ctx context.Context, w io.Writer, h ControlHandler, line []byte) error {
	if h == nil {
		return errors.New("claude requested host control but no control handler is configured")
	}
	var req OutputControlRequestMsg
	if err := internal.UnmarshalJSON(line, &req); err != nil {
		return fmt.Errorf("parse control request: %w", err)
	}
	res, err := h(ctx, req)
	if err != nil {
		return fmt.Errorf("handle control request %q: %w", req.RequestID, err)
	}
	if res.Type == "" {
		res.Type = InputControlResponse
	}
	if res.Type != InputControlResponse {
		return fmt.Errorf("control response type = %q, want %q", res.Type, InputControlResponse)
	}
	if res.Response.RequestID == "" {
		res.Response.RequestID = req.RequestID
	}
	switch res.Response.Subtype {
	case ControlResponseSuccess, ControlResponseError:
	default:
		return fmt.Errorf("control response subtype = %q, want success or error", res.Response.Subtype)
	}
	return writeControlResponse(w, &res)
}

func writeControlResponse(w io.Writer, m *InputControlResponseMsg) error {
	data, err := json.Marshal(m)
	if err != nil {
		return fmt.Errorf("marshal control response: %w", err)
	}
	_, err = fmt.Fprintf(w, "%s\n", data)
	return err
}

// writeUserMsg encodes a genai.Message as NDJSON and writes it to stdin.
func writeUserMsg(w io.Writer, msg *genai.Message) error {
	blocks := make([]InputContentBlock, 0, len(msg.Requests))
	for i := range msg.Requests {
		req := &msg.Requests[i]
		if !req.Doc.IsZero() {
			blk, err := docToInputBlock(req.Doc)
			if err != nil {
				return err
			}
			blocks = append(blocks, blk)
		} else if req.Text != "" {
			blocks = append(blocks, InputContentBlock{Type: "text", Text: req.Text})
		}
	}
	if len(blocks) == 0 {
		return errors.New("user message has no content")
	}
	m := InputUserMsg{
		Type:    InputUser,
		Message: InputUserContent{Role: "user", Content: blocks},
	}
	data, err := json.Marshal(m)
	if err != nil {
		return fmt.Errorf("marshal user message: %w", err)
	}
	_, err = fmt.Fprintf(w, "%s\n", data)
	return err
}

func writeInitialize(w io.Writer, co *callOpts) error {
	if !co.progressSummaries {
		return nil
	}
	req, err := json.Marshal(ControlReqInitialize{
		Subtype:                ControlInitialize,
		AgentProgressSummaries: true,
	})
	if err != nil {
		return fmt.Errorf("marshal initialize request body: %w", err)
	}
	m := InputControlRequestMsg{
		Type:      InputControlRequest,
		RequestID: "genai-init",
		Request:   json.RawMessage(req),
	}
	data, err := json.Marshal(m)
	if err != nil {
		return fmt.Errorf("marshal initialize request: %w", err)
	}
	_, err = fmt.Fprintf(w, "%s\n", data)
	return err
}

// docToInputBlock converts a genai.Doc to an InputContentBlock for the claude CLI.
// Text documents are inlined as text blocks; images are sent as base64 or URL.
func docToInputBlock(doc genai.Doc) (InputContentBlock, error) {
	if doc.URL != "" {
		return InputContentBlock{
			Type:   "image",
			Source: anthropic.Source{Type: "url", URL: doc.URL},
		}, nil
	}
	mimeType, data, err := doc.Read(10 * 1024 * 1024)
	if err != nil {
		return InputContentBlock{}, fmt.Errorf("read doc: %w", err)
	}
	switch {
	case strings.HasPrefix(mimeType, "text/"):
		return InputContentBlock{Type: "text", Text: string(data)}, nil
	case strings.HasPrefix(mimeType, "image/"):
		return InputContentBlock{
			Type: "image",
			Source: anthropic.Source{
				Type:      "base64",
				MediaType: mimeType,
				Data:      base64.StdEncoding.EncodeToString(data),
			},
		}, nil
	default:
		return InputContentBlock{}, fmt.Errorf("unsupported doc MIME type %q", mimeType)
	}
}

// buildResult converts a resultMsg and accumulated assistant content blocks into a genai.Result.
func buildResult(res *OutputResultMsg, asstBlocks []OutputContentBlock, summaries []string, sessionID string) genai.Result {
	r := genai.Result{
		Usage: genai.Usage{
			InputTokens:       res.Usage.InputTokens,
			InputCachedTokens: res.Usage.CacheReadInputTokens,
			ReasoningTokens:   res.Usage.OutputTokensDetails.ThinkingTokens,
			OutputTokens:      res.Usage.OutputTokens,
			TotalTokens:       res.Usage.InputTokens + res.Usage.OutputTokens,
			FinishReason:      mapFinishReason(res.StopReason),
			ServiceTier:       res.Usage.ServiceTier,
		},
	}

	for _, s := range summaries {
		r.Replies = append(r.Replies, genai.Reply{Reasoning: s})
	}

	// Prefer text content from the assistant messages (present in both sync and
	// stream modes) over the plain-text result field. Claude Code may split
	// thinking and text into separate assistant messages, so we accumulate
	// content blocks from all assistant messages.
	for i := range asstBlocks {
		blk := &asstBlocks[i]
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

// claudeConfig is the subset of ~/.claude/claude.json we inspect.
type claudeConfig struct {
	OAuthAccount *json.RawMessage `json:"oauthAccount"`
}

// hasOAuth reports whether Claude Code has an OAuth session configured in
// ~/.claude/claude.json. When true, stripping ANTHROPIC_API_KEY is safe
// because Claude Code can fall back to OAuth. When false, the API key may be
// the only auth method available.
func hasOAuth() bool {
	home, err := os.UserHomeDir()
	if err != nil {
		return false
	}
	f, err := os.Open(filepath.Join(home, ".claude", "claude.json"))
	if err != nil {
		return false
	}
	defer func() { _ = f.Close() }()
	var cfg claudeConfig
	if json.NewDecoder(f).Decode(&cfg) != nil {
		return false
	}
	return cfg.OAuthAccount != nil
}

// environWithout returns env with all entries whose key matches any of the
// given names removed, so the subprocess inherits the full environment minus
// those variables.
func environWithout(env []string, names ...string) []string {
	out := make([]string, 0, len(env))
	for _, e := range env {
		skip := false
		for _, name := range names {
			if strings.HasPrefix(e, name+"=") {
				skip = true
				break
			}
		}
		if !skip {
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
