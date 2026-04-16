// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package pi implements a genai provider backed by the Pi coding agent CLI.
//
// Instead of making HTTP requests directly, it launches `pi --mode rpc
// --no-session` as a subprocess and communicates over stdin/stdout using Pi's
// custom JSONL protocol.
//
// See https://github.com/badlogic/pi-mono for source and
// https://www.npmjs.com/package/@mariozechner/pi-coding-agent for the npm
// package.
//
// # Protocol
//
// Pi uses a type-dispatched JSONL protocol (not JSON-RPC 2.0). There is no
// handshake; the subprocess is immediately ready to accept commands. Commands
// are sent as JSON lines on stdin, events and responses are emitted on stdout.
//
// # Session / multi-turn
//
// Each GenSync or GenStream call launches a fresh subprocess. Pi supports
// multi-turn within a single process, but we spawn fresh for simplicity
// (matching the codex/opencode pattern).
package pi

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

// Scoreboard returns the Pi provider's known capabilities.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// cmdExecutor is the production executor backed by exec.Cmd.
type cmdExecutor struct{ bin string }

func (e *cmdExecutor) start(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	dir, err := os.MkdirTemp("", "genai-pi-*")
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
		return nil, nil, nil, errors.Join(fmt.Errorf("start pi: %w", err), os.RemoveAll(dir))
	}
	return stdin, stdout, func() error {
		return errors.Join(cmd.Wait(), os.RemoveAll(dir))
	}, nil
}

// newScanner wraps an io.Reader in a bufio.Scanner sized for large JSONL
// messages (up to 32 MB).
func newScanner(r io.Reader) *bufio.Scanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 32<<20) // 1 MB initial, 32 MB max
	return sc
}

// Client is a genai provider that delegates to the local `pi` CLI.
type Client struct {
	base.NotImplemented
	exec           genai.Starter
	starterWrapper genai.ProviderOptionStarterWrapper
	bin            string
	model          string

	binOnce sync.Once
	binErr  error
}

// New creates a Client for the `pi` CLI.
//
// The binary is located lazily on the first call to GenSync, GenStream, or
// Ping, so New succeeds even when the CLI is not installed.
//
// Supported ProviderOptions:
//   - genai.ProviderOptionModel — model ID (e.g. "claude-sonnet-4-20250514").
//     Use genai.ModelCheap, genai.ModelGood, or genai.ModelSOTA for automatic selection.
func New(opts ...genai.ProviderOption) (*Client, error) {
	c := &Client{}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionModel:
			// Pi models are set via set_model command with provider+modelId.
			// For now, store the raw string; we'll resolve shortcuts when we
			// have the model list.
			c.model = string(v)
		case genai.ProviderOptionStarterWrapper:
			c.starterWrapper = v
		default:
			return nil, fmt.Errorf("unsupported provider option %T", opt)
		}
	}
	return c, nil
}

// ensureBin locates the pi binary on first call. Safe for concurrent use.
func (c *Client) ensureBin() error {
	c.binOnce.Do(func() {
		bin, err := exec.LookPath("pi")
		if err != nil {
			if c.starterWrapper == nil {
				c.binErr = fmt.Errorf("pi CLI not found on PATH: %w", err)
				return
			}
			bin = "pi"
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
func (c *Client) Name() string { return "pi" }

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

// Ping implements genai.ProviderPing by running `pi --version`.
func (c *Client) Ping(ctx context.Context) error {
	if err := c.ensureBin(); err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, c.bin, "--version")
	out, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("pi --version: %w", err)
	}
	if len(out) == 0 {
		return errors.New("pi --version returned empty output")
	}
	return nil
}

// ListModels implements genai.Provider by querying Pi for available models.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if err := c.ensureBin(); err != nil {
		return nil, err
	}
	stdin, stdout, wait, err := c.exec(ctx, []string{"--mode", "rpc", "--no-session"})
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = stdin.Close()
		_ = wait()
	}()

	sc := newScanner(stdout)
	if err := msgutil.WriteNDJSON(stdin, cmdGetModels{Type: CmdGetModels}); err != nil {
		return nil, fmt.Errorf("write get_available_models: %w", err)
	}
	resp, err := readResponseForCommand(sc, string(CmdGetModels))
	if err != nil {
		return nil, err
	}
	var md modelsData
	if err := internal.UnmarshalJSON(resp.Data, &md); err != nil {
		return nil, fmt.Errorf("parse models data: %w", err)
	}
	out := make([]genai.Model, 0, len(md.Models))
	for i := range md.Models {
		out = append(out, &md.Models[i])
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

	stdin, stdout, wait, err := c.exec(ctx, []string{"--mode", "rpc", "--no-session"})
	if err != nil {
		return genai.Result{}, err
	}
	defer func() {
		_ = stdin.Close()
		err = errors.Join(err, wait())
	}()

	sc := newScanner(stdout)
	if err := c.setupModel(stdin, sc); err != nil {
		return genai.Result{}, err
	}
	if err := sendPrompt(stdin, &userMsg); err != nil {
		return genai.Result{}, err
	}
	return readUntilDone(sc, stdin, func(string, string) bool { return true })
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

	var (
		result   genai.Result
		finalErr error
	)
	seq := func(yield func(genai.Reply) bool) {
		stdin, stdout, wait, startErr := c.exec(ctx, []string{"--mode", "rpc", "--no-session"})
		if startErr != nil {
			finalErr = startErr
			return
		}
		defer func() {
			_ = stdin.Close()
			finalErr = errors.Join(finalErr, wait())
		}()

		sc := newScanner(stdout)
		if setupErr := c.setupModel(stdin, sc); setupErr != nil {
			finalErr = setupErr
			return
		}
		if err := sendPrompt(stdin, &userMsg); err != nil {
			finalErr = err
			return
		}
		result, finalErr = readUntilDone(sc, stdin, func(text, reasoning string) bool {
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

// setupModel sends set_model if a model is configured.
func (c *Client) setupModel(stdin io.Writer, sc *bufio.Scanner) error {
	if c.model == "" || c.model == string(genai.ModelCheap) || c.model == string(genai.ModelGood) || c.model == string(genai.ModelSOTA) {
		// Pi defaults to a good model; skip set_model for marker values.
		// TODO: resolve marker values against get_available_models.
		return nil
	}
	// The set_model command requires provider + modelId. If the model string
	// contains a "/" we split on it; otherwise we send it as modelId with
	// empty provider (Pi will try to find it).
	provider, modelID := "", c.model
	if i := strings.IndexByte(c.model, '/'); i >= 0 {
		provider = c.model[:i]
		modelID = c.model[i+1:]
	}
	if err := msgutil.WriteNDJSON(stdin, cmdSetModel{
		Type:     CmdSetModel,
		Provider: provider,
		ModelID:  modelID,
	}); err != nil {
		return fmt.Errorf("write set_model: %w", err)
	}
	_, err := readResponseForCommand(sc, string(CmdSetModel))
	return err
}

// sendPrompt sends the user message as a prompt command.
func sendPrompt(stdin io.Writer, msg *genai.Message) error {
	text, images, err := msgToPromptParts(msg)
	if err != nil {
		return err
	}
	cmd := cmdPrompt{Type: CmdPrompt, Message: text, Images: images}
	return msgutil.WriteNDJSON(stdin, cmd)
}

// msgToPromptParts extracts the text and images from a genai.Message.
//
// Text documents (text/plain, text/markdown, etc.) are inlined as text.
// Image documents are sent as base64-encoded images.
func msgToPromptParts(msg *genai.Message) (string, []imageContent, error) {
	var texts []string
	var images []imageContent
	for i := range msg.Requests {
		req := &msg.Requests[i]
		if req.Text != "" {
			texts = append(texts, req.Text)
		}
		if !req.Doc.IsZero() {
			mimeType, data, err := req.Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return "", nil, fmt.Errorf("read doc: %w", err)
			}
			if strings.HasPrefix(mimeType, "text/") {
				texts = append(texts, string(data))
			} else if strings.HasPrefix(mimeType, "image/") {
				images = append(images, imageContent{
					Type:     "image",
					Data:     base64.StdEncoding.EncodeToString(data),
					MimeType: mimeType,
				})
			} else if req.Doc.URL != "" {
				return "", nil, errors.New("URL documents not supported by pi RPC; inline the content")
			} else {
				return "", nil, fmt.Errorf("unsupported doc MIME type %q for pi", mimeType)
			}
		}
	}
	if len(texts) == 0 {
		return "", nil, errors.New("user message has no text content")
	}
	return strings.Join(texts, "\n"), images, nil
}

// readResponseForCommand reads lines until a response for the given command is found.
func readResponseForCommand(sc *bufio.Scanner, cmd string) (*response, error) {
	for sc.Scan() {
		var probe lineProbe
		if json.Unmarshal(sc.Bytes(), &probe) != nil {
			continue
		}
		if probe.Type != EventResponse {
			continue
		}
		var resp response
		if err := internal.UnmarshalJSON(sc.Bytes(), &resp); err != nil {
			continue
		}
		if resp.Command != cmd {
			continue
		}
		if !resp.Success {
			return nil, fmt.Errorf("pi %s error: %s", cmd, resp.Error)
		}
		return &resp, nil
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}
	return nil, fmt.Errorf("pi exited without %s response", cmd)
}

// readUntilDone reads events until agent_end, building a genai.Result.
// For each text or reasoning delta, onDelta is called; returning false stops early.
func readUntilDone(sc *bufio.Scanner, stdin io.Writer, onDelta func(text, reasoning string) bool) (genai.Result, error) {
	var textBuf, thinkBuf strings.Builder
	for sc.Scan() {
		line := sc.Bytes()
		var probe lineProbe
		if json.Unmarshal(line, &probe) != nil {
			continue
		}

		switch probe.Type {
		case EventMessageUpdate:
			text, reasoning, err := parseMessageUpdateDelta(line)
			if err != nil {
				return genai.Result{}, err
			}
			textBuf.WriteString(text)
			thinkBuf.WriteString(reasoning)
			if (text != "" || reasoning != "") && !onDelta(text, reasoning) {
				return genai.Result{}, nil
			}

		case EventAgentEnd:
			return buildResult(line, textBuf.String(), thinkBuf.String())

		case EventExtensionUI:
			if err := handleExtensionUI(stdin, line); err != nil {
				return genai.Result{}, fmt.Errorf("handle extension UI: %w", err)
			}

		case EventResponse:
			// Responses to commands (e.g. prompt ack); skip.
			var resp response
			if internal.UnmarshalJSON(line, &resp) == nil && !resp.Success {
				return genai.Result{}, fmt.Errorf("pi error (command=%s): %s", resp.Command, resp.Error)
			}
		}
	}
	if err := sc.Err(); err != nil {
		return genai.Result{}, fmt.Errorf("read stdout: %w", err)
	}
	return genai.Result{}, errors.New("pi exited without agent_end")
}

// parseMessageUpdateDelta extracts text and reasoning deltas from a message_update event.
func parseMessageUpdateDelta(line []byte) (text, reasoning string, err error) {
	var ev eventMessageUpdate
	if err := json.Unmarshal(line, &ev); err != nil {
		return "", "", fmt.Errorf("unmarshal message_update: %w", err)
	}
	switch ev.AssistantMessageEvent.Type {
	case DeltaTextDelta:
		return ev.AssistantMessageEvent.Delta, "", nil
	case DeltaThinkDelta:
		return "", ev.AssistantMessageEvent.Delta, nil
	}
	return "", "", nil
}

// handleExtensionUI auto-responds to extension UI requests.
// For select: picks the first option. For confirm: confirms true.
// For all others: sends an empty value response.
func handleExtensionUI(stdin io.Writer, line []byte) error {
	var req extensionUIRequest
	if err := json.Unmarshal(line, &req); err != nil {
		return fmt.Errorf("unmarshal extension_ui_request: %w", err)
	}
	switch req.Method {
	case UIMethodConfirm:
		return msgutil.WriteNDJSON(stdin, extensionUIResponseConfirm{
			Type:      "extension_ui_response",
			ID:        req.ID,
			Confirmed: true,
		})
	case UIMethodSelect:
		val := ""
		if len(req.Options) > 0 {
			val = req.Options[0]
		}
		return msgutil.WriteNDJSON(stdin, extensionUIResponseValue{
			Type:  "extension_ui_response",
			ID:    req.ID,
			Value: val,
		})
	case UIMethodNotify, UIMethodSetStatus, UIMethodSetWidget, UIMethodSetTitle:
		// Fire-and-forget methods don't need a response, but Pi still waits
		// for one on dialog methods. These are safe to ignore.
		return nil
	default:
		// input, editor, or unknown: send empty value.
		return msgutil.WriteNDJSON(stdin, extensionUIResponseValue{
			Type:  "extension_ui_response",
			ID:    req.ID,
			Value: "",
		})
	}
}

// buildResult constructs a genai.Result from an agent_end event, using
// accumulated text/thinking buffers and the final assistant message's usage.
func buildResult(line []byte, text, thinking string) (genai.Result, error) {
	r := genai.Result{}
	var ev eventAgentEnd
	if err := json.Unmarshal(line, &ev); err != nil {
		return r, fmt.Errorf("unmarshal agent_end: %w", err)
	}

	// Find the last assistant message for usage and stop reason.
	for i := len(ev.Messages) - 1; i >= 0; i-- {
		msg := &ev.Messages[i]
		if msg.Role != "assistant" {
			continue
		}
		r.Usage.InputTokens = msg.Usage.Input
		r.Usage.OutputTokens = msg.Usage.Output
		r.Usage.InputCachedTokens = msg.Usage.CacheRead
		r.Usage.TotalTokens = msg.Usage.TotalTokens
		r.Usage.FinishReason = stopReasonToFinishReason(msg.StopReason)

		// If we didn't accumulate streaming text, extract from the message content.
		if text == "" && thinking == "" {
			t, th := extractContent(msg)
			text = t
			thinking = th
		}
		break
	}

	if thinking != "" {
		r.Replies = append(r.Replies, genai.Reply{Reasoning: thinking})
	}
	if text != "" {
		r.Replies = append(r.Replies, genai.Reply{Text: text})
	}
	return r, nil
}

// extractContent extracts text and thinking from an assistant message's content blocks.
func extractContent(msg *agentMessage) (text, thinking string) {
	if msg.Content == nil {
		return "", ""
	}
	var blocks []contentBlock
	if json.Unmarshal(msg.Content, &blocks) != nil {
		return "", ""
	}
	var textParts, thinkParts []string
	for i := range blocks {
		switch blocks[i].Type {
		case "text":
			if blocks[i].Text != "" {
				textParts = append(textParts, blocks[i].Text)
			}
		case "thinking":
			if blocks[i].Thinking != "" {
				thinkParts = append(thinkParts, blocks[i].Thinking)
			}
		}
	}
	return strings.Join(textParts, ""), strings.Join(thinkParts, "")
}

// stopReasonToFinishReason maps a Pi StopReason to a genai.FinishReason.
func stopReasonToFinishReason(reason StopReason) genai.FinishReason {
	switch reason {
	case StopReasonLength:
		return genai.FinishedLength
	case StopReasonToolUse:
		return genai.FinishedToolCalls
	case StopReasonError, StopReasonAborted:
		return genai.FinishedContentFilter
	default:
		return genai.FinishedStop
	}
}

// yieldNothing is an empty iterator used for early error returns in GenStream.
func yieldNothing(func(genai.Reply) bool) {}

// errFinish returns a finish function that always returns the given error.
func errFinish(err error) func() (genai.Result, error) {
	return func() (genai.Result, error) { return genai.Result{}, err }
}

// Compile-time interface checks.
var (
	_ genai.Provider     = (*Client)(nil)
	_ genai.ProviderPing = (*Client)(nil)
	_ genai.Model        = (*Model)(nil)
)
