// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package antigravity implements a genai provider backed by the Google
// Antigravity CLI (`agy`).
//
// Instead of making HTTP requests directly, it launches `agy` in print mode as
// a subprocess and communicates over stdin/stdout with the NDJSON stream-json
// protocol (--input-format stream-json --output-format stream-json).
//
// Install the CLI with:
//
//	curl -fsSL https://antigravity.google/cli/install.sh | bash
//
// then run `agy` once interactively to sign in.
//
// # Side effects
//
// agy always exposes its built-in agent tools (file edits, shell commands, web
// search, browser) and has no flag to disable them. In print mode it
// auto-approves workspace writes and soft-denies requests that need review;
// GenOption.DangerouslySkipPermissions approves them all. The provider runs
// each subprocess in a fresh empty temporary directory, removed after the
// call, to contain workspace writes.
//
// # Session / multi-turn
//
// Each GenSync or GenStream call launches a fresh subprocess. agy persists
// every conversation under ~/.gemini/antigravity-cli and has no flag to
// disable it. The conversation ID is returned in
// Reply.Opaque["conversation_id"]. When the message history contains one, the
// provider passes --conversation <id> and sends only the last user message.
package antigravity

import (
	"bufio"
	"bytes"
	"cmp"
	"context"
	_ "embed"
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
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/msgutil"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for the Antigravity CLI provider.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// Effort levels for GenOption.Effort, as accepted by `agy --effort`.
const (
	EffortLow    = "low"
	EffortMedium = "medium"
	EffortHigh   = "high"
	EffortMax    = "max"
)

// Mode is an agent execution mode, as accepted by `agy --mode`.
type Mode string

// Agent execution modes.
const (
	ModeAcceptEdits Mode = "accept-edits"
	ModePlan        Mode = "plan"
)

// GenOption configures an Antigravity CLI call.
//
// All fields are opt-in. By default agy runs with its default mode, review
// permissions, no terminal sandbox, and slash commands disabled.
type GenOption struct {
	// Effort sets the reasoning effort (--effort). Use the Effort* constants.
	// Model IDs already encode a default effort, e.g. "gemini-3.8-flash-low".
	Effort string
	// Mode sets the agent execution mode (--mode).
	Mode Mode
	// DangerouslySkipPermissions auto-approves every tool permission request
	// (--dangerously-skip-permissions), including shell commands and file
	// access outside the temporary workspace.
	DangerouslySkipPermissions bool
	// Sandbox runs terminal commands with restrictions (--sandbox).
	Sandbox bool
	// SlashCommands expands slash commands and skills in the prompt. By
	// default the provider passes --disable-slash-commands so that a prompt
	// starting with "/" reaches the model verbatim.
	SlashCommands bool
	// ExtraArgs are appended to the agy command line after the provider's
	// flags. A repeated flag overrides the earlier value. Flags that select the
	// print mode or the input and output formats are rejected because the
	// provider owns the protocol.
	ExtraArgs []string

	_ struct{}
}

// Validate implements genai.GenOption.
func (g *GenOption) Validate() error {
	switch g.Effort {
	case "", EffortLow, EffortMedium, EffortHigh, EffortMax:
	default:
		return fmt.Errorf("GenOption.Effort: invalid level %q; must be one of low, medium, high, max", g.Effort)
	}
	switch g.Mode {
	case "", ModeAcceptEdits, ModePlan:
	default:
		return fmt.Errorf("GenOption.Mode: invalid mode %q; must be one of accept-edits, plan", g.Mode)
	}
	for _, a := range g.ExtraArgs {
		name, _, _ := strings.Cut(strings.TrimPrefix(strings.TrimPrefix(a, "-"), "-"), "=")
		switch name {
		case "i", "input-format", "output-format", "p", "print", "prompt", "prompt-interactive":
			return fmt.Errorf("GenOption.ExtraArgs: %q conflicts with the stream-json protocol", a)
		default:
		}
	}
	return nil
}

// New creates a Client for the `agy` CLI.
//
// The binary is located lazily, so New succeeds without the CLI unless a model
// marker requires listing the models.
//
// Supported ProviderOptions:
//   - genai.ProviderOptionModel — model ID as listed by `agy models`, e.g.
//     "gemini-3.8-flash-low". genai.ModelCheap, genai.ModelGood, and
//     genai.ModelSOTA are resolved against the live model list by running
//     `agy models`.
//   - genai.ProviderOptionStarterWrapper — intercepts subprocess creation.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	if err := base.CheckDuplicateProviderOptions(opts); err != nil {
		return nil, err
	}
	c := &Client{}
	var model genai.ProviderOptionModel
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionModel:
			model = v
		case genai.ProviderOptionStarterWrapper:
			c.starterWrapper = v
		default:
			return nil, fmt.Errorf("unsupported provider option %T", opt)
		}
	}
	switch model {
	case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA:
		models, err := c.listModels(ctx)
		if err != nil {
			return nil, err
		}
		if c.model, err = selectModel(models, model); err != nil {
			return nil, err
		}
	default:
		c.model = string(model)
	}
	return c, nil
}

// Client is a genai provider that delegates to the local `agy` CLI.
type Client struct {
	base.NotImplemented
	starterWrapper genai.ProviderOptionStarterWrapper
	model          string

	binOnce sync.Once
	binErr  error
	bin     string
	exec    genai.Starter
}

// Name implements genai.Provider.
func (c *Client) Name() string { return "antigravity" }

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

// Ping implements genai.ProviderPing by running `agy --version`.
func (c *Client) Ping(ctx context.Context) error {
	if err := c.ensureBin(); err != nil {
		return err
	}
	out, err := exec.CommandContext(ctx, c.bin, "--version").Output()
	if err != nil {
		return fmt.Errorf("agy --version: %w", err)
	}
	if len(bytes.TrimSpace(out)) == 0 {
		return errors.New("agy --version returned empty output")
	}
	return nil
}

// ListModels implements genai.Provider by running `agy models`.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	models, err := c.listModels(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]genai.Model, len(models))
	for i := range models {
		out[i] = &models[i]
	}
	return out, nil
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	co, optsErr := parseOpts(opts)
	if optsErr != nil {
		if _, ok := errors.AsType[*base.ErrNotSupported](optsErr); !ok {
			return genai.Result{}, optsErr
		}
	}
	res, err := c.run(ctx, msgs, &co, nil)
	if err == nil {
		err = optsErr
	}
	return res, err
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	co, optsErr := parseOpts(opts)
	if optsErr != nil {
		if _, ok := errors.AsType[*base.ErrNotSupported](optsErr); !ok {
			return func(func(genai.Reply) bool) {}, func() (genai.Result, error) { return genai.Result{}, optsErr }
		}
	}
	var (
		result   genai.Result
		finalErr error
	)
	seq := func(yield func(genai.Reply) bool) {
		result, finalErr = c.run(ctx, msgs, &co, func(text string) bool {
			return yield(genai.Reply{Text: text})
		})
	}
	return seq, func() (genai.Result, error) {
		if finalErr == nil {
			finalErr = optsErr
		}
		return result, finalErr
	}
}

// conversationIDKey is the Reply.Opaque key carrying the agy conversation ID.
const conversationIDKey = "conversation_id"

// callOpts holds per-call options parsed from the GenOption slice.
type callOpts struct {
	gen        GenOption
	jsonSchema genai.JSONSchema
}

// parseOpts validates and collects per-call options.
//
// It returns a *base.ErrNotSupported alongside valid callOpts when an option
// is ignored, so the call can proceed.
func parseOpts(opts []genai.GenOption) (callOpts, error) {
	if err := base.CheckDuplicateGenOptions(opts); err != nil {
		return callOpts{}, err
	}
	var co callOpts
	var unsupported []string
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return callOpts{}, err
		}
		switch v := opt.(type) {
		case *GenOption:
			co.gen = *v
		case *genai.GenOptionText:
			if v.DecodeAs != nil {
				s, err := v.DecodeSchema()
				if err != nil {
					return callOpts{}, fmt.Errorf("GenOptionText.DecodeAs: %w", err)
				}
				co.jsonSchema = s
			}
			for _, f := range []struct {
				name  string
				isSet bool
			}{
				{"GenOptionText.MaxTokens", v.MaxTokens != 0},
				{"GenOptionText.ReplyAsJSON", v.ReplyAsJSON},
				{"GenOptionText.Stop", len(v.Stop) != 0},
				{"GenOptionText.SystemPrompt", v.SystemPrompt != ""},
				{"GenOptionText.Temperature", v.Temperature != 0},
				{"GenOptionText.TopK", v.TopK != 0},
				{"GenOptionText.TopLogprobs", v.TopLogprobs != 0},
				{"GenOptionText.TopP", v.TopP != 0},
			} {
				if f.isSet {
					unsupported = append(unsupported, f.name)
				}
			}
		case genai.GenOptionSeed:
			unsupported = append(unsupported, "GenOptionSeed")
		case *genai.GenOptionTools:
			unsupported = append(unsupported, "GenOptionTools")
		case *genai.GenOptionWeb:
			unsupported = append(unsupported, "GenOptionWeb")
		default:
			return callOpts{}, fmt.Errorf("unsupported option %T", opt)
		}
	}
	if len(unsupported) != 0 {
		return co, &base.ErrNotSupported{Options: unsupported}
	}
	return co, nil
}

// ensureBin locates the agy binary on first call. It is safe for concurrent
// use.
func (c *Client) ensureBin() error {
	c.binOnce.Do(func() {
		bin, err := exec.LookPath("agy")
		if err != nil {
			if c.starterWrapper == nil {
				c.binErr = fmt.Errorf("agy CLI not found on PATH: %w", err)
				return
			}
			// A wrapper (e.g. test replay) may never call the inner starter.
			bin = "agy"
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

func (c *Client) listModels(ctx context.Context) ([]Model, error) {
	if err := c.ensureBin(); err != nil {
		return nil, err
	}
	// The models subcommand rejects its own --output-format; the global flag
	// works.
	stdin, stdout, wait, err := c.exec(ctx, []string{"--output-format", "stream-json", "models"})
	if err != nil {
		return nil, err
	}
	// agy does not read stdin for this command.
	if err := stdin.Close(); err != nil {
		return nil, errors.Join(fmt.Errorf("close stdin: %w", err), wait())
	}
	models, err := readModels(newScanner(stdout))
	return models, errors.Join(err, wait())
}

// readModels reads the `models` command_result until the terminating result.
func readModels(sc *bufio.Scanner) ([]Model, error) {
	var models []Model
	for sc.Scan() {
		var ev StreamEvent
		if err := internal.UnmarshalJSON(sc.Bytes(), &ev); err != nil {
			return nil, fmt.Errorf("parse agy output: %w", err)
		}
		switch ev.Event {
		case EventCommandResult:
			if ev.Command.Name != "models" {
				return nil, fmt.Errorf("unexpected command_result %q, want models", ev.Command.Name)
			}
			var md ModelsData
			if err := internal.UnmarshalJSON(ev.Command.Data, &md); err != nil {
				return nil, fmt.Errorf("parse models: %w", err)
			}
			models = md.Models
		case EventResult:
			if err := ev.Result.AsError(); err != nil {
				return nil, err
			}
			if models == nil {
				return nil, errors.New("agy returned no models command_result")
			}
			return models, nil
		default:
			return nil, fmt.Errorf("unexpected agy event %q", ev.Event)
		}
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read stdout: %w", err)
	}
	return nil, errors.New("agy exited without a result event")
}

// selectModel picks the newest model of the family and effort matching the
// marker. Model IDs follow "gemini-<major>.<minor>-<family>-<effort>".
//
//   - Cheap: newest flash, low effort.
//   - Good: newest flash, medium effort.
//   - SOTA: newest pro, high effort.
func selectModel(models []Model, marker genai.ProviderOptionModel) (string, error) {
	family, effort := "flash", "low"
	switch marker {
	case genai.ModelGood:
		effort = "medium"
	case genai.ModelSOTA:
		family, effort = "pro", "high"
	default:
	}
	var best string
	var bestVer [2]int
	for i := range models {
		ver, ok := parseGeminiID(models[i].ID, family, effort)
		if ok && (best == "" || cmp.Or(cmp.Compare(ver[0], bestVer[0]), cmp.Compare(ver[1], bestVer[1])) > 0) {
			best, bestVer = models[i].ID, ver
		}
	}
	if best == "" {
		return "", fmt.Errorf("no gemini %s model with %s effort for %s", family, effort, marker)
	}
	return best, nil
}

// parseGeminiID returns the version of a "gemini-<major>.<minor>-<family>-<effort>" ID.
func parseGeminiID(id, family, effort string) ([2]int, bool) {
	rest, ok := strings.CutPrefix(id, "gemini-")
	if !ok {
		return [2]int{}, false
	}
	rest, ok = strings.CutSuffix(rest, "-"+family+"-"+effort)
	if !ok {
		return [2]int{}, false
	}
	major, minor, ok := strings.Cut(rest, ".")
	if !ok {
		return [2]int{}, false
	}
	a, err1 := strconv.Atoi(major)
	b, err2 := strconv.Atoi(minor)
	if err1 != nil || err2 != nil {
		return [2]int{}, false
	}
	return [2]int{a, b}, true
}

// run executes one turn. When onText is non-nil, it receives text deltas and
// returning false stops the call.
func (c *Client) run(ctx context.Context, msgs genai.Messages, co *callOpts, onText func(string) bool) (res genai.Result, err error) {
	if err := c.ensureBin(); err != nil {
		return res, err
	}
	userMsg, err := msgutil.LastUserMsg(msgs)
	if err != nil {
		return res, err
	}
	in, err := userMsgToInput(&userMsg)
	if err != nil {
		return res, err
	}
	args := c.buildArgs(co, msgutil.ExtractOpaqueID(msgs, conversationIDKey))

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stdin, stdout, wait, err := c.exec(ctx, args)
	if err != nil {
		return res, err
	}
	stopped := false
	defer func() {
		if werr := wait(); werr != nil && !stopped {
			err = errors.Join(err, werr)
		}
	}()
	// agy runs one turn per stdin line and exits after EOF once the turn ends.
	if err := errors.Join(msgutil.WriteNDJSON(stdin, &in), stdin.Close()); err != nil {
		return res, fmt.Errorf("write user message: %w", err)
	}
	// Structured output replaces the streamed text, which carries extra
	// fields; stream it once at the end so the stream matches the result.
	stream := onText != nil && co.jsonSchema == nil
	res, err = readTurn(newScanner(stdout), func(text string) bool {
		if stream && !onText(text) {
			// The consumer abandoned the stream: kill the subprocess.
			stopped = true
			cancel()
		}
		return !stopped
	})
	if err != nil || stopped || onText == nil || stream {
		return res, err
	}
	for i := range res.Replies {
		if t := res.Replies[i].Text; t != "" {
			onText(t)
		}
	}
	return res, nil
}

// buildArgs constructs the agy argument list for a call.
func (c *Client) buildArgs(co *callOpts, conversationID string) []string {
	args := []string{
		// Empty prompt: the prompt arrives on stdin. A bare -p would consume the
		// next flag as its prompt.
		"-p=",
		"--input-format", "stream-json",
		"--output-format", "stream-json",
	}
	if !co.gen.SlashCommands {
		args = append(args, "--disable-slash-commands")
	}
	if c.model != "" {
		args = append(args, "--model", c.model)
	}
	if co.gen.Effort != "" {
		args = append(args, "--effort", co.gen.Effort)
	}
	if co.gen.Mode != "" {
		args = append(args, "--mode", string(co.gen.Mode))
	}
	if co.gen.DangerouslySkipPermissions {
		args = append(args, "--dangerously-skip-permissions")
	}
	if co.gen.Sandbox {
		args = append(args, "--sandbox")
	}
	if co.jsonSchema != nil {
		args = append(args, "--json-schema", string(co.jsonSchema))
	}
	if conversationID != "" {
		args = append(args, "--conversation", conversationID)
	}
	return append(args, co.gen.ExtraArgs...)
}

// userMsgToInput converts a genai user message into an agy stdin message.
//
// agy accepts only text blocks; text documents are inlined.
func userMsgToInput(msg *genai.Message) (StreamInputMessage, error) {
	blocks := make([]StreamInputContentBlock, 0, len(msg.Requests))
	for i := range msg.Requests {
		req := &msg.Requests[i]
		if req.Text != "" {
			blocks = append(blocks, StreamInputContentBlock{Type: "text", Text: req.Text})
		}
		if req.Doc.IsZero() {
			continue
		}
		if req.Doc.URL != "" {
			return StreamInputMessage{}, &base.ErrNotSupported{Options: []string{"document URL"}}
		}
		mimeType, data, err := req.Doc.Read(10 * 1024 * 1024)
		if err != nil {
			return StreamInputMessage{}, fmt.Errorf("read doc: %w", err)
		}
		if !strings.HasPrefix(mimeType, "text/") {
			return StreamInputMessage{}, &base.ErrNotSupported{Options: []string{mimeType + " document"}}
		}
		blocks = append(blocks, StreamInputContentBlock{Type: "text", Text: string(data)})
	}
	if len(blocks) == 0 {
		return StreamInputMessage{}, errors.New("user message has no content")
	}
	return StreamInputMessage{Event: EventUser, Message: StreamInputUserMessage{Content: blocks}}, nil
}

// readTurn reads stdout events until the turn's result. onText receives each
// agent_response text delta; returning false stops reading.
//
// Usage sums the per-step usage because JSONOutput.Usage is cumulative over
// the whole conversation.
func readTurn(sc *bufio.Scanner, onText func(string) bool) (genai.Result, error) {
	var text strings.Builder
	var usage JSONUsage
	for sc.Scan() {
		var ev StreamEvent
		if err := internal.UnmarshalJSON(sc.Bytes(), &ev); err != nil {
			return genai.Result{}, fmt.Errorf("parse agy output: %w", err)
		}
		switch ev.Event {
		case EventInit:
		case EventStepUpdate:
			s := &ev.StepUpdate
			usage.Add(&s.Usage)
			if s.StepType == StepAgentResponse && s.TextDelta != "" {
				text.WriteString(s.TextDelta)
				if !onText(s.TextDelta) {
					return genai.Result{}, nil
				}
			}
		case EventResult:
			if err := ev.Result.AsError(); err != nil {
				return genai.Result{}, err
			}
			return buildResult(&ev.Result, text.String(), &usage), nil
		default:
			return genai.Result{}, fmt.Errorf("unexpected agy event %q", ev.Event)
		}
	}
	if err := sc.Err(); err != nil {
		return genai.Result{}, fmt.Errorf("read stdout: %w", err)
	}
	return genai.Result{}, errors.New("agy exited without a result event")
}

func buildResult(r *JSONOutput, text string, u *JSONUsage) genai.Result {
	res := genai.Result{
		Usage: genai.Usage{
			InputTokens:       u.InputTokens,
			InputCachedTokens: u.CacheReadTokens,
			ReasoningTokens:   u.ThinkingTokens,
			OutputTokens:      u.OutputTokens,
			TotalTokens:       u.TotalTokens,
			FinishReason:      genai.FinishedStop,
		},
	}
	if len(r.StructuredOutput) != 0 {
		text = string(r.StructuredOutput)
	}
	if text != "" {
		res.Replies = append(res.Replies, genai.Reply{Text: text})
	}
	if r.ConversationID != "" {
		res.Replies = append(res.Replies, genai.Reply{Opaque: map[string]any{conversationIDKey: r.ConversationID}})
	}
	return res
}

// cmdExecutor is the production executor backed by exec.Cmd.
type cmdExecutor struct{ bin string }

func (e *cmdExecutor) start(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
	// Fresh empty workspace: agy sees no project files and its writes are
	// discarded with the directory.
	dir, err := os.MkdirTemp("", "genai-antigravity-*")
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
		return nil, nil, nil, errors.Join(fmt.Errorf("start agy: %w", err), os.RemoveAll(dir))
	}
	return stdin, stdout, func() error {
		return errors.Join(cmd.Wait(), os.RemoveAll(dir))
	}, nil
}

// newScanner returns a scanner sized for large NDJSON lines (up to 32 MB).
func newScanner(r io.Reader) *bufio.Scanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 32<<20)
	return sc
}

// Compile-time interface checks.
var (
	_ genai.Provider     = (*Client)(nil)
	_ genai.ProviderPing = (*Client)(nil)
	_ genai.Model        = (*Model)(nil)
)
