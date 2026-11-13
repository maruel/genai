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
//
// # System prompt
//
// ACP has no system prompt field. For GenOptionText.SystemPrompt, the provider
// writes an OpenCode agent file to .opencode/agent/<mode>.md in the temporary
// working directory and selects that mode. Consequences:
//
//   - Without GenOption.Mode, the call runs the agent "genai" instead of
//     OpenCode's default agent. It has OpenCode's default permissions and
//     tools, and any "genai" agent from the user's configuration is merged
//     into it.
//   - With GenOption.Mode, the file overrides only that agent's prompt and
//     sets it to a primary agent; its permissions and tools stay. A "/" in the
//     mode names a nested agent file. A mode that leaves the working
//     directory, such as one with "..", fails the call.
//   - The prompt replaces OpenCode's provider prompt. OpenCode still appends
//     its skills, environment, and instruction files.
//   - OpenCode trims leading and trailing white space from the prompt.
//   - OpenCode applies the agent prompt on every request, so a resumed
//     session uses the system prompt of the latest call.
//   - The agent file is visible to the agent's tools in its working
//     directory.
package opencode

import (
	"bytes"
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
	"path/filepath"
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/acp"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/cli"
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

// ProviderOption configures the opencode process. It implements
// genai.ProviderOption.
//
// All fields are opt-in. By default the provider rejects every permission
// request the agent makes.
type ProviderOption struct {
	// DangerouslySkipPermissions approves every permission request the agent
	// makes, such as file edits and shell commands.
	DangerouslySkipPermissions bool
	// ExtraArgs are appended to the `opencode acp` command line.
	ExtraArgs []string

	_ struct{}
}

// Validate implements genai.ProviderOption.
func (p *ProviderOption) Validate() error {
	return msgutil.CheckExtraArgs(p.ExtraArgs)
}

// GenOption configures one prompt of a running opencode session.
type GenOption struct {
	// Mode selects an OpenCode session mode returned by ACP. An empty value
	// leaves OpenCode's current mode intact, unless GenOptionText.SystemPrompt
	// is set: then the call selects the agent "genai". With a system prompt,
	// Mode is also a file name; see the package documentation.
	Mode Mode

	_ struct{}
}

// Validate implements genai.GenOption.
func (g *GenOption) Validate() error {
	if g.Mode != "" && strings.TrimSpace(string(g.Mode)) == "" {
		return errors.New("GenOption.Mode: mode must not be whitespace")
	}
	return nil
}

// Client is a genai provider that delegates to the local `opencode` CLI.
type Client struct {
	base.NotImplemented

	starterWrapper genai.ProviderOptionStarterWrapper
	bin            string
	model          string
	opts           ProviderOption
	binOnce        sync.Once
	binErr         error
}

// New creates a Client for the `opencode` CLI.
//
// The binary is located lazily on the first call to GenSync, GenStream, or
// Ping, so New succeeds even when the CLI is not installed, unless a model
// marker requires listing models.
//
// Supported ProviderOptions:
//   - genai.ProviderOptionModel — model ID (e.g. "opencode/big-pickle").
//     genai.ModelCheap, genai.ModelGood, and genai.ModelSOTA select the newest
//     "openai/" luna, terra, or astra model. They use the ChatGPT
//     subscription when OpenCode is logged in with it.
//   - *ProviderOption — process settings.
//   - genai.ProviderOptionStarterWrapper — intercepts subprocess creation.
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	c := &Client{}
	var marker genai.ProviderOptionModel
	if err := base.CheckDuplicateProviderOptions(opts); err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionModel:
			switch v {
			case genai.ModelCheap, genai.ModelGood, genai.ModelSOTA:
				marker = v
			default:
				c.model = string(v)
			}
		case *ProviderOption:
			c.opts = *v
		case genai.ProviderOptionStarterWrapper:
			c.starterWrapper = v
		default:
			return nil, fmt.Errorf("unsupported provider option %T", opt)
		}
	}
	if marker != "" {
		models, err := c.ListModels(ctx)
		if err != nil {
			return nil, err
		}
		if c.model, err = msgutil.SelectGPTTier(models, "openai/", marker); err != nil {
			return nil, err
		}
	}
	return c, nil
}

// Close implements io.Closer. It currently does nothing.
func (c *Client) Close() error {
	return nil
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
	stdin, stdout, wait, cwd, err := c.start(ctx, []string{"acp"}, &callOpts{})
	if err != nil {
		return nil, err
	}
	defer func() {
		_ = stdin.Close()
		_ = wait()
	}()

	s, err := acp.Handshake(stdin, stdout, acpOptions(cwd), acp.SessionRequest{})
	if err != nil {
		return nil, wrapACP(err)
	}
	choices := s.ModelChoices()
	out := make([]genai.Model, 0, len(choices))
	for _, m := range choices {
		name := m.Name
		if name == "" {
			name = m.Value
		}
		out = append(out, &model{id: m.Value, displayName: name})
	}
	return out, nil
}

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (r genai.Result, err error) {
	if err := c.ensureBin(); err != nil {
		return genai.Result{}, err
	}
	co, optsErr := parseOpts(opts)
	if optsErr != nil {
		if _, ok := errors.AsType[*base.ErrNotSupported](optsErr); !ok {
			return genai.Result{}, optsErr
		}
		defer func() {
			if err == nil {
				err = optsErr
			}
		}()
	}
	userMsg, err := msgutil.LastUserMsg(msgs)
	if err != nil {
		return genai.Result{}, err
	}
	resumeSessionID := msgutil.ExtractOpaqueID(msgs, acp.SessionIDKey)

	stdin, stdout, wait, cwd, err := c.start(ctx, append([]string{"acp"}, c.opts.ExtraArgs...), &co)
	if err != nil {
		return genai.Result{}, err
	}
	defer func() {
		_ = stdin.Close()
		err = errors.Join(err, wait())
	}()

	s, err := acp.Handshake(stdin, stdout, acpOptions(cwd), acp.SessionRequest{
		Model:           c.model,
		Effort:          string(co.effort),
		Mode:            string(co.sessionMode()),
		ResumeSessionID: resumeSessionID,
	})
	if err != nil {
		return genai.Result{}, wrapACP(err)
	}

	promptID, err := s.SendPrompt(&userMsg)
	if err != nil {
		return genai.Result{}, wrapACP(err)
	}

	r, err = s.ReadTurn(promptID, c.opts.DangerouslySkipPermissions, func(string, string) bool { return true })
	return r, wrapACP(err)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if err := c.ensureBin(); err != nil {
		return yieldNothing, errFinish(err)
	}
	co, optsErr := parseOpts(opts)
	if optsErr != nil {
		if _, ok := errors.AsType[*base.ErrNotSupported](optsErr); !ok {
			return yieldNothing, errFinish(optsErr)
		}
	}
	userMsg, err := msgutil.LastUserMsg(msgs)
	if err != nil {
		return yieldNothing, errFinish(err)
	}
	resumeSessionID := msgutil.ExtractOpaqueID(msgs, acp.SessionIDKey)

	var (
		result   genai.Result
		finalErr error
	)
	seq := func(yield func(genai.Reply) bool) {
		stdin, stdout, wait, cwd, startErr := c.start(ctx, append([]string{"acp"}, c.opts.ExtraArgs...), &co)
		if startErr != nil {
			finalErr = startErr
			return
		}
		defer func() {
			_ = stdin.Close()
			finalErr = errors.Join(finalErr, wait())
		}()

		s, hsErr := acp.Handshake(stdin, stdout, acpOptions(cwd), acp.SessionRequest{
			Model:           c.model,
			Effort:          string(co.effort),
			Mode:            string(co.sessionMode()),
			ResumeSessionID: resumeSessionID,
		})
		if hsErr != nil {
			finalErr = wrapACP(hsErr)
			return
		}
		promptID, err := s.SendPrompt(&userMsg)
		if err != nil {
			finalErr = wrapACP(err)
			return
		}

		var readErr error
		result, readErr = s.ReadTurn(promptID, c.opts.DangerouslySkipPermissions, func(text, reasoning string) bool {
			if text != "" && !yield(genai.Reply{Text: text}) {
				return false
			}
			if reasoning != "" && !yield(genai.Reply{Reasoning: reasoning}) {
				return false
			}
			return true
		})
		finalErr = wrapACP(readErr)
	}
	return seq, func() (genai.Result, error) {
		if finalErr == nil {
			finalErr = optsErr
		}
		return result, finalErr
	}
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
	})
	return c.binErr
}

// start launches the ACP subprocess in a fresh temporary directory and returns
// the ACP session working directory. It writes the agent carrying
// co.systemPrompt there.
func (c *Client) start(ctx context.Context, args []string, co *callOpts) (io.WriteCloser, io.ReadCloser, func() error, string, error) {
	dir, err := os.MkdirTemp("", "genai-opencode-*")
	if err != nil {
		return nil, nil, nil, "", fmt.Errorf("create temp dir: %w", err)
	}
	if co.systemPrompt != "" {
		if err := writeAgent(dir, co.sessionMode(), co.systemPrompt); err != nil {
			return nil, nil, nil, "", errors.Join(err, os.RemoveAll(dir))
		}
	}
	s := cli.NewStarter(c.bin, dir)
	if c.starterWrapper != nil {
		s = c.starterWrapper(s)
	}
	stdin, stdout, wait, err := s(ctx, args)
	if err != nil {
		return nil, nil, nil, "", errors.Join(err, os.RemoveAll(dir))
	}
	return stdin, stdout, func() error {
		return errors.Join(wait(), os.RemoveAll(dir))
	}, dir, nil
}

// agentName names the agent that carries GenOptionText.SystemPrompt when no
// GenOption.Mode is set.
const agentName Mode = "genai"

// writeAgent writes the primary agent mode with systemPrompt as its prompt
// into the working directory dir. The file overrides only the prompt and mode
// of an existing agent with that name.
func writeAgent(dir string, mode Mode, systemPrompt string) (err error) {
	// mode comes from the caller and may contain path separators; the root
	// confines the write to dir.
	root, err := os.OpenRoot(dir)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, root.Close()) }()
	name := filepath.Join(".opencode", "agent", filepath.FromSlash(string(mode))+".md")
	if err := root.MkdirAll(filepath.Dir(name), 0o700); err != nil {
		return fmt.Errorf("create agent dir: %w", err)
	}
	// The front matter keeps a prompt that starts with "---" from being parsed
	// as front matter.
	b := []byte("---\nmode: primary\n---\n\n" + systemPrompt + "\n")
	if err := root.WriteFile(name, b, 0o600); err != nil {
		return fmt.Errorf("write agent: %w", err)
	}
	return nil
}

// callOpts holds the settings of one call.
type callOpts struct {
	effort       Effort
	mode         Mode
	systemPrompt string
}

// parseOpts collects per-call generation settings.
func parseOpts(opts []genai.GenOption) (callOpts, error) {
	co := callOpts{}
	if err := base.CheckDuplicateGenOptions(opts); err != nil {
		return co, err
	}
	var unsupported []string
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return co, err
		}
		switch v := opt.(type) {
		case *GenOption:
			co.mode = v.Mode
		case genai.EffortLevel:
			switch v {
			case "":
			case genai.EffortOff:
				co.effort = EffortNone
			case genai.EffortMinimal:
				co.effort = EffortMinimal
			case genai.EffortLow:
				co.effort = EffortLow
			case genai.EffortMedium:
				co.effort = EffortMedium
			case genai.EffortHigh:
				co.effort = EffortHigh
			case genai.EffortXHigh:
				co.effort = EffortXHigh
			case genai.EffortMax:
				co.effort = EffortMax
			default:
				return callOpts{}, &base.ErrNotSupported{Options: []string{"EffortLevel"}}
			}
		case genai.ReasoningOutput:
		case *genai.GenOptionText:
			co.systemPrompt = v.SystemPrompt
			if v.Temperature != 0 {
				unsupported = append(unsupported, "GenOptionText.Temperature")
			}
			if v.TopP != 0 {
				unsupported = append(unsupported, "GenOptionText.TopP")
			}
			if v.MaxTokens != 0 {
				unsupported = append(unsupported, "GenOptionText.MaxTokens")
			}
			if v.TopLogprobs != 0 {
				unsupported = append(unsupported, "GenOptionText.TopLogprobs")
			}
			if v.TopK != 0 {
				unsupported = append(unsupported, "GenOptionText.TopK")
			}
			if len(v.Stop) != 0 {
				unsupported = append(unsupported, "GenOptionText.Stop")
			}
			if v.ReplyAsJSON {
				unsupported = append(unsupported, "GenOptionText.ReplyAsJSON")
			}
			if v.DecodeAs != nil {
				unsupported = append(unsupported, "GenOptionText.DecodeAs")
			}
		default:
			unsupported = append(unsupported, internal.TypeName(opt))
		}
	}
	if len(unsupported) != 0 {
		return co, &base.ErrNotSupported{Options: unsupported}
	}
	return co, nil
}

// sessionMode returns the mode to select: GenOption.Mode, or agentName for a
// system prompt without a mode.
func (co *callOpts) sessionMode() Mode {
	if co.mode == "" && co.systemPrompt != "" {
		return agentName
	}
	return co.mode
}

// wrapACP adds the provider name to an error from the shared ACP engine.
func wrapACP(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("opencode: %w", err)
}

// acpOptions returns the ACP settings for one call in cwd.
func acpOptions(cwd string) *acp.Options {
	return &acp.Options{
		ClientName:         "genai-opencode",
		ClientVersion:      "1.0.0",
		Cwd:                cwd,
		ResumeMethod:       acp.MethodSessionLoad,
		SetModelMethod:     acp.MethodSessionSetModel,
		ModelID:            acp.ConfigOptionModel,
		EffortID:           ConfigOptionEffort,
		ModeID:             ConfigOptionMode,
		AllowUnlistedModel: true,
	}
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

// Compile-time interface checks.
var (
	_ genai.Provider     = (*Client)(nil)
	_ genai.ProviderPing = (*Client)(nil)
)
