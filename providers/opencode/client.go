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
	"strings"
	"sync"

	"github.com/maruel/genai"
	"github.com/maruel/genai/acp"
	"github.com/maruel/genai/base"
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
	// Effort selects the default reasoning-effort model variant. Use the
	// Effort* constants or a provider-specific variant returned by OpenCode.
	// GenOption.Effort overrides it per call.
	Effort Effort
	// DangerouslySkipPermissions approves every permission request the agent
	// makes, such as file edits and shell commands.
	DangerouslySkipPermissions bool
	// ExtraArgs are appended to the `opencode acp` command line.
	ExtraArgs []string

	_ struct{}
}

// Validate implements genai.ProviderOption.
func (p *ProviderOption) Validate() error {
	if p.Effort != "" && strings.TrimSpace(string(p.Effort)) == "" {
		return errors.New("ProviderOption.Effort: level must not be whitespace")
	}
	return msgutil.CheckExtraArgs(p.ExtraArgs)
}

// GenOption configures one prompt of a running opencode session.
type GenOption struct {
	// Effort selects the reasoning-effort model variant, overriding
	// ProviderOption.Effort. An empty value keeps the session's effort.
	Effort Effort
	// Mode selects an OpenCode session mode returned by ACP. An empty value
	// leaves OpenCode's current mode intact.
	Mode Mode

	_ struct{}
}

// Validate implements genai.GenOption.
func (g *GenOption) Validate() error {
	if g.Effort != "" && strings.TrimSpace(string(g.Effort)) == "" {
		return errors.New("GenOption.Effort: level must not be whitespace")
	}
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
// Ping, so New succeeds even when the CLI is not installed.
//
// Supported ProviderOptions:
//   - genai.ProviderOptionModel — model ID (e.g. "opencode/big-pickle").
//     genai.ModelCheap, genai.ModelGood, and genai.ModelSOTA are rejected.
//   - *ProviderOption — process settings.
//   - genai.ProviderOptionStarterWrapper — intercepts subprocess creation.
func New(_ context.Context, opts ...genai.ProviderOption) (*Client, error) {
	c := &Client{}
	if err := base.CheckDuplicateProviderOptions(opts); err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionModel:
			if err := msgutil.RejectModelMarker(v); err != nil {
				return nil, err
			}
			c.model = string(v)
		case *ProviderOption:
			c.opts = *v
		case genai.ProviderOptionStarterWrapper:
			c.starterWrapper = v
		default:
			return nil, fmt.Errorf("unsupported provider option %T", opt)
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
	stdin, stdout, wait, cwd, err := c.start(ctx, []string{"acp"})
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
	co, optsErr := parseOpts(&c.opts, opts)
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

	stdin, stdout, wait, cwd, err := c.start(ctx, append([]string{"acp"}, c.opts.ExtraArgs...))
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
		Mode:            string(co.mode),
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
	co, optsErr := parseOpts(&c.opts, opts)
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
		stdin, stdout, wait, cwd, startErr := c.start(ctx, append([]string{"acp"}, c.opts.ExtraArgs...))
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
			Mode:            string(co.mode),
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
// the ACP session working directory.
func (c *Client) start(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, string, error) {
	dir, err := os.MkdirTemp("", "genai-opencode-*")
	if err != nil {
		return nil, nil, nil, "", fmt.Errorf("create temp dir: %w", err)
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

// callOpts holds the settings of one call.
type callOpts struct {
	effort Effort
	mode   Mode
}

// parseOpts collects the call settings from the process settings p and opts.
func parseOpts(p *ProviderOption, opts []genai.GenOption) (callOpts, error) {
	co := callOpts{effort: p.Effort}
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
			co.effort = cmp.Or(v.Effort, co.effort)
			co.mode = v.Mode
		default:
			unsupported = append(unsupported, fmt.Sprintf("%T", opt))
		}
	}
	if len(unsupported) != 0 {
		return co, &base.ErrNotSupported{Options: unsupported}
	}
	return co, nil
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
