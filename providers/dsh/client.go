// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package dsh implements a genai provider backed by the DeepSeek
// Harness CLI.
//
// Instead of making HTTP requests directly, it launches `dsh --profile acp` as
// a subprocess and communicates over stdin/stdout using the ACP (Agent Client
// Protocol) JSON-RPC 2.0 protocol.
//
// See https://github.com/deepseek-ai/deepseek-harness for DeepSeek Harness
// documentation and https://agentclientprotocol.com for the ACP specification.
//
// # Protocol
//
// The provider performs a JSON-RPC handshake (initialize → session/new) on each
// call, applies the model and reasoning effort through
// session/set_config_option, then sends a session/prompt request and reads
// session/update notifications until the session/prompt response arrives.
//
// # Model IDs
//
// ACP identifies a model by an opaque route value. This provider exposes it as
// "provider/model", for example "deepseek-official/deepseek-v4-flash".
//
// # Usage
//
// The session/prompt result carries only stopReason. Context occupancy arrives
// as a usage_update notification and is reported in Usage.ContextTokens and
// Usage.ContextWindow. Per-request token counts are unavailable.
//
// # Session / multi-turn
//
// Each GenSync or GenStream call launches a fresh subprocess. The session ID is
// always returned inside Reply.Opaque["session_id"]. When the message history
// contains a previous session ID, it is automatically picked up: session/resume
// is used instead of session/new and only the last user message is sent.
//
// # System prompt
//
// ACP has no system prompt field. For GenOptionText.SystemPrompt, the provider
// writes the profile patch genai.patch.yml to the temporary working directory
// and passes it with --patch. Consequences:
//
//   - The prompt replaces only dsh's harness identity and persona. dsh still
//     appends its first-party guidance and runtime context, and the agent
//     keeps its tools.
//   - dsh treats {{name}} in the prompt as a template variable, such as
//     {{model}}. An unknown or malformed reference fails the call; dsh has no
//     escape syntax. A lone "{{" without a later "}}" stays literal.
//   - dsh reads launcher flags such as --patch only before the first app
//     argument. A leading --patch in ProviderOption.ExtraArgs applies after
//     the provider's patch; one that targets the system-prompt entry replaces
//     the system prompt.
//   - The patch file is visible to the agent's tools in its working
//     directory.
package dsh

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

// Scoreboard for the DeepSeek Harness CLI provider.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
}

// ProviderOption configures the dsh process. It implements genai.ProviderOption.
//
// All fields are opt-in. By default the provider rejects every permission
// request the agent makes.
type ProviderOption struct {
	// DangerouslySkipPermissions approves every permission request the agent
	// makes, such as file edits and shell commands.
	DangerouslySkipPermissions bool
	// ExtraArgs are appended to the `dsh --profile acp` command line, after the
	// --patch that carries GenOptionText.SystemPrompt.
	ExtraArgs []string

	_ struct{}
}

// Validate implements genai.ProviderOption.
func (p *ProviderOption) Validate() error {
	return msgutil.CheckExtraArgs(p.ExtraArgs, "profile")
}

// Client is a genai provider that delegates to the local `dsh` CLI.
type Client struct {
	base.NotImplemented

	starterWrapper genai.ProviderOptionStarterWrapper
	bin            string
	model          string
	modelRoute     string
	opts           ProviderOption
	binOnce        sync.Once
	binErr         error
}

// New creates a Client for the `dsh` CLI.
//
// The binary is located lazily on the first call to GenSync, GenStream, or
// Ping, so New succeeds even when the CLI is not installed.
//
// Supported ProviderOptions:
//   - genai.ProviderOptionModel — model ID (e.g.
//     "deepseek-official/deepseek-v4-flash"). genai.ModelCheap and
//     genai.ModelGood select "deepseek-official/deepseek-flash";
//     genai.ModelSOTA selects "deepseek-official/deepseek-v4-pro".
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
			switch v {
			case genai.ModelCheap, genai.ModelGood:
				c.model = "deepseek-official/deepseek-flash"
			case genai.ModelSOTA:
				c.model = "deepseek-official/deepseek-v4-pro"
			default:
				c.model = string(v)
			}
			route, err := encodeModel(c.model)
			if err != nil {
				return nil, err
			}
			c.modelRoute = route
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
func (c *Client) Name() string { return "dsh" }

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

// Ping implements genai.ProviderPing by running `dsh --version`.
func (c *Client) Ping(ctx context.Context) error {
	if err := c.ensureBin(); err != nil {
		return err
	}
	out, err := exec.CommandContext(ctx, c.bin, "--version").Output()
	if err != nil {
		return fmt.Errorf("dsh --version: %w", err)
	}
	if len(bytes.TrimSpace(out)) == 0 {
		return errors.New("dsh --version returned empty output")
	}
	return nil
}

// ListModels implements genai.Provider by querying the DeepSeek Harness ACP
// server's model selector. It launches a subprocess, performs the handshake,
// and returns the routes from session/new.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if err := c.ensureBin(); err != nil {
		return nil, err
	}
	stdin, stdout, wait, cwd, err := c.start(ctx, "")
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
	for _, choice := range choices {
		id, err := decodeModel(choice.Value)
		if err != nil {
			return nil, err
		}
		name := choice.Name
		if name == "" {
			name = id
		}
		out = append(out, &model{id: id, displayName: name})
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
	req := c.sessionRequest(msgs, co)

	stdin, stdout, wait, cwd, err := c.start(ctx, co.systemPrompt)
	if err != nil {
		return genai.Result{}, err
	}
	defer func() {
		_ = stdin.Close()
		err = errors.Join(err, wait())
	}()

	s, err := acp.Handshake(stdin, stdout, acpOptions(cwd), req)
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
	req := c.sessionRequest(msgs, co)

	var (
		result   genai.Result
		finalErr error
	)
	seq := func(yield func(genai.Reply) bool) {
		stdin, stdout, wait, cwd, startErr := c.start(ctx, co.systemPrompt)
		if startErr != nil {
			finalErr = startErr
			return
		}
		defer func() {
			_ = stdin.Close()
			finalErr = errors.Join(finalErr, wait())
		}()

		s, hsErr := acp.Handshake(stdin, stdout, acpOptions(cwd), req)
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

// ensureBin locates the dsh binary on first call. Safe for concurrent use.
func (c *Client) ensureBin() error {
	c.binOnce.Do(func() {
		bin, err := exec.LookPath("dsh")
		if err != nil {
			if c.starterWrapper == nil {
				c.binErr = fmt.Errorf("dsh CLI not found on PATH: %w", err)
				return
			}
			bin = "dsh"
		}
		c.bin = bin
	})
	return c.binErr
}

// start launches the ACP subprocess in a fresh temporary directory and returns
// the ACP session working directory. A non-empty systemPrompt is written there
// as a profile patch.
func (c *Client) start(ctx context.Context, systemPrompt string) (io.WriteCloser, io.ReadCloser, func() error, string, error) {
	dir, err := os.MkdirTemp("", "genai-dsh-*")
	if err != nil {
		return nil, nil, nil, "", fmt.Errorf("create temp dir: %w", err)
	}
	patch := ""
	if systemPrompt != "" {
		patch = filepath.Join(dir, "genai.patch.yml")
		if err := writePatch(patch, systemPrompt); err != nil {
			return nil, nil, nil, "", errors.Join(err, os.RemoveAll(dir))
		}
	}
	s := cli.NewStarter(c.bin, dir)
	if c.starterWrapper != nil {
		s = c.starterWrapper(s)
	}
	stdin, stdout, wait, err := s(ctx, c.args(patch))
	if err != nil {
		return nil, nil, nil, "", errors.Join(err, os.RemoveAll(dir))
	}
	return stdin, stdout, func() error {
		return errors.Join(wait(), os.RemoveAll(dir))
	}, dir, nil
}

// args returns the command-line arguments for the ACP profile. A non-empty
// patch adds that profile patch file.
func (c *Client) args(patch string) []string {
	args := []string{"--profile", "acp"}
	if patch != "" {
		args = append(args, "--patch", patch)
	}
	return append(args, c.opts.ExtraArgs...)
}

// writePatch writes the profile patch that sets systemPrompt to path. The patch
// list is JSON, which the YAML patch loader accepts.
func writePatch(path, systemPrompt string) error {
	b, err := json.Marshal([]Patch{{
		ID:     patchIDSystemPrompt,
		Config: SystemPromptConfig{PersonaPrefix: systemPrompt},
	}})
	if err != nil {
		return err
	}
	if err := os.WriteFile(path, b, 0o600); err != nil {
		return fmt.Errorf("write patch: %w", err)
	}
	return nil
}

// sessionRequest builds the handshake settings for a call.
func (c *Client) sessionRequest(msgs genai.Messages, co callOpts) acp.SessionRequest {
	return acp.SessionRequest{
		Model:           c.modelRoute,
		Effort:          string(co.effort),
		ResumeSessionID: msgutil.ExtractOpaqueID(msgs, acp.SessionIDKey),
	}
}

// callOpts holds the settings of one call.
type callOpts struct {
	effort       Effort
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
		case genai.EffortLevel:
			switch v {
			case "":
			case genai.EffortOff:
				co.effort = EffortOff
			case genai.EffortMinimal, genai.EffortLow:
				co.effort = EffortLow
			case genai.EffortMedium, genai.EffortHigh:
				co.effort = EffortHigh
			case genai.EffortXHigh, genai.EffortMax:
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

// wrapACP adds the provider name to an error from the shared ACP engine.
func wrapACP(err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("dsh: %w", err)
}

// acpOptions returns the ACP settings for one call in cwd.
func acpOptions(cwd string) *acp.Options {
	return &acp.Options{
		ClientName:    "genai-dsh",
		ClientVersion: "1.0.0",
		Cwd:           cwd,
		ResumeMethod:  acp.MethodSessionResume,
		ModelID:       acp.ConfigOptionModel,
		EffortID:      "reasoning_effort",
	}
}

// encodeModel converts a "provider/model" ID to the opaque ACP route value.
func encodeModel(id string) (string, error) {
	provider, model, ok := strings.Cut(id, "/")
	if !ok || provider == "" || model == "" {
		return "", fmt.Errorf("invalid model ID %q; want provider/model", id)
	}
	raw, err := json.Marshal([2]string{provider, model})
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

// decodeModel converts an opaque ACP route value to a "provider/model" ID.
func decodeModel(value string) (string, error) {
	var route []string
	if err := internal.UnmarshalJSON([]byte(value), &route); err != nil {
		return "", fmt.Errorf("invalid model value %q: %w", value, err)
	}
	if len(route) != 2 || route[0] == "" || route[1] == "" || strings.Contains(route[0], "/") {
		return "", fmt.Errorf("invalid model value %q; want [provider, model]", value)
	}
	return route[0] + "/" + route[1], nil
}

// yieldNothing is an empty iterator used for early error returns in GenStream.
func yieldNothing(func(genai.Reply) bool) {}

// errFinish returns a finish function that always returns the given error.
func errFinish(err error) func() (genai.Result, error) {
	return func() (genai.Result, error) { return genai.Result{}, err }
}

// model is the genai.Model implementation for DeepSeek Harness routes.
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
