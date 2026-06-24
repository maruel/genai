// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Claude Code provider configuration options.

package claudecode

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// ProviderOptionAPIKeyAuth keeps the ANTHROPIC_API_KEY environment variable in
// the subprocess environment, causing Claude Code to authenticate with the API
// key instead of OAuth.
//
// By default the provider strips ANTHROPIC_API_KEY so that Claude Code uses the
// user's subscription (OAuth). Set this option when the caller intentionally
// wants API key billing.
type ProviderOptionAPIKeyAuth bool

// Validate implements genai.Validatable.
func (p ProviderOptionAPIKeyAuth) Validate() error { return nil }

// ControlHandler answers Claude Code control requests emitted on stdout.
//
// The returned response may omit Type and Response.RequestID; the provider
// fills them from the control request before writing the response to stdin.
type ControlHandler func(context.Context, OutputControlRequestMsg) (InputControlResponseMsg, error)

// Effort levels for the Effort field in GenOption.
const (
	EffortLow    = "low"
	EffortMedium = "medium"
	EffortHigh   = "high"
	EffortXHigh  = "xhigh"
	EffortMax    = "max"
)

// GenOption configures a Claude Code CLI call.
//
// All fields are opt-in. By default the subprocess runs with all tools
// disabled, no slash-command skills, no user settings, and no budget cap.
type GenOption struct {
	// Tools enables specific Claude Code built-in tools.
	// Nil (default) disables all tools (--tools "").
	// When set, you likely also need PermissionMode.
	Tools []string
	// Skills enables Claude Code slash-command skills.
	Skills bool
	// ProjectSettings enables loading all settings sources (user, project,
	// and local), including any CLAUDE.md files in the hierarchy.
	ProjectSettings bool
	// MaxBudgetUSD caps the dollar amount the session may spend (--max-budget-usd).
	MaxBudgetUSD float64
	// PermissionMode sets the permission mode (--permission-mode).
	// Valid values: "acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan".
	PermissionMode string
	// ControlHandler handles Claude Code stdout control requests. When set, the
	// provider enables the stdio permission prompt tool and sends the returned
	// control response back to Claude on stdin.
	ControlHandler ControlHandler
	// Effort sets the reasoning effort level (--effort).
	// Use the Effort* constants.
	Effort string

	_ struct{}
}

// Validate implements genai.GenOption.
func (g *GenOption) Validate() error {
	for _, t := range g.Tools {
		if strings.TrimSpace(t) == "" {
			return errors.New("GenOption.Tools: tool name must not be empty")
		}
	}
	if g.MaxBudgetUSD < 0 {
		return fmt.Errorf("GenOption.MaxBudgetUSD must be non-negative, got %g", g.MaxBudgetUSD)
	}
	if g.PermissionMode != "" {
		switch g.PermissionMode {
		case "acceptEdits", "auto", "bypassPermissions", "default", "dontAsk", "plan":
		default:
			return fmt.Errorf("GenOption.PermissionMode: invalid mode %q; must be one of acceptEdits, auto, bypassPermissions, default, dontAsk, plan", g.PermissionMode)
		}
	}
	if g.Effort != "" {
		switch g.Effort {
		case "low", "medium", "high", "xhigh", "max":
		default:
			return fmt.Errorf("GenOption.Effort: invalid level %q; must be one of low, medium, high, xhigh, max", g.Effort)
		}
	}
	return nil
}

// callOpts holds per-call options parsed from the GenOption slice.
type callOpts struct {
	tools             []string // nil = disabled; non-nil = enabled tool list
	skills            bool
	projSettings      bool
	maxBudgetUSD      float64
	permissionMode    string
	controlHandler    ControlHandler
	effort            string
	systemPrompt      string
	progressSummaries bool
}

// parseOpts validates and collects the per-call options.
func parseOpts(opts []genai.GenOption) (callOpts, error) {
	var co callOpts
	var unsupported []string
	var webTools []string
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return callOpts{}, err
		}
		switch v := opt.(type) {
		case *GenOption:
			co.tools = append(co.tools, v.Tools...)
			co.skills = v.Skills
			co.projSettings = v.ProjectSettings
			co.maxBudgetUSD = v.MaxBudgetUSD
			co.permissionMode = v.PermissionMode
			if v.ControlHandler != nil {
				if co.controlHandler != nil {
					return callOpts{}, errors.New("GenOption.ControlHandler: multiple handlers configured")
				}
				co.controlHandler = v.ControlHandler
			}
			co.effort = v.Effort
			co.progressSummaries = v.Effort != ""
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
		case genai.GenOptionSeed:
			unsupported = append(unsupported, "GenOptionSeed")
		case *genai.GenOptionTools:
			unsupported = append(unsupported, "GenOptionTools")
		case *genai.GenOptionWeb:
			if v.Search {
				webTools = append(webTools, "WebSearch")
			}
			if v.Search || v.Fetch {
				webTools = append(webTools, "WebFetch")
			}
		default:
			return callOpts{}, fmt.Errorf("unsupported option %T", opt)
		}
	}
	if len(webTools) > 0 {
		co.tools = append(co.tools, webTools...)
	}
	if len(unsupported) != 0 {
		return co, &base.ErrNotSupported{Options: unsupported}
	}
	return co, nil
}
