// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Codex subscription authentication and request policy for Responses.
package openairesponses

import (
	"errors"
	"fmt"
	"regexp"

	"github.com/maruel/genai/base"
)

const apiURL = "https://chatgpt.com/backend-api/codex"

// Auth selects the credential source for OpenAI Responses.
type Auth string

// Validate implements genai.Validatable.
func (a Auth) Validate() error {
	switch a {
	case AuthAPIKey, AuthCodex:
		return nil
	default:
		return fmt.Errorf("unsupported OpenAI authentication mode %q", a)
	}
}

// Authentication modes. The zero value uses the OpenAI API key.
const (
	AuthAPIKey Auth = ""
	// AuthCodex reads $CODEX_HOME/auth.json (default ~/.codex/auth.json).
	// Sign in with codex login and cli_auth_credentials_store = "file" first.
	// The transport never starts Codex. Keyring credential storage is unsupported.
	AuthCodex Auth = "codex"
)

// ProviderOption configures the application's identity on subscription requests.
// AppName is a bounded ASCII identifier sent as Originator and User-Agent. An
// empty name uses "genai". It does not change the OAuth client registration.
type ProviderOption struct {
	Auth    Auth
	AppName string
}

// Validate implements genai.ProviderOption.
func (o *ProviderOption) Validate() error {
	if err := o.Auth.Validate(); err != nil {
		return err
	}
	if o.AppName != "" && !appNamePattern.MatchString(o.AppName) {
		return errors.New("AppName must contain 1-64 ASCII letters, digits, dots, underscores, or hyphens and start with a letter or digit")
	}
	return nil
}

var appNamePattern = regexp.MustCompile(`^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$`)

// prepareCodexRequest enforces the subscription endpoint's request contract.
func prepareCodexRequest(r *Response) error {
	if r.Model == "" {
		return errors.New("codex authentication requires a model for generation")
	}
	if r.PreviousResponseID != "" || r.Background {
		return errors.New("codex authentication requires full history without server-side state")
	}
	var unsupported []string
	add := func(set bool, name string) {
		if set {
			unsupported = append(unsupported, name)
		}
	}
	add(r.MaxOutputTokens != 0, "GenOptionText.MaxTokens")
	add(r.Temperature != 0, "GenOptionText.Temperature")
	add(r.TopP != 0, "GenOptionText.TopP")
	add(r.TopLogprobs != 0, "GenOptionText.TopLogprobs")
	add(r.MaxToolCalls != 0, "max_tool_calls")
	add(r.Truncation != "", "truncation")
	add(len(r.Metadata) != 0, "metadata")
	add(r.PromptCacheRetention != "", "prompt_cache_retention")
	add(r.SafetyIdentifier != "", "safety_identifier")
	add(r.User != "", "user")
	add(r.Moderation != "", "moderation")
	if len(unsupported) != 0 {
		return &base.ErrNotSupported{Options: unsupported}
	}
	if len(r.Input) == 0 {
		return errors.New("codex authentication requires input history")
	}
	r.Store = false
	r.Stream = true
	r.codex = true
	return nil
}

// Model is the useful subset of the account-specific Codex model catalog.
type codexModel struct {
	Slug          string `json:"slug"`
	DisplayName   string `json:"display_name"`
	ContextWindow int64  `json:"context_window"`
}

// GetID implements genai.Model.
func (m *codexModel) GetID() string { return m.Slug }

// String implements fmt.Stringer.
func (m *codexModel) String() string { return m.DisplayName }

// Context implements genai.Model.
func (m *codexModel) Context() int64 { return m.ContextWindow }

type codexModelsResponse struct {
	Models []codexModel `json:"models"`
}
