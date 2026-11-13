// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Native DeepSeek Harness ACP effort values and profile patch entries.
//
// Source: https://github.com/deepseek-ai/deepseek-harness,
// dsh-v0.2.1-rc.5, commit b830ec950faba0a46dda564be3e2d854afed94f1.

package dsh

// Effort is a DeepSeek Harness reasoning-effort value serialized in the ACP
// reasoning_effort configuration option.
//
// Availability is model-specific. The provider validates it against the effort
// values returned by the ACP session.
type Effort string

// Reasoning effort levels supported by the shipped DeepSeek models.
const (
	EffortOff  Effort = "off"
	EffortLow  Effort = "low"
	EffortHigh Effort = "high"
	EffortMax  Effort = "max"
)

// Patch is one entry of a Cordis patch list passed with --patch. It targets
// the profile entry with the same ID. A non-nil Config replaces that entry's
// whole config.
type Patch struct {
	ID       string              `json:"id"`
	Disabled bool                `json:"disabled,omitzero"`
	Config   *SystemPromptConfig `json:"config,omitzero"`
}

// patchIDSystemPrompt is the ID of the @deepseek-ai/dsh-system-prompt entry.
const patchIDSystemPrompt = "system-prompt"

// SystemPromptConfig is the @deepseek-ai/dsh-system-prompt plugin config,
// from packages/core/system-prompt/src/index.ts.
type SystemPromptConfig struct {
	// IncludeHarnessIdentity includes the fixed harness identity before the
	// persona. dsh defaults it to true.
	IncludeHarnessIdentity bool `json:"includeHarnessIdentity"`
	// IncludeRuntimeContext includes optional runtime-context snapshots in the
	// model history. dsh defaults it to true.
	IncludeRuntimeContext bool `json:"includeRuntimeContext"`
	// PersonaPrefix is the persona template before first-party guidance. dsh
	// replaces {{name}} references and fails on unknown or malformed ones.
	PersonaPrefix string `json:"personaPrefix,omitzero"`
}
