// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// OpenCode ACP session option values.

package opencode

import "github.com/maruel/genai/acp"

// Session configuration option IDs exposed by OpenCode.
const (
	ConfigOptionEffort = acp.ConfigOptionID("effort")
	ConfigOptionMode   = acp.ConfigOptionID("mode")
)

// Effort is an OpenCode reasoning-effort value serialized in the ACP effort
// configuration option.
//
// Availability is provider- and model-specific. The provider validates it
// against the effort values returned by the ACP session.
type Effort string

// Reasoning effort levels supported by OpenCode model variants.
const (
	EffortDefault Effort = "default"
	EffortNone    Effort = "none"
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortXHigh   Effort = "xhigh"
	EffortMax     Effort = "max"
)

// Mode is an OpenCode session-mode value serialized in the ACP mode
// configuration option.
type Mode string
