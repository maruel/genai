// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Native DeepSeek Harness ACP effort values.
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
