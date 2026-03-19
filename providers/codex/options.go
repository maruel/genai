// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package codex

import (
	"fmt"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
)

// callOpts holds per-call options parsed from the GenOption slice.
type callOpts struct {
	systemPrompt string
}

// parseOpts validates and collects the per-call options.
func parseOpts(opts []genai.GenOption) (callOpts, error) {
	var co callOpts
	var unsupported []string
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return callOpts{}, err
		}
		switch v := opt.(type) {
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
