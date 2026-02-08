// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"io"
	"maps"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers"
	"github.com/maruel/genai/scoreboard"
)

func printList(ctx context.Context, w io.Writer) error {
	all := maps.Clone(providers.All)
	for _, name := range slices.Sorted(maps.Keys(all)) {
		_, _ = fmt.Fprintf(w, "- %s\n", name)
		c, err := all[name].Factory(ctx)
		// The function can return an error and still return a client when no API key was found. It's okay here
		// because we won't use the service provider.
		if c == nil {
			_, _ = fmt.Fprintf(w, "ignoring provider %s: %v\n", name, err)
			continue
		}
		async := ""
		if c.Capabilities().GenAsync {
			async = "✅batch "
		}
		for _, scenario := range c.Scoreboard().Scenarios {
			m := scenario.Models
			if len(m) > 3 {
				m = append(slices.Clone(m[:3]), "...")
			}
			_, _ = fmt.Fprintf(w, "  - %s\n", strings.Join(m, ", "))
			if isTextOnly(scenario.In) && isTextOnly(scenario.Out) {
				_, _ = fmt.Fprintf(w, "    in/out:   text only\n")
			} else {
				in := getDeliveryConstraints(scenario.In)
				out := getDeliveryConstraints(scenario.Out)
				_, _ = fmt.Fprintf(w, "    in/out:   ⇒ %s%s / %s%s ⇒\n", modalityMapToString(scenario.In), in, modalityMapToString(scenario.Out), out)
			}
			chat := ""
			stream := ""
			if scenario.GenSync != nil {
				chat = functionality(scenario.GenSync)
			}
			if scenario.GenStream != nil {
				stream = functionality(scenario.GenStream)
			}
			if chat == stream {
				if chat != "" {
					_, _ = fmt.Fprintf(w, "    features: %s%s\n", async, chat)
				}
			} else {
				if chat != "" {
					_, _ = fmt.Fprintf(w, "    buffered: %s%s\n", async, chat)
				}
				if stream != "" {
					_, _ = fmt.Fprintf(w, "    streamed: %s%s\n", async, stream)
				}
			}
		}
	}
	return nil
}

// modalityMapToString converts a modality capability map to a readable string.
func modalityMapToString(m map[genai.Modality]scoreboard.ModalCapability) string {
	if len(m) == 0 {
		return ""
	}
	var modalities []string
	for modality := range m {
		modalities = append(modalities, string(modality))
	}
	slices.Sort(modalities)
	return strings.Join(modalities, ", ")
}

// isTextOnly checks if a modality map contains only text.
func isTextOnly(m map[genai.Modality]scoreboard.ModalCapability) bool {
	return len(m) == 1 && m[genai.ModalityText].Inline
}

func functionality(f *scoreboard.Functionality) string {
	var items []string
	if f.JSON {
		items = append(items, "✅json")
	}
	if f.JSONSchema {
		items = append(items, "✅jsonschema")
	}
	flakyTool := false
	switch f.Tools {
	case scoreboard.True:
		items = append(items, "✅tools")
	case scoreboard.Flaky:
		items = append(items, "✅tools")
		flakyTool = true
	case scoreboard.False:
	}

	if flakyTool {
		items = append(items, "💔flaky tool")
	} else if f.ToolsBiased == scoreboard.True && f.ToolsIndecisive == scoreboard.False {
		// Flaky is okay.
		items = append(items, "💔biased tool")
	}
	if f.ReportTokenUsage != scoreboard.True {
		items = append(items, "💔usage")
	}
	if f.ReportFinishReason != scoreboard.True {
		items = append(items, "💔finishreason")
	}
	if !f.StopSequence {
		items = append(items, "💔stopsequence")
	}
	return strings.Join(items, " ")
}

// getDeliveryConstraints analyzes modality capabilities to determine delivery constraints.
func getDeliveryConstraints(capabilities map[genai.Modality]scoreboard.ModalCapability) string {
	if len(capabilities) == 0 || isTextOnly(capabilities) {
		return ""
	}

	// Check if all non-text modalities have the same delivery constraints
	allInlineOnly := true
	allURLOnly := true
	hasBoth := false

	for modality, cap := range capabilities {
		if modality == genai.ModalityText {
			continue
		}
		switch {
		case cap.Inline && cap.URL:
			hasBoth = true
			allInlineOnly = false
			allURLOnly = false
		case cap.Inline:
			allURLOnly = false
		case cap.URL:
			allInlineOnly = false
		}
	}

	switch {
	case hasBoth:
		return ""
	case allInlineOnly:
		return " (inline only)"
	case allURLOnly:
		return " (url only)"
	}
	return ""
}
