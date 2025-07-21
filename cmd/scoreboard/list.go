// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"fmt"
	"maps"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers"
	"github.com/maruel/genai/providers/openaicompatible"
)

func printList() error {
	all := maps.Clone(providers.All)
	all["openaicompatible"] = func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return openaicompatible.New("http://localhost:8080/v1", nil, model, wrapper)
	}
	names := make([]string, 0, len(all))
	for name := range all {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		fmt.Printf("- %s\n", name)
		c, err := all[name]("", nil)
		// The function can return an error and still return a client when no API key was found. It's okay here
		// because we won't use the service provider.
		if c == nil {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: %v\n", name, err)
			continue
		}
		ps, ok := c.(genai.ProviderScoreboard)
		if !ok {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: doesn't support scoreboard\n", name)
			continue
		}
		async := ""
		if _, isAsync := c.(genai.ProviderGenAsync); isAsync {
			async = "✅batch "
		}
		for _, scenario := range ps.Scoreboard().Scenarios {
			m := scenario.Models
			if len(m) > 3 {
				m = append(slices.Clone(m[:3]), "...")
			}
			fmt.Printf("  - %s\n", strings.Join(m, ", "))
			if isTextOnly(scenario.In) && isTextOnly(scenario.Out) {
				fmt.Printf("    in/out:   text only\n")
			} else {
				in := getDeliveryConstraints(scenario.In)
				out := getDeliveryConstraints(scenario.Out)
				fmt.Printf("    in/out:   ⇒ %s%s / %s%s ⇒\n", modalityMapToString(scenario.In), in, modalityMapToString(scenario.Out), out)
			}
			chat := ""
			stream := ""
			if scenario.GenSync != nil {
				chat = functionalityText(scenario.GenSync)
			}
			if scenario.GenStream != nil {
				stream = functionalityText(scenario.GenStream)
			}
			if chat == stream {
				if chat != "" {
					fmt.Printf("    features: %s%s\n", async, chat)
				}
			} else {
				if chat != "" {
					fmt.Printf("    buffered: %s%s\n", async, chat)
				}
				if stream != "" {
					fmt.Printf("    streamed: %s%s\n", async, stream)
				}
			}
			if scenario.GenDoc != nil {
				if s := functionalityDoc(scenario.GenDoc); s != "" {
					fmt.Printf("    doc:      %s\n", s)
				}
			}
		}
	}
	return nil
}

// modalityMapToString converts a modality capability map to a readable string
func modalityMapToString(m map[genai.Modality]genai.ModalCapability) string {
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

// isTextOnly checks if a modality map contains only text
func isTextOnly(m map[genai.Modality]genai.ModalCapability) bool {
	return len(m) == 1 && m[genai.ModalityText].Inline
}

func functionalityText(f *genai.FunctionalityText) string {
	var items []string
	if f.JSON {
		items = append(items, "✅json")
	}
	if f.JSONSchema {
		items = append(items, "✅jsonschema")
	}
	flakyTool := false
	switch f.Tools {
	case genai.True:
		items = append(items, "✅tools")
	case genai.Flaky:
		items = append(items, "✅tools")
		flakyTool = true
	}

	if flakyTool {
		items = append(items, "💔flaky tool")
	} else if f.BiasedTool == genai.True && f.IndecisiveTool == genai.False {
		// Flaky is okay.
		items = append(items, "💔biased tool")
	}
	if f.BrokenTokenUsage != genai.False {
		items = append(items, "💔usage")
	}
	if f.BrokenFinishReason {
		items = append(items, "💔finishreason")
	}
	if f.NoStopSequence {
		items = append(items, "💔stopsequence")
	}
	return strings.Join(items, " ")
}

func functionalityDoc(f *genai.FunctionalityDoc) string {
	var items []string
	if f.BrokenTokenUsage != genai.False {
		items = append(items, "💔usage")
	}
	if f.BrokenFinishReason {
		items = append(items, "💔finishreason")
	}
	return strings.Join(items, " ")
}

// getDeliveryConstraints analyzes modality capabilities to determine delivery constraints
func getDeliveryConstraints(capabilities map[genai.Modality]genai.ModalCapability) string {
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
		if cap.Inline && cap.URL {
			hasBoth = true
			allInlineOnly = false
			allURLOnly = false
		} else if cap.Inline {
			allURLOnly = false
		} else if cap.URL {
			allInlineOnly = false
		}
	}

	if hasBoth {
		return ""
	} else if allInlineOnly {
		return " (inline only)"
	} else if allURLOnly {
		return " (url only)"
	}
	return ""
}
