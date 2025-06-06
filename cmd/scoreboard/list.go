// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/maruel/genai"
)

func printList() error {
	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		fmt.Printf("- %s\n", name)
		c, err := providers[name]()
		if err != nil {
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
			if slices.Equal(scenario.In, textOnly) && slices.Equal(scenario.Out, textOnly) {
				fmt.Printf("    in/out:   text only\n")
			} else {
				in := ""
				out := ""
				if scenario.GenSync != nil {
					if scenario.GenSync.InputInline && !scenario.GenSync.InputURL {
						in = " (inline only)"
					}
					if scenario.GenSync.InputURL && !scenario.GenSync.InputInline {
						in = " (url only)"
					}
				}
				if scenario.GenDoc != nil {
					if scenario.GenDoc.OutputInline && !scenario.GenDoc.OutputURL {
						out = " (inline only)"
					}
					if scenario.GenDoc.OutputURL && !scenario.GenDoc.OutputInline {
						out = " (url only)"
					}
				}
				fmt.Printf("    in/out:   ⇒ %s%s / %s%s ⇒\n", scenario.In, in, scenario.Out, out)
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

var textOnly = genai.Modalities{genai.ModalityText}

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
	if f.BrokenTokenUsage {
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
	if f.BrokenTokenUsage {
		items = append(items, "💔usage")
	}
	if f.BrokenFinishReason {
		items = append(items, "💔finishreason")
	}
	return strings.Join(items, " ")
}
