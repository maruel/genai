// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command scoreboard generates a scoreboard for every providers supported.
package main

import (
	"context"
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/genai/openai"
	"github.com/maruel/genai/perplexity"
	"github.com/maruel/genai/togetherai"
)

// providers is the list of known providers. We only look at their scoreboard, so no need for an API key or
// model name.
var providers = map[string]func() (genai.ProviderChat, error){
	"anthropic": func() (genai.ProviderChat, error) {
		return anthropic.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"cerebras": func() (genai.ProviderChat, error) {
		return cerebras.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"cloudflare": func() (genai.ProviderChat, error) {
		return cloudflare.New("FAKE_API_KEY", "", "FAKE_MODEL", nil)
	},
	"cohere": func() (genai.ProviderChat, error) {
		return cohere.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"deepseek": func() (genai.ProviderChat, error) {
		return deepseek.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"gemini": func() (genai.ProviderChat, error) {
		return gemini.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"groq": func() (genai.ProviderChat, error) {
		return groq.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"huggingface": func() (genai.ProviderChat, error) {
		return huggingface.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"mistral": func() (genai.ProviderChat, error) {
		return mistral.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"openai": func() (genai.ProviderChat, error) {
		return openai.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"perplexity": func() (genai.ProviderChat, error) {
		return perplexity.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"togetherai": func() (genai.ProviderChat, error) {
		return togetherai.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
}

func mainImpl() error {
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
		for _, scenario := range ps.Scoreboard().Scenarios {
			m := scenario.Models
			if len(m) > 3 {
				m = append(append(([]string)(nil), m[:3]...), "...")
			}
			fmt.Printf("  - %s\n", strings.Join(m, ", "))
			if slices.Equal(scenario.In, textOnly) && slices.Equal(scenario.Out, textOnly) {
				fmt.Printf("    in/out:   text only\n")
			} else {
				v := ""
				if scenario.Chat.Inline && !scenario.Chat.URL {
					v = " (inline only)"
				}
				if scenario.Chat.URL && !scenario.Chat.Inline {
					v = " (url only)"
				}
				fmt.Printf("    in/out:   ⇒ %s%s / %s ⇒\n", scenario.In, v, scenario.Out)
			}
			chat := functionality(&scenario.Chat)
			stream := functionality(&scenario.ChatStream)
			if chat == stream {
				if chat != "" {
					fmt.Printf("    features: %s\n", chat)
				}
			} else {
				if chat != "" {
					fmt.Printf("    buffered: %s\n", chat)
				}
				if stream != "" {
					fmt.Printf("    streamed: %s\n", stream)
				}
			}
		}
	}
	return nil
}

var textOnly = genai.Modalities{genai.ModalityText}

func functionality(f *genai.Functionality) string {
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
	} else if !f.UnbiasedTool {
		items = append(items, "💔biased tool")
	}
	if !f.ReportTokenUsage {
		items = append(items, "💔usage")
	}
	if !f.ReportFinishReason {
		items = append(items, "💔finishreason")
	}
	if !f.StopSequence {
		items = append(items, "💔stopsequence")
	}
	return strings.Join(items, ", ")
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "scoreboard: %s\n", err)
		}
		os.Exit(1)
	}
}
