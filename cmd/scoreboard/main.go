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
var providers = map[string]func() (genai.ChatProvider, error){
	"anthropic": func() (genai.ChatProvider, error) {
		return anthropic.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"cerebras": func() (genai.ChatProvider, error) {
		return cerebras.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"cloudflare": func() (genai.ChatProvider, error) {
		return cloudflare.New("FAKE_API_KEY", "", "FAKE_MODEL", nil)
	},
	"cohere": func() (genai.ChatProvider, error) {
		return cohere.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"deepseek": func() (genai.ChatProvider, error) {
		return deepseek.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"gemini": func() (genai.ChatProvider, error) {
		return gemini.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"groq": func() (genai.ChatProvider, error) {
		return groq.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"huggingface": func() (genai.ChatProvider, error) {
		return huggingface.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"mistral": func() (genai.ChatProvider, error) {
		return mistral.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"openai": func() (genai.ChatProvider, error) {
		return openai.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"perplexity": func() (genai.ChatProvider, error) {
		return perplexity.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"togetherai": func() (genai.ChatProvider, error) {
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
	var x []string
	if f.JSON {
		x = append(x, "json")
	}
	if f.JSONSchema {
		x = append(x, "jsonschema")
	}
	if f.Tools {
		x = append(x, "tools")
	}
	s := ""
	if len(x) != 0 {
		s = "✅ " + strings.Join(x, ", ")
	}
	if !f.ReportTokenUsage || !f.ReportFinishReason || !f.StopSequence {
		var y []string
		if f.ReportTokenUsage {
			y = append(y, "usage")
		}
		if f.ReportFinishReason {
			y = append(y, "finishreason")
		}
		if f.StopSequence {
			y = append(y, "stopsequence")
		}
		if s != "" {
			s += "  "
		}
		s += "💔 " + strings.Join(y, ", ")
	}
	return s
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "scoreboard: %s\n", err)
		}
		os.Exit(1)
	}
}
