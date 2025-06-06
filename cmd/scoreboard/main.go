// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command scoreboard generates a scoreboard for every providers supported.
package main

import (
	"context"
	"errors"
	"flag"
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
var providers = map[string]func() (genai.ProviderGen, error){
	"anthropic": func() (genai.ProviderGen, error) {
		return anthropic.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"cerebras": func() (genai.ProviderGen, error) {
		return cerebras.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"cloudflare": func() (genai.ProviderGen, error) {
		return cloudflare.New("FAKE_API_KEY", "", "FAKE_MODEL", nil)
	},
	"cohere": func() (genai.ProviderGen, error) {
		return cohere.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"deepseek": func() (genai.ProviderGen, error) {
		return deepseek.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"gemini": func() (genai.ProviderGen, error) {
		return gemini.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"groq": func() (genai.ProviderGen, error) {
		return groq.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"huggingface": func() (genai.ProviderGen, error) {
		return huggingface.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"mistral": func() (genai.ProviderGen, error) {
		return mistral.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"openai": func() (genai.ProviderGen, error) {
		return openai.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"perplexity": func() (genai.ProviderGen, error) {
		return perplexity.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
	"togetherai": func() (genai.ProviderGen, error) {
		return togetherai.New("FAKE_API_KEY", "FAKE_MODEL", nil)
	},
}

type column struct {
	Provider   string
	Country    string
	Vision     string
	PDF        string
	Audio      string
	Video      string
	JSON       string
	JSONSchema string
	ImageGen   string
	AudioGen   string
	Chat       string
	Streaming  string
	Seed       string
	Tools      string
	Caching    string
}

type columni struct {
	Provider   int
	Country    int
	Vision     int
	PDF        int
	Audio      int
	Video      int
	JSON       int
	JSONSchema int
	ImageGen   int
	AudioGen   int
	Chat       int
	Streaming  int
	Seed       int
	Tools      int
	Caching    int
}

func printTable() error {
	var columns []column
	for name, f := range providers {
		c, err := f()
		if err != nil {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: %v\n", name, err)
			continue
		}
		ps, ok := c.(genai.ProviderScoreboard)
		if !ok {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: doesn't support scoreboard\n", name)
			continue
		}
		sb := ps.Scoreboard()
		col := column{
			Provider: ps.Name(),
			// Country:    sb.Country,
		}
		for _, s := range sb.Scenarios {
			if slices.Contains(s.In, genai.ModalityImage) {
				col.Vision = "✅"
			} else {
				col.Vision = "❌"
			}
			if slices.Contains(s.In, genai.ModalityPDF) {
				col.PDF = "✅"
			} else {
				col.PDF = "❌"
			}
			if slices.Contains(s.In, genai.ModalityAudio) {
				col.Audio = "✅"
			} else {
				col.Audio = "❌"
			}
			if slices.Contains(s.In, genai.ModalityVideo) {
				col.Video = "✅"
			} else {
				col.Video = "❌"
			}
			if s.GenSync.JSON && s.GenStream.JSON {
				col.JSON = "✅"
			} else {
				col.JSON = "❌"
			}
			if s.GenSync.JSONSchema && s.GenStream.JSONSchema {
				col.JSONSchema = "✅"
			} else {
				col.JSONSchema = "❌"
			}
			if slices.Contains(s.In, genai.ModalityText) && slices.Contains(s.Out, genai.ModalityImage) {
				col.ImageGen = "✅"
			} else {
				col.ImageGen = "❌"
			}
			if slices.Contains(s.In, genai.ModalityText) && slices.Contains(s.Out, genai.ModalityText) {
				col.Chat = "✅"
				col.Streaming = "✅"
			} else {
				col.Chat = "❌"
				col.Streaming = "❌"
			}
			if s.GenSync.Tools == genai.True && s.GenStream.Tools == genai.True {
				col.Tools = "✅"
			} else {
				col.Tools = "❌"
			}
			// TODO: Add these.
			col.AudioGen = "❌"
			col.Seed = "❌"
			col.Caching = "❌"
		}
		columns = append(columns, col)
	}
	slices.SortFunc(columns, func(a, b column) int {
		return strings.Compare(a.Provider, b.Provider)
	})
	columns = append([]column{
		{
			Provider:   "Provider",
			Vision:     "➛Vision",
			PDF:        "➛PDF",
			Audio:      "➛Audio",
			Video:      "➛Video",
			JSON:       "JSON➛",
			JSONSchema: "JSON+Schema➛",
			ImageGen:   "Image➛",
			AudioGen:   "Audio➛",
			Chat:       "Chat",
			Streaming:  "Streaming",
			Seed:       "Seed",
			Tools:      "Tools",
			Caching:    "Caching",
		},
	}, columns...)
	coli := columni{}
	for _, c := range columns {
		if i := countCharsWithEmoji(c.Provider); i > coli.Provider {
			coli.Provider = i
		}
		if i := countCharsWithEmoji(c.Vision); i > coli.Vision {
			coli.Vision = i
		}
		if i := countCharsWithEmoji(c.PDF); i > coli.PDF {
			coli.PDF = i
		}
		if i := countCharsWithEmoji(c.Audio); i > coli.Audio {
			coli.Audio = i
		}
		if i := countCharsWithEmoji(c.Video); i > coli.Video {
			coli.Video = i
		}
		if i := countCharsWithEmoji(c.JSON); i > coli.JSON {
			coli.JSON = i
		}
		if i := countCharsWithEmoji(c.JSONSchema); i > coli.JSONSchema {
			coli.JSONSchema = i
		}
		if i := countCharsWithEmoji(c.ImageGen); i > coli.ImageGen {
			coli.ImageGen = i
		}
		if i := countCharsWithEmoji(c.AudioGen); i > coli.AudioGen {
			coli.AudioGen = i
		}
		if i := countCharsWithEmoji(c.Chat); i > coli.Chat {
			coli.Chat = i
		}
		if i := countCharsWithEmoji(c.Streaming); i > coli.Streaming {
			coli.Streaming = i
		}
		if i := countCharsWithEmoji(c.Seed); i > coli.Seed {
			coli.Seed = i
		}
		if i := countCharsWithEmoji(c.Tools); i > coli.Tools {
			coli.Tools = i
		}
		if i := countCharsWithEmoji(c.Caching); i > coli.Caching {
			coli.Caching = i
		}
	}

	// | Country
	{
		c := columns[0]
		fmt.Printf("| %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s |\n",
			coli.Provider, c.Provider,
			coli.Vision, c.Vision,
			coli.PDF, c.PDF,
			coli.Audio, c.Audio,
			coli.Video, c.Video,
			coli.JSON, c.JSON,
			coli.JSONSchema, c.JSONSchema,
			coli.ImageGen, c.ImageGen,
			coli.AudioGen, c.AudioGen,
			coli.Chat, c.Chat,
			coli.Streaming, c.Streaming,
			coli.Seed, c.Seed,
			coli.Tools, c.Tools,
			coli.Caching, c.Caching)
		fmt.Printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
			strings.Repeat("-", coli.Provider),
			strings.Repeat("-", coli.Vision),
			strings.Repeat("-", coli.PDF),
			strings.Repeat("-", coli.Audio),
			strings.Repeat("-", coli.Video),
			strings.Repeat("-", coli.JSON),
			strings.Repeat("-", coli.JSONSchema),
			strings.Repeat("-", coli.ImageGen),
			strings.Repeat("-", coli.AudioGen),
			strings.Repeat("-", coli.Chat),
			strings.Repeat("-", coli.Streaming),
			strings.Repeat("-", coli.Seed),
			strings.Repeat("-", coli.Tools),
			strings.Repeat("-", coli.Caching))
	}
	for _, c := range columns[1:] {
		fmt.Printf("| %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s |\n",
			coli.Provider, c.Provider,
			coli.Vision-1, c.Vision,
			coli.PDF-1, c.PDF,
			coli.Audio-1, c.Audio,
			coli.Video-1, c.Video,
			coli.JSON-1, c.JSON,
			coli.JSONSchema-1, c.JSONSchema,
			coli.ImageGen-1, c.ImageGen,
			coli.AudioGen-1, c.AudioGen,
			coli.Chat-1, c.Chat,
			coli.Streaming-1, c.Streaming,
			coli.Seed-1, c.Seed,
			coli.Tools-1, c.Tools,
			coli.Caching-1, c.Caching)
	}
	return nil
}

func countCharsWithEmoji(s string) int {
	count := 0
	for _, r := range s {
		switch r {
		// case '✅', '💔', '❌':
		//	count += 3
		default:
			count += 1
		}
	}
	return count
}

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
		for _, scenario := range ps.Scoreboard().Scenarios {
			m := scenario.Models
			if len(m) > 3 {
				m = append(slices.Clone(m[:3]), "...")
			}
			fmt.Printf("  - %s\n", strings.Join(m, ", "))
			if slices.Equal(scenario.In, textOnly) && slices.Equal(scenario.Out, textOnly) {
				fmt.Printf("    in/out:   text only\n")
			} else {
				v := ""
				if scenario.GenSync.Inline && !scenario.GenSync.URL {
					v = " (inline only)"
				}
				if scenario.GenSync.URL && !scenario.GenSync.Inline {
					v = " (url only)"
				}
				fmt.Printf("    in/out:   ⇒ %s%s / %s ⇒\n", scenario.In, v, scenario.Out)
			}
			chat := functionality(&scenario.GenSync)
			stream := functionality(&scenario.GenStream)
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

func functionality(f *genai.FunctionalityText) string {
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
	if f.BrokenTokenUsage {
		items = append(items, "💔usage")
	}
	if f.BrokenFinishReason {
		items = append(items, "💔finishreason")
	}
	if f.NoStopSequence {
		items = append(items, "💔stopsequence")
	}
	return strings.Join(items, ", ")
}

func mainImpl() error {
	table := flag.Bool("table", false, "output a markdown table")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("unexpected arguments")
	}
	if *table {
		return printTable()
	}
	return printList()
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "scoreboard: %s\n", err)
		}
		os.Exit(1)
	}
}
