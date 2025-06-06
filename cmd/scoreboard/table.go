// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/maruel/genai"
)

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
	Doc        string
	Batch      string
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
	Doc        int
	Batch      int
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
		_, isText := c.(genai.ProviderGen)
		_, isDoc := c.(genai.ProviderGenDoc)
		_, isAsync := c.(genai.ProviderGenAsync)
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
		col.Vision = "❌"
		col.PDF = "❌"
		col.Audio = "❌"
		col.Video = "❌"
		col.JSON = "❌"
		col.JSONSchema = "❌"
		col.ImageGen = "❌"
		col.Chat = "❌"
		col.Streaming = "❌"
		col.Doc = "❌"
		col.Batch = "❌"
		col.Tools = "❌"
		col.AudioGen = "❌"
		for _, s := range sb.Scenarios {
			if slices.Contains(s.In, genai.ModalityImage) {
				col.Vision = "✅"
			}
			if slices.Contains(s.In, genai.ModalityPDF) {
				col.PDF = "✅"
			}
			if slices.Contains(s.In, genai.ModalityAudio) {
				col.Audio = "✅"
			}
			if slices.Contains(s.In, genai.ModalityVideo) {
				col.Video = "✅"
			}
			if s.GenSync != nil && s.GenSync.JSON && s.GenStream.JSON {
				col.JSON = "✅"
			}
			if s.GenSync != nil && s.GenSync.JSONSchema && s.GenStream.JSONSchema {
				col.JSONSchema = "✅"
			}
			if slices.Contains(s.In, genai.ModalityText) && slices.Contains(s.Out, genai.ModalityImage) {
				col.ImageGen = "✅"
			}
			if slices.Contains(s.In, genai.ModalityText) && slices.Contains(s.Out, genai.ModalityText) && isText {
				col.Chat = "✅"
				col.Streaming = "✅"
			}
			if isDoc {
				col.Doc = "✅"
			}
			if isAsync {
				col.Batch = "✅"
			}
			if s.GenSync != nil {
				// TODO: Keep the best out of all the options. This is "✅⚖️"
				if s.GenSync.Tools == genai.True && s.GenStream.Tools == genai.True {
					col.Tools = "✅"
					if s.GenSync.BiasedTool == genai.False {
						col.Tools += "⚖️"
					}
					if s.GenSync.IndecisiveTool == genai.True {
						col.Tools += " 🤷"
					}
				} else if s.GenSync.Tools == genai.Flaky || s.GenStream.Tools == genai.Flaky && col.Tools == "❌" {
					col.Tools = "💨"
					if s.GenSync.BiasedTool == genai.False {
						col.Tools += "⚖️"
					}
					if s.GenSync.IndecisiveTool == genai.True {
						col.Tools += " 🤷"
					}
				}
			}
			if slices.Contains(s.Out, genai.ModalityAudio) {
				col.AudioGen = "✅"
			}
			// TODO: Add these.
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
			Doc:        "Doc",
			Batch:      "Batch",
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
		if i := countCharsWithEmoji(c.Doc); i > coli.Doc {
			coli.Doc = i
		}
		if i := countCharsWithEmoji(c.Batch); i > coli.Batch {
			coli.Batch = i
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
		fmt.Printf("| %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s |\n",
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
			coli.Doc, c.Doc,
			coli.Batch, c.Batch,
			coli.Seed, c.Seed,
			coli.Tools, c.Tools,
			coli.Caching, c.Caching)
		fmt.Printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n",
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
			strings.Repeat("-", coli.Doc),
			strings.Repeat("-", coli.Batch),
			strings.Repeat("-", coli.Seed),
			strings.Repeat("-", coli.Tools),
			strings.Repeat("-", coli.Caching))
	}
	for _, c := range columns[1:] {
		fmt.Printf("| %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s | %-*s |\n",
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
			coli.Doc-1, c.Doc,
			coli.Batch-1, c.Batch,
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
		case '✅', '💔', '❌', '💨', '🤷': // '⚖️':
			count += 2
		default:
			count += 1
		}
	}
	return count
}
