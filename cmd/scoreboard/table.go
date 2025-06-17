// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers"
	"github.com/maruel/genai/providers/openaicompatible"
)

type column struct {
	// General
	Provider string `title:"Provider"`
	Country  string `title:"Country"`

	// Model specific
	Inputs     string `title:"➛Inputs"` // Has to be large enough otherwise the emojis warp on github visualization
	Outputs    string `title:"Outputs➛"`
	JSON       string `title:"JSON➛"`
	JSONSchema string `title:"JSON+Schema➛"`
	Chat       string `title:"Chat"`
	Streaming  string `title:"Streaming"`
	Tools      string `title:"Tools"`
	Batch      string `title:"Batch"`
	Seed       string `title:"Seed"`
	Files      string `title:"Files"`
	Citations  string `title:"Citations"`
	Thinking   string `title:"Thinking"`
}

func (c *column) initFromScoreboard(p genai.ProviderScoreboard) {
	sb := p.Scoreboard()
	// resetToNope(c)
	c.Provider = p.Name()
	if sb.DashboardURL != "" {
		c.Provider = "[" + p.Name() + "](" + sb.DashboardURL + ")"
	}
	country := countryMap[strings.ToLower(sb.Country)]
	if country == "" {
		country = sb.Country
	}
	c.Country = country
	_, isAsync := p.(genai.ProviderGenAsync)
	_, isFiles := p.(genai.ProviderCache)
	if isAsync {
		c.Batch = "✅"
	}
	if isFiles {
		c.Files = "✅"
	}
	for i := range sb.Scenarios {
		c.initFromScenario(&sb.Scenarios[i])
	}
	addNopes(c)
}

func (c *column) initFromScenario(s *genai.Scenario) {
	for m := range s.In {
		if v, ok := modalityMap[m]; !ok {
			panic("unknown modality: " + m)
		} else if !strings.Contains(c.Inputs, v) {
			c.Inputs += v
		}
	}
	c.Inputs = sortString(c.Inputs)
	for m := range s.Out {
		if v, ok := modalityMap[m]; !ok {
			panic("unknown modality: " + m)
		} else if !strings.Contains(c.Outputs, v) {
			c.Outputs += v
		}
	}
	c.Outputs = sortString(c.Outputs)
	if s.GenSync != nil {
		if s.GenSync.JSON {
			if s.GenStream != nil && s.GenStream.JSON {
				if c.JSON == "" {
					c.JSON = "✅"
				}
			} else {
				c.JSON = "🤪"
			}
		}
		if s.GenSync.JSONSchema {
			if s.GenStream != nil && s.GenStream.JSONSchema {
				if c.JSONSchema == "" {
					c.JSONSchema = "✅"
				}
			} else {
				c.JSONSchema = "🤪"
			}
		}
		if _, hasTextIn := s.In[genai.ModalityText]; hasTextIn {
			if _, hasTextOut := s.Out[genai.ModalityText]; hasTextOut {
				if c.Chat == "" {
					c.Chat = "✅"
				}
				if s.GenSync.BrokenTokenUsage && !strings.Contains(c.Chat, "💸") {
					c.Chat += "💸"
				}
				if s.GenSync.BrokenFinishReason && !strings.Contains(c.Chat, "🚩") {
					c.Chat += "🚩"
				}
				if (s.GenSync.NoMaxTokens || s.GenSync.NoStopSequence) && !strings.Contains(c.Chat, "🤪") {
					c.Chat += "🤪"
				}
				c.Chat = sortString(c.Chat)
			}
		}
		// TODO: Keep the best out of all the options. This is "✅"
		if s.GenSync.Tools == genai.True && (s.GenStream != nil && s.GenStream.Tools == genai.True) {
			if c.Tools == "" {
				c.Tools = "✅"
			} else if strings.Contains(c.Tools, "💨") {
				c.Tools = strings.Replace(c.Tools, "💨", "✅", 1)
			}
		} else {
			if c.Tools == "" {
				c.Tools = "💨"
			}
		}
		if s.GenSync.BiasedTool == genai.False && !strings.Contains(c.Tools, "🧐") {
			c.Tools += "🧐"
		}
		if s.GenSync.IndecisiveTool == genai.True && !strings.Contains(c.Tools, "💥") {
			c.Tools += "💥"
		}
		c.Tools = sortString(c.Tools)
		if s.GenSync.Citations {
			c.Citations = "✅"
		}
		if s.GenSync.Thinking {
			c.Thinking = "✅"
		}
		if s.GenSync.Seed {
			c.Seed = "✅"
		}
	}
	if s.GenStream != nil {
		if _, hasTextIn := s.In[genai.ModalityText]; hasTextIn {
			if _, hasTextOut := s.Out[genai.ModalityText]; hasTextOut {
				if c.Streaming == "" {
					c.Streaming = "✅"
				}
				if s.GenStream.BrokenTokenUsage && !strings.Contains(c.Streaming, "💸") {
					c.Streaming += "💸"
				}
				if s.GenStream.BrokenFinishReason && !strings.Contains(c.Streaming, "🚩") {
					c.Streaming += "🚩"
				}
				if (s.GenStream.NoMaxTokens || s.GenStream.NoStopSequence) && !strings.Contains(c.Streaming, "🤪") {
					c.Streaming += "🤪"
				}
				c.Streaming = sortString(c.Streaming)
			}
		}
	}
	if s.GenDoc != nil {
		if s.GenDoc.Seed {
			c.Seed = "✅"
		}
		if s.GenDoc.BrokenTokenUsage || s.GenDoc.BrokenFinishReason {
			// TODO.
		}
	}
}

// Magical markdown table generator.

func getTitles[T any]() []string {
	var titles []string
	t := reflect.TypeOf((*T)(nil)).Elem()
	for i := range t.NumField() {
		if title := t.Field(i).Tag.Get("title"); title != "" {
			titles = append(titles, title)
		}
	}
	return titles
}

// addNopes adds "❌" to all empty string fields.
func addNopes[T any](c *T) {
	val := reflect.ValueOf(c).Elem()
	for j := range val.NumField() {
		if l := len(val.Field(j).String()); l == 0 {
			val.Field(j).SetString("❌")
		}
	}
}

func getMaxFieldLengths[T any](cols []T) []int {
	fields := reflect.TypeOf((*T)(nil)).Elem()
	lengths := make([]int, fields.NumField())
	for i := range cols {
		val := reflect.ValueOf(cols[i])
		for j := range lengths {
			if l := visibleWidth(val.Field(j).String()); l > lengths[j] {
				lengths[j] = l
			}
		}
	}
	return lengths
}

func printMarkdownTable[T any](w io.Writer, cols []T) {
	titles := getTitles[T]()
	lengths := getMaxFieldLengths(cols)
	// Ensure the title row is at least as wide as the field name
	for i, t := range titles {
		if len(t) > lengths[i] {
			lengths[i] = len(t)
		}
	}
	fmt.Fprint(w, "| ")
	for i, t := range titles {
		fmt.Fprintf(w, "%s |", extendString(t, lengths[i]))
		if i != len(titles)-1 {
			fmt.Fprint(w, " ")
		}
	}
	fmt.Fprintln(w)
	fmt.Fprint(w, "| ")
	for i, l := range lengths {
		fmt.Fprintf(w, "%s |", strings.Repeat("-", l))
		if i != len(lengths)-1 {
			fmt.Fprint(w, " ")
		}
	}
	fmt.Fprintln(w)
	for _, c := range cols {
		fmt.Fprint(w, "| ")
		v := reflect.ValueOf(c)
		for i := range lengths {
			fmt.Fprintf(w, "%s |", extendString(v.Field(i).String(), lengths[i]))
			if i != len(lengths)-1 {
				fmt.Fprint(w, " ")
			}
		}
		fmt.Fprintln(w)
	}
}

var countryMap = map[string]string{
	"ca":    "🇨🇦",
	"cn":    "🇨🇳",
	"de":    "🇩🇪",
	"fr":    "🇫🇷",
	"us":    "🇺🇸",
	"local": "🏠",
}

var modalityMap = map[genai.Modality]string{
	genai.ModalityText:  "💬", // "📝",
	genai.ModalityImage: "📸", // "🖼️",
	genai.ModalityAudio: "🎤",
	genai.ModalityVideo: "🎥", // "🎞️",
	genai.ModalityPDF:   "📄", // "📚",
}

func printTable() error {
	all := maps.Clone(providers.All)
	all["openaicompatible"] = func(model string, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error) {
		return openaicompatible.New("http://localhost:8080/v1", nil, model, wrapper)
	}
	var columns []column
	for name, f := range all {
		p, err := f("", nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: %v\n", name, err)
			continue
		}
		ps, ok := p.(genai.ProviderScoreboard)
		if !ok {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: doesn't support scoreboard\n", name)
			continue
		}
		col := column{}
		col.initFromScoreboard(ps)
		columns = append(columns, col)
	}
	slices.SortFunc(columns, func(a, b column) int {
		return strings.Compare(a.Provider, b.Provider)
	})

	printMarkdownTable(os.Stdout, columns)
	return nil
}

// sortString sorts the characters in a string.
func sortString(s string) string {
	r := []rune(s)
	slices.SortFunc(r, func(i, j rune) int {
		if i == j {
			return 0
		}
		if i < j {
			return -1
		}
		return 1
	})
	return string(r)
}

func visibleWidth(s string) int {
	width := 0
	for _, r := range s {
		width += runeWidth(r)
	}
	return width
}

func runeWidth(r rune) int {
	switch r {
	case '🏠', '❌', '💬', '✅', '📄', '🎤', '🤪', '🚩', '💨', '💸', '🤷', '📸', '🎥', '💥', '🤐', '🧐':
		return 2
	case '🖼', '🎞', '⚖':
		return 0
	default:
		return 1
	}
}

func extendString(s string, l int) string {
	w := visibleWidth(s)
	return s + strings.Repeat(" ", l-w)
}
