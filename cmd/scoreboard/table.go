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
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/providers"
	"github.com/maruel/genai/scoreboard"
)

type tableSummaryRow struct {
	Provider string `title:"Provider"`
	Country  string `title:"🌐"`

	tableDataRow
}

func (t *tableSummaryRow) initFromScoreboard(p scoreboard.ProviderScore) {
	sb := p.Scoreboard()
	t.Provider = p.Name()
	if sb.DashboardURL != "" {
		t.Provider = "[" + p.Name() + "](" + sb.DashboardURL + ")"
	}
	t.Country = countryMap[strings.ToLower(sb.Country)]
	if t.Country == "" {
		if t.Country = sb.Country; t.Country == "" {
			t.Country = "N/A"
		}
	}
	if _, isAsync := p.(genai.ProviderGenAsync); isAsync {
		t.Batch = "✅"
	}
	if _, isFiles := p.(genai.ProviderCache); isFiles {
		t.Files = "✅"
	}
	for i := range sb.Scenarios {
		t.initFromScenario(&sb.Scenarios[i])
	}
	addNopes(t)
}

type tableModelRow struct {
	Model string `title:"Model"`

	tableDataRow
}

type tableDataRow struct {
	// Model specific
	Inputs           string `title:"➛In"` // Has to be large enough otherwise the emojis warp on github visualization
	Outputs          string `title:"Out➛"`
	JSON             string `title:"JSON"`
	JSONSchema       string `title:"Schema"`
	Chat             string `title:"Chat"`
	Streaming        string `title:"Stream"`
	Tools            string `title:"Tool"`
	Batch            string `title:"Batch"`
	Seed             string `title:"Seed"`
	Files            string `title:"File"`
	Citations        string `title:"Cite"`
	Thinking         string `title:"Think"`
	Logprobs         string `title:"Probs"`
	ReportRateLimits string `title:"Limits"`
}

func (t *tableDataRow) initFromScenario(s *scoreboard.Scenario) {
	for m := range s.In {
		if s.Thinking {
			t.Thinking = "✅"
		}
		if v, ok := modalityMap[m]; !ok {
			panic("unknown modality: " + m)
		} else if !strings.Contains(t.Inputs, v) {
			t.Inputs += v
		}
	}
	t.Inputs = sortString(t.Inputs)
	for m := range s.Out {
		if v, ok := modalityMap[m]; !ok {
			panic("unknown modality: " + m)
		} else if !strings.Contains(t.Outputs, v) {
			t.Outputs += v
		}
	}
	t.Outputs = sortString(t.Outputs)
	if s.GenSync != nil {
		if s.GenSync.JSON {
			if s.GenStream != nil && s.GenStream.JSON {
				if t.JSON == "" {
					t.JSON = "✅"
				}
			} else {
				t.JSON = "🤪"
			}
		}
		if s.GenSync.JSONSchema {
			if s.GenStream != nil && s.GenStream.JSONSchema {
				if t.JSONSchema == "" {
					t.JSONSchema = "✅"
				}
			} else {
				t.JSONSchema = "🤪"
			}
		}
		if _, hasTextIn := s.In[genai.ModalityText]; hasTextIn {
			if _, hasTextOut := s.Out[genai.ModalityText]; hasTextOut {
				if t.Chat == "" {
					t.Chat = "✅"
				}
				if s.GenSync.BrokenTokenUsage != scoreboard.False && !strings.Contains(t.Chat, "💸") {
					t.Chat += "💸"
				}
				if s.GenSync.BrokenFinishReason && !strings.Contains(t.Chat, "🚩") {
					t.Chat += "🚩"
				}
				if (s.GenSync.NoMaxTokens || s.GenSync.NoStopSequence) && !strings.Contains(t.Chat, "🤪") {
					t.Chat += "🤪"
				}
				t.Chat = sortString(t.Chat)
			}
		}
		// TODO: Keep the best out of all the options. This is "✅"
		if s.GenSync.Tools == scoreboard.True && (s.GenStream != nil && s.GenStream.Tools == scoreboard.True) {
			if t.Tools == "" {
				t.Tools = "✅"
			} else if strings.Contains(t.Tools, "💨") {
				t.Tools = strings.Replace(t.Tools, "💨", "✅", 1)
			}
		} else if s.GenSync.Tools == scoreboard.Flaky && (s.GenStream != nil && s.GenStream.Tools == scoreboard.Flaky) {
			if t.Tools == "" {
				t.Tools = "💨"
			}
		}
		if s.GenSync.BiasedTool != scoreboard.False && !strings.Contains(t.Tools, "🧐") {
			t.Tools += "🧐"
		}
		if s.GenSync.IndecisiveTool == scoreboard.True && !strings.Contains(t.Tools, "💥") {
			t.Tools += "💥"
		}
		t.Tools = sortString(t.Tools)
		if s.GenSync.Citations {
			t.Citations = "✅"
		}
		if s.GenSync.Seed {
			t.Seed = "✅"
		}
		if s.GenSync.TopLogprobs {
			t.Logprobs = "✅"
		}
		if s.GenSync.ReportRateLimits {
			t.ReportRateLimits = "✅"
		}
	}
	if s.GenStream != nil {
		if _, hasTextIn := s.In[genai.ModalityText]; hasTextIn {
			if _, hasTextOut := s.Out[genai.ModalityText]; hasTextOut {
				if t.Streaming == "" {
					t.Streaming = "✅"
				}
				if s.GenStream.BrokenTokenUsage != scoreboard.False && !strings.Contains(t.Streaming, "💸") {
					t.Streaming += "💸"
				}
				if s.GenStream.BrokenFinishReason && !strings.Contains(t.Streaming, "🚩") {
					t.Streaming += "🚩"
				}
				if (s.GenStream.NoMaxTokens || s.GenStream.NoStopSequence) && !strings.Contains(t.Streaming, "🤪") {
					t.Streaming += "🤪"
				}
				t.Streaming = sortString(t.Streaming)
			}
		}
	}
	if s.GenDoc != nil {
		if s.GenDoc.Seed {
			t.Seed = "✅"
		}
		if s.GenDoc.ReportRateLimits {
			t.ReportRateLimits = "✅"
		}
		if s.GenDoc.BrokenTokenUsage != scoreboard.False || s.GenDoc.BrokenFinishReason {
			// TODO.
		}
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
	genai.ModalityText:     "💬", // "📝",
	genai.ModalityImage:    "📸", // "🖼️",
	genai.ModalityAudio:    "🎤",
	genai.ModalityVideo:    "🎥", // "🎞️",
	genai.ModalityDocument: "📄", // "📚",
}

func printTable(provider string) error {
	all := maps.Clone(providers.All)
	if provider == "" {
		return printSummaryTable(all)
	}
	f := all[provider]
	if f == nil {
		return fmt.Errorf("provider %s: not found", provider)
	}
	c, err := f(&genai.OptionsProvider{Model: base.NoModel}, nil)
	if c == nil {
		return fmt.Errorf("provider %s: %w", provider, err)
	}
	return printProviderTable(c)
}

func printSummaryTable(all map[string]func(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error)) error {
	var rows []tableSummaryRow
	seen := map[string]struct{}{}
	for name, f := range all {
		p, err := f(&genai.OptionsProvider{Model: base.NoModel}, nil)
		// The function can return an error and still return a client when no API key was found. It's okay here
		// because we won't use the service provider.
		if p == nil {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: %v\n", name, err)
			continue
		}
		if _, ok := seen[p.Name()]; ok {
			continue
		}
		seen[p.Name()] = struct{}{}
		ps, ok := p.(scoreboard.ProviderScore)
		if !ok {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: doesn't support scoreboard\n", name)
			continue
		}
		row := tableSummaryRow{}
		row.initFromScoreboard(ps)
		rows = append(rows, row)
	}
	slices.SortFunc(rows, func(a, b tableSummaryRow) int {
		return strings.Compare(a.Provider, b.Provider)
	})
	printMarkdownTable(os.Stdout, rows)
	return nil
}

func printProviderTable(p genai.Provider) error {
	var rows []tableModelRow
	ps, ok := p.(scoreboard.ProviderScore)
	if !ok {
		return fmt.Errorf("provider %s: doesn't support scoreboardn", p.Name())
	}
	sb := ps.Scoreboard()
	for i := range sb.Scenarios {
		row := tableModelRow{}
		row.initFromScenario(&sb.Scenarios[i])
		if _, isAsync := p.(genai.ProviderGenAsync); isAsync {
			row.Batch = "✅"
		}
		if _, isFiles := p.(genai.ProviderCache); isFiles {
			row.Files = "✅"
		}
		addNopes(&row)
		for j := range sb.Scenarios[i].Models {
			row.Model = sb.Scenarios[i].Models[j]
			rows = append(rows, row)
		}
	}
	printMarkdownTable(os.Stdout, rows)
	return nil
}

// Magical markdown table generator.

func visitFieldsType(t reflect.Type, fn func(f reflect.StructField)) {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	for i := range t.NumField() {
		if f := t.Field(i); f.Anonymous {
			visitFieldsType(f.Type, fn)
		} else {
			fn(f)
		}
	}
}

func visitFields(v reflect.Value, fn func(v reflect.Value)) {
	t := v.Elem().Type()
	for i := range t.NumField() {
		if fv := v.Elem().Field(i); t.Field(i).Anonymous {
			visitFields(fv.Addr(), fn)
		} else {
			fn(fv.Addr())
		}
	}
}

func getTitles[T any]() []string {
	var titles []string
	visitFieldsType(reflect.TypeOf((*T)(nil)), func(f reflect.StructField) {
		if title := f.Tag.Get("title"); title != "" {
			titles = append(titles, title)
		}
	})
	return titles
}

// addNopes adds "❌" to all empty string fields.
func addNopes(c any) {
	visitFields(reflect.ValueOf(c), func(v reflect.Value) {
		v = v.Elem()
		if l := len(v.String()); l == 0 {
			if v.Kind() == reflect.String {
				v.SetString("❌")
			}
		}
	})
}

func getMaxFieldLengths[T any](rows []T) []int {
	titles := getTitles[T]()
	lengths := make([]int, len(titles))
	for i, t := range titles {
		lengths[i] = visibleWidth(t)
	}
	for i := range rows {
		j := 0
		visitFields(reflect.ValueOf(&rows[i]), func(v reflect.Value) {
			if l := visibleWidth(v.Elem().String()); l > lengths[j] {
				lengths[j] = l
			}
			j++
		})
	}
	return lengths
}

func printMarkdownTable[T any](w io.Writer, rows []T) {
	titles := getTitles[T]()
	lengths := getMaxFieldLengths(rows)
	if len(titles) != len(lengths) {
		panic(fmt.Sprintf("title length mismatch: %d vs %d", len(titles), len(lengths)))
	}

	// Ensure the title row is at least as wide as the field name
	for i, t := range titles {
		if len(t) > lengths[i] {
			lengths[i] = len(t)
		}
	}
	fmt.Fprint(w, "|")
	for i, t := range titles {
		fmt.Fprintf(w, " %s |", extendString(t, lengths[i]))
	}
	fmt.Fprint(w, "\n|")
	for _, l := range lengths {
		fmt.Fprintf(w, " %s |", strings.Repeat("-", l))
	}
	fmt.Fprintln(w)
	for i := range rows {
		fmt.Fprint(w, "|")
		j := 0
		visitFields(reflect.ValueOf(&rows[i]), func(v reflect.Value) {
			fmt.Fprintf(w, " %s |", extendString(v.Elem().String(), lengths[j]))
			j++
		})
		fmt.Fprintf(w, "\n")
	}
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
	case '🏠', '❌', '💬', '✅', '📄', '🎤', '🤪', '🚩', '💨', '💸', '🤷', '📸', '🎥', '💥', '🤐', '🧐', '🌐':
		return 2
	case '🖼', '🎞', '⚖':
		return 0
	default:
		return 1
	}
}

func extendString(s string, l int) string {
	w := visibleWidth(s)
	if w >= l {
		return s
	}
	return s + strings.Repeat(" ", l-w)
}
