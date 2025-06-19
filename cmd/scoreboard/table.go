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

type tableRow struct {
	Provider string `title:"Provider"`
	Country  string `title:"Country"`

	tableRowData
}

func (t *tableRow) initFromScoreboard(p genai.ProviderScoreboard) {
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

type tableRowData struct {
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

func (t *tableRowData) initFromScenario(s *genai.Scenario) {
	for m := range s.In {
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
				if s.GenSync.BrokenTokenUsage && !strings.Contains(t.Chat, "💸") {
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
		if s.GenSync.Tools == genai.True && (s.GenStream != nil && s.GenStream.Tools == genai.True) {
			if t.Tools == "" {
				t.Tools = "✅"
			} else if strings.Contains(t.Tools, "💨") {
				t.Tools = strings.Replace(t.Tools, "💨", "✅", 1)
			}
		} else {
			if t.Tools == "" {
				t.Tools = "💨"
			}
		}
		if s.GenSync.BiasedTool == genai.False && !strings.Contains(t.Tools, "🧐") {
			t.Tools += "🧐"
		}
		if s.GenSync.IndecisiveTool == genai.True && !strings.Contains(t.Tools, "💥") {
			t.Tools += "💥"
		}
		t.Tools = sortString(t.Tools)
		if s.GenSync.Citations {
			t.Citations = "✅"
		}
		if s.GenSync.Thinking {
			t.Thinking = "✅"
		}
		if s.GenSync.Seed {
			t.Seed = "✅"
		}
	}
	if s.GenStream != nil {
		if _, hasTextIn := s.In[genai.ModalityText]; hasTextIn {
			if _, hasTextOut := s.Out[genai.ModalityText]; hasTextOut {
				if t.Streaming == "" {
					t.Streaming = "✅"
				}
				if s.GenStream.BrokenTokenUsage && !strings.Contains(t.Streaming, "💸") {
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
		if s.GenDoc.BrokenTokenUsage || s.GenDoc.BrokenFinishReason {
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
	var columns []tableRow
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
		col := tableRow{}
		col.initFromScoreboard(ps)
		columns = append(columns, col)
	}
	slices.SortFunc(columns, func(a, b tableRow) int {
		return strings.Compare(a.Provider, b.Provider)
	})
	printMarkdownTable(os.Stdout, columns)
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

func getMaxFieldLengths[T any](cols []T) []int {
	titles := getTitles[T]()
	lengths := make([]int, len(titles))
	for i, t := range titles {
		lengths[i] = visibleWidth(t)
	}
	for i := range cols {
		j := 0
		visitFields(reflect.ValueOf(&cols[i]), func(v reflect.Value) {
			if l := visibleWidth(v.Elem().String()); l > lengths[j] {
				lengths[j] = l
			}
			j++
		})
	}
	return lengths
}

func printMarkdownTable[T any](w io.Writer, cols []T) {
	titles := getTitles[T]()
	lengths := getMaxFieldLengths(cols)
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
	for i := range cols {
		fmt.Fprint(w, "|")
		j := 0
		visitFields(reflect.ValueOf(&cols[i]), func(v reflect.Value) {
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
	if w >= l {
		return s
	}
	return s + strings.Repeat(" ", l-w)
}
