// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
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
	"github.com/maruel/genai/scoreboard"
)

const legend = `<details>
<summary>‼️ Click here for the legend of columns and symbols</summary>

- 🏠: Runs locally.
- Sync:   Runs synchronously, the reply is only returned once completely generated
- Stream: Streams the reply as it is generated. Occasionally less features are supported in this mode
- 🧠: Has chain-of-thought thinking process
    - Both redacted (Anthropic, Gemini, OpenAI) and explicit (Deepseek R1, Qwen3, etc)
    - Many models can be used in both mode. In this case they will have two rows, one with thinking and one
      without. It is frequent that certain functionalities are limited in thinking mode, like tool calling.
- ✅: Implemented and works great
- ❌: Not supported by genai. The provider may support it, but genai does not (yet). Please send a PR to add
  it!
- 💬: Text
- 📄: PDF: process a PDF as input, possibly with OCR
- 📸: Image: process an image as input; most providers support PNG, JPG, WEBP and non-animated GIF, or generate images
- 🎤: Audio: process an audio file (e.g. MP3, WAV, Flac, Opus) as input, or generate audio
- 🎥: Video: process a video (e.g. MP4) as input, or generate a video (e.g. Veo 3)
- 💨: Feature is flaky (Tool calling) or inconsistent (Usage is not always reported)
- 🪨: Tool calling can be forced; aka you can force the model to call a tool. This is great.
- 🧐: Tool calling is **not** biased towards the first value in an enum. This is good. If the provider doesn't
	have this, be mindful of the order of the values presented in the prompt!
- 💥: Tool calling is indecisive. When unsure about an answer, it'll call both options. This is good.
- 🌐: Country where the company is located
- JSON and Schema: ability to output JSON in free form, or with a forced schema specified as a Go struct
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef); best is ✅🪨🧐💥
- Batch: Process asynchronously batches during off peak hours at a discounts
- Text: Text features
    - '🌱': Seed option for deterministic output
    - '📏': MaxTokens option to cap the amount of returned tokens
    - '🛑': Stop sequence to stop generation when a token is generated
- File: Upload and store large files via a separate API
- Cite: Citation generation from a provided document, specially useful for RAG
- Probs: Return logprobs to analyse each token probabilities
- Limits: Returns the rate limits, including the remaining quota
</details>
`

type tableSummaryRow struct {
	Provider string `title:"Provider"`
	Country  string `title:"🌐"`

	tableDataRow
}

func (t *tableSummaryRow) initFromScoreboard(p genai.Provider) {
	sb := p.Scoreboard()
	t.Provider = p.Name()
	// We could link to the dashboard (sb.DashboardURL) but assume we want to link to the generated doc.
	base := "docs/"
	if sb.DashboardURL != "" {
		t.Provider = "[" + p.Name() + "](" + base + p.Name() + ".md" + ")"
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
		// Assume GenSync has the best values.
		f := sb.Scenarios[i].GenSync
		if f == nil {
			continue
		}
		t.initFromScenario(&sb.Scenarios[i], f)
		if sb.Scenarios[i].GenStream != nil && !strings.Contains(t.Mode, "Stream") {
			if t.Mode != "" {
				t.Mode += ", "
			}
			t.Mode += "Stream"
		}
		// Do a small hack to put the brain at the end.
		if strings.Contains(t.Mode, "🧠") {
			t.Mode = strings.ReplaceAll(t.Mode, "🧠", "") + "🧠"
		}
	}
	fillEmptyFields(t, "❌")
}

type tableModelRow struct {
	Model string `title:"Model"`

	tableDataRow
}

type tableDataRow struct {
	// Model specific
	Mode         string `title:"Mode"`
	Inputs       string `title:"➛In"` // Has to be large enough otherwise the emojis warp on github visualization
	Outputs      string `title:"Out➛"`
	JSON         string `title:"JSON"`
	JSONSchema   string `title:"Schema"`
	Tools        string `title:"Tool"`
	Batch        string `title:"Batch"`
	Files        string `title:"File"`
	Citations    string `title:"Cite"`
	TextFeatures string `title:"Text"`
	// Reporting.
	Logprobs   string `title:"Probs"`
	RateLimits string `title:"Limits"`
	Usage      string `title:"Usage"`
	Finish     string `title:"Finish"`
}

func (t *tableDataRow) initFromScenario(s *scoreboard.Scenario, f *scoreboard.Functionality) {
	if s.GenSync == f && !strings.Contains(t.Mode, "Sync") {
		if t.Mode != "" {
			t.Mode += ", "
		}
		t.Mode += "Sync"
	}
	if s.GenStream == f && !strings.Contains(t.Mode, "Stream") {
		if t.Mode != "" {
			t.Mode += ", "
		}
		t.Mode += "Stream"
	}
	if s.Thinking && !strings.Contains(t.Mode, "🧠") {
		t.Mode += "🧠"
	}
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
	if f.JSON {
		t.JSON = "✅"
	}
	if f.JSONSchema {
		t.JSONSchema = "✅"
	}
	if f.Tools == scoreboard.True && !strings.Contains(t.Tools, "✅") {
		t.Tools = "✅"
	} else if s.GenSync.Tools == scoreboard.Flaky && !strings.Contains(t.Tools, "✅") {
		t.Tools = "💨"
	}
	if f.ToolCallRequired && !strings.Contains(t.Tools, "🪨") {
		t.Tools += "🪨"
	}
	if f.ToolsBiased != scoreboard.False && !strings.Contains(t.Tools, "🧐") {
		t.Tools += "🧐"
	}
	if f.ToolsIndecisive == scoreboard.True && !strings.Contains(t.Tools, "💥") {
		t.Tools += "💥"
	}
	if f.Citations {
		t.Citations = "✅"
	}

	if f.Seed && !strings.Contains(t.TextFeatures, "🌱") {
		t.TextFeatures += "🌱"
	}
	if f.MaxTokens && !strings.Contains(t.TextFeatures, "📏") {
		t.TextFeatures += "📏"
	}
	if f.StopSequence && !strings.Contains(t.TextFeatures, "🛑") {
		t.TextFeatures += "🛑"
	}

	// Reporting.
	if f.TopLogprobs {
		t.Logprobs = "✅"
	}
	if f.ReportRateLimits {
		t.RateLimits = "✅"
	}
	if f.ReportTokenUsage == scoreboard.True {
		t.Usage = "✅"
	} else if s.GenSync.ReportTokenUsage == scoreboard.Flaky && !strings.Contains(t.Usage, "✅") {
		t.Usage = "💨"
	}
	if f.ReportFinishReason == scoreboard.True {
		t.Finish = "✅"
	} else if s.GenSync.ReportFinishReason == scoreboard.Flaky && !strings.Contains(t.Finish, "✅") {
		t.Finish = "💨"
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

func printTable(ctx context.Context, provider string) error {
	all := maps.Clone(providers.All)
	if provider == "" {
		return printSummaryTable(ctx, all)
	}
	f := all[provider]
	if f == nil {
		return fmt.Errorf("provider %s: not found", provider)
	}
	c, err := f(ctx, &genai.ProviderOptions{Model: genai.ModelNone}, nil)
	if c == nil {
		return fmt.Errorf("provider %s: %w", provider, err)
	}
	return printProviderTable(c)
}

func printSummaryTable(ctx context.Context, all map[string]func(ctx context.Context, opts *genai.ProviderOptions, wrapper func(http.RoundTripper) http.RoundTripper) (genai.Provider, error)) error {
	var rows []tableSummaryRow
	seen := map[string]struct{}{}
	for name, f := range all {
		opts := &genai.ProviderOptions{Model: genai.ModelNone}
		if name == "openaicompatible" {
			// Make sure the remote it set for this one.
			opts.Remote = "http://localhost:0"
		}
		p, err := f(ctx, opts, nil)
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
		row := tableSummaryRow{}
		row.initFromScoreboard(p)
		rows = append(rows, row)
	}
	slices.SortFunc(rows, func(a, b tableSummaryRow) int {
		return strings.Compare(a.Provider, b.Provider)
	})
	printMarkdownTable(os.Stdout, rows)
	_, _ = os.Stdout.WriteString(legend)
	return nil
}

func printProviderTable(p genai.Provider) error {
	var rows []tableModelRow
	sb := p.Scoreboard()
	for _, sc := range sb.Scenarios {
		var tmpRows []tableModelRow
		for _, f := range []*scoreboard.Functionality{sc.GenSync, sc.GenStream} {
			if f == nil {
				continue
			}
			row := tableModelRow{}
			row.initFromScenario(&sc, f)
			if _, isAsync := p.(genai.ProviderGenAsync); isAsync {
				row.Batch = "✅"
			}
			if _, isFiles := p.(genai.ProviderCache); isFiles {
				row.Files = "✅"
			}
			fillEmptyFields(&row, "❌")
			tmpRows = append(tmpRows, row)
		}
		if len(tmpRows) == 0 {
			row := tableModelRow{}
			fillEmptyFields(&row, "?")
			tmpRows = append(tmpRows, row)
		}
		for _, m := range sc.Models {
			for i := range tmpRows {
				tmpRows[i].Model = m
				rows = append(rows, tmpRows[i])
			}
		}
	}
	printMarkdownTable(os.Stdout, rows)
	_, _ = os.Stdout.WriteString(legend)
	return nil
}

// Magical markdown table generator.

func visitFieldsType(t reflect.Type, fn func(f reflect.StructField)) {
	if t.Kind() == reflect.Pointer {
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

// fillEmptyFields sets all empty string fields to def.
func fillEmptyFields(c any, def string) {
	visitFields(reflect.ValueOf(c), func(v reflect.Value) {
		v = v.Elem()
		if l := len(v.String()); l == 0 {
			if v.Kind() == reflect.String {
				v.SetString(def)
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
	case '🏠', '❌', '💬', '✅', '📄', '🎤', '🤪', '🚩', '💨', '💸', '🤷', '📸', '🎥', '💥', '🤐', '🧐', '🌐', '🤏', '📡', '🌱', '🪨':
		return 2
	case '🖼', '🎞', '⚖': // '🕰️'
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
