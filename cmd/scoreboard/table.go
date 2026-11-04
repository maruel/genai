// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command scoreboard provides a table view of models.

package main

import (
	"context"
	"fmt"
	"io"
	"maps"
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
- 🎯: Decision: typed answers returned by SystemOne
- 🧬: Embedding: text vectors with qualified input order and semantic retrieval
- Realtime: Bidirectional audio and text session through ConnectRealtime
- WS: Measured over the WebSocket wire instead of HTTP.
- 📄: PDF: process a PDF as input, possibly with OCR
- 📸: Image: process an image as input; most providers support PNG, JPG, WEBP and non-animated GIF, or generate images
- 🎤: Audio: process an audio file (e.g. MP3, WAV, Flac, Opus) as input, or generate audio
- 🎥: Video: process a video (e.g. MP4) as input, or generate a video (e.g. Veo 3)
- 💨: Feature is flaky (Tool calling) or inconsistent (Usage or Finish reason is not always reported)
- 🌐: Country where the company is located
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef); best is ✅🪨🕸️🔀
		- 🪨: Tool calling can be forced; aka you can force the model to call a tool. This is great.
		- 🕸️: Web search
		- 🔀: Supports messages and tool calls out of order
- JSON: ability to output JSON in free form, or with a forced schema specified as a Go struct
    - ✅: Supports both free form and with a schema
    - ☁️ :Supports only free form
		- 📐: Supports only a schema
		- []: Also supports an array at the root of the schema, most providers require an object
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

/*
- 🧐: Tool calling is **not** biased towards the first value in an enum. This is good. If the provider doesn't
	have this, be mindful of the order of the values presented in the prompt!
- 💥: Tool calling is indecisive. When unsure about an answer, it'll call both options. This is good.
- Tool: Tool calling, using [genai.ToolDef](https://pkg.go.dev/github.com/maruel/genai#ToolDef); best is ✅🪨🧐💥
*/

type tableSummaryRow struct {
	Provider string `title:"Provider"`
	Country  string `title:"🌐"`

	tableDataRow
}

func (t *tableSummaryRow) initFromScoreboard(p genai.Provider) {
	sb := p.Scoreboard()
	t.Provider = p.Name()
	// Link to the generated doc page. openaicompatible is excluded from doc
	// generation (see regen_docs.go) so skip it here too.
	if p.Name() != "openaicompatible" {
		t.Provider = "[" + p.Name() + "](docs/" + p.Name() + ".md)"
	}
	t.Country = countryMap[strings.ToLower(sb.Country)]
	if t.Country == "" {
		if t.Country = sb.Country; t.Country == "" {
			t.Country = "N/A"
		}
	}
	if p.Capabilities().GenAsync {
		t.Batch = "✅"
	}
	if p.Capabilities().Caching {
		t.Files = "✅"
	}
	// Merge all variant scoreboards when the provider has multiple backends.
	scoreboards := []scoreboard.Score{sb}
	if pv, ok := p.(genai.ProviderScoreboardVariants); ok {
		variants := pv.ScoreboardVariants()
		scoreboards = make([]scoreboard.Score, len(variants))
		for i := range variants {
			scoreboards[i] = variants[i].Score
		}
	}
	for i := range scoreboards {
		s := &scoreboards[i]
		for j := range s.Embed {
			if r := &s.Embed[j]; r.Embed != nil {
				t.initFromEmbed(r)
			}
		}
		for j := range s.SystemOne {
			if r := &s.SystemOne[j]; r.SystemOne != nil {
				t.initFromSystemOne(r)
			}
		}
		for j := range s.Realtime {
			if r := &s.Realtime[j]; r.Realtime != nil {
				t.initFromRealtime(r)
			}
		}
		for j := range s.Chat {
			r := &s.Chat[j]
			// Assume GenSync has the best values.
			if r.GenSync == nil {
				continue
			}
			t.initFromChat(r, r.GenSync)
			if r.GenStream != nil && !strings.Contains(t.Mode, "Stream") {
				if t.Mode != "" {
					t.Mode += ", "
				}
				t.Mode += "Stream"
			}
		}
	}
	t.finishMode()
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
	Tools        string `title:"Tool"`
	JSON         string `title:"JSON"`
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

// finishMode puts the brain where thinking happens, after the last generation
// mode, instead of wherever a reasoning scenario happened to be merged.
func (t *tableDataRow) finishMode() {
	if !strings.Contains(t.Mode, "🧠") {
		return
	}
	modes := slices.DeleteFunc(strings.Split(strings.ReplaceAll(t.Mode, "🧠", ""), ", "), func(m string) bool { return m == "" })
	if len(modes) == 0 {
		t.Mode = "🧠"
		return
	}
	i := len(modes) - 1
	for j := i; j >= 0; j-- {
		if modes[j] == "Sync" || modes[j] == "Stream" {
			i = j
			break
		}
	}
	modes[i] += "🧠"
	t.Mode = strings.Join(modes, ", ")
}

// sectionRow is what a table row needs from a row of any section.
type sectionRow struct {
	scoreboard.Row
	Out       map[scoreboard.Modality]scoreboard.ModalCapability
	Reason    bool
	GenSync   *scoreboard.Functionality
	GenStream *scoreboard.Functionality
}

func (t *tableDataRow) initFromChat(s *scoreboard.Chat, f *scoreboard.Functionality) {
	t.initFromScenario(&sectionRow{
		Row: s.Row, Out: s.Out, Reason: s.Reason, GenSync: s.GenSync, GenStream: s.GenStream,
	}, f)
}

func (t *tableDataRow) initFromEmbed(s *scoreboard.Embed) {
	f := s.Embed
	t.initFromScenario(&sectionRow{Row: s.Row, Out: s.Outputs()}, &scoreboard.Functionality{ReportTokenUsage: f.ReportTokenUsage})
	if !strings.Contains(t.Mode, "Embed") {
		if t.Mode != "" {
			t.Mode += ", "
		}
		t.Mode += "Embed"
	}
}

func (t *tableDataRow) initFromSystemOne(s *scoreboard.SystemOne) {
	f := s.SystemOne
	t.initFromScenario(&sectionRow{Row: s.Row, Out: s.Out}, &scoreboard.Functionality{ReportTokenUsage: f.ReportTokenUsage})
	if !strings.Contains(t.Mode, "SystemOne") {
		if t.Mode != "" {
			t.Mode += ", "
		}
		t.Mode += "SystemOne"
	}
}

func (t *tableDataRow) initFromRealtime(s *scoreboard.Realtime) {
	if !strings.Contains(t.Mode, "Realtime") {
		if t.Mode != "" {
			t.Mode += ", "
		}
		t.Mode += "Realtime"
	}
	t.initFromScenario(&sectionRow{Row: s.Row, Out: s.Out, Reason: s.Reason}, &scoreboard.Functionality{
		Tools:            s.Realtime.Tools,
		ReportRateLimits: s.Realtime.ReportRateLimits,
		ReportTokenUsage: s.Realtime.ReportTokenUsage,
	})
}

func (t *tableDataRow) initFromScenario(s *sectionRow, f *scoreboard.Functionality) {
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
	if s.Reason && !strings.Contains(t.Mode, "🧠") {
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
	switch {
	case f.JSON && f.JSONSchema.Object != scoreboard.False:
		t.JSON = "✅"
	case f.JSON && !strings.Contains(t.JSON, "✅"):
		t.JSON = "☁️"
	case f.JSONSchema.Object != scoreboard.False && !strings.Contains(t.JSON, "✅"):
		t.JSON = "📐"
	}
	if f.JSONSchema.Array != scoreboard.False {
		t.JSON += "[]"
	}
	switch {
	case f.Tools == scoreboard.True && !strings.Contains(t.Tools, "✅"):
		t.Tools = "✅"
	case f.Tools == scoreboard.Flaky && !strings.Contains(t.Tools, "✅"):
		t.Tools = "💨"
	}
	if f.ToolCallRequired && !strings.Contains(t.Tools, "🪨") {
		t.Tools += "🪨"
	}
	if f.WebSearch && !strings.Contains(t.Tools, "🕸️") {
		t.Tools += "🕸️"
	}
	if f.OutOfOrder != nil && *f.OutOfOrder == scoreboard.True && !strings.Contains(t.Tools, "🔀") {
		t.Tools += "🔀"
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
	} else if f.ReportTokenUsage == scoreboard.Flaky && !strings.Contains(t.Usage, "✅") {
		t.Usage = "💨"
	}
	if f.ReportFinishReason == scoreboard.True {
		t.Finish = "✅"
	} else if f.ReportFinishReason == scoreboard.Flaky && !strings.Contains(t.Finish, "✅") {
		t.Finish = "💨"
	}
}

var countryMap = map[string]string{
	"at":    "🇦🇹",
	"ca":    "🇨🇦",
	"cn":    "🇨🇳",
	"de":    "🇩🇪",
	"fr":    "🇫🇷",
	"us":    "🇺🇸",
	"local": "🏠",
}

var modalityMap = map[genai.Modality]string{
	genai.ModalityDecision:  "🎯",
	genai.ModalityEmbedding: "🧬",
	genai.ModalityText:      "💬", // "📝",
	genai.ModalityImage:     "📸", // "🖼️",
	genai.ModalityAudio:     "🎤",
	genai.ModalityVideo:     "🎥", // "🎞️",
	genai.ModalityDocument:  "📄", // "📚",
}

func printTable(ctx context.Context, w io.Writer, provider string) error {
	all := maps.Clone(providers.All)
	if provider == "" {
		return printSummaryTable(ctx, w, all)
	}
	cfg := all[provider]
	if cfg.Factory == nil {
		return fmt.Errorf("provider %s: not found", provider)
	}
	c, err := cfg.Factory(ctx)
	if c == nil {
		return fmt.Errorf("provider %s: %w", provider, err)
	}
	return printProviderTable(c, w)
}

// printSummaryTable prints a summary table for all the providers.
func printSummaryTable(ctx context.Context, w io.Writer, all map[string]providers.Config) error {
	var rows []tableSummaryRow
	seen := map[string]struct{}{}
	for name, cfg := range all {
		if cfg.Alias != "" {
			continue
		}
		var opts []genai.ProviderOption
		if name == "openaicompatible" {
			// Make sure the remote it set for this one.
			opts = append(opts, genai.ProviderOptionRemote("http://localhost:0"))
		}
		p, err := cfg.Factory(ctx, opts...)
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
	printMarkdownTable(w, rows)
	_, _ = io.WriteString(w, legend)
	return nil
}

// printProviderTable prints a table of all the models for a specific provider.
func printProviderTable(p genai.Provider, w io.Writer) error {
	if pv, ok := p.(genai.ProviderScoreboardVariants); ok {
		variants := pv.ScoreboardVariants()
		if len(variants) > 1 {
			return printProviderVariantTables(p, w, variants)
		}
	}
	sb := p.Scoreboard()
	printScoreboardTable(p, w, &sb)
	return nil
}

// printProviderVariantTables prints a separate table per scoreboard variant.
func printProviderVariantTables(p genai.Provider, w io.Writer, variants []genai.ScoreboardVariant) error {
	for i := range variants {
		v := &variants[i]
		if i > 0 {
			_, _ = io.WriteString(w, "\n")
		}
		_, _ = fmt.Fprintf(w, "## %s\n\n", v.Name)
		printScoreboardTable(p, w, &v.Score)
	}
	return nil
}

// tableRows builds the rows of one scoreboard: one per measured suite, then one queue
// entry per model no suite measured.
func tableRows(sb *scoreboard.Score, caps genai.ProviderCapabilities) []tableModelRow {
	rows := make([]tableModelRow, 0, len(sb.Chat)+len(sb.Embed)+len(sb.SystemOne)+len(sb.Realtime))
	for i := range sb.Chat {
		r := &sb.Chat[i]
		var tmpRows []tableModelRow
		for _, f := range []*scoreboard.Functionality{r.GenSync, r.GenStream} {
			if f == nil {
				continue
			}
			row := tableModelRow{}
			row.initFromChat(r, f)
			if caps.GenAsync {
				row.Batch = "✅"
			}
			if caps.Caching {
				row.Files = "✅"
			}
			fillEmptyFields(&row, "❌")
			tmpRows = append(tmpRows, row)
		}
		if len(tmpRows) == 0 && r.Refused != "" {
			// The table lists what a model does. A refusal is a result the scoreboard
			// keeps, in the provider's own words.
			continue
		}
		markWire(tmpRows, r.Base())
		rows = appendModels(rows, r.Base(), tmpRows)
	}
	for i := range sb.Embed {
		r := &sb.Embed[i]
		var tmpRows []tableModelRow
		if r.Embed != nil {
			row := tableModelRow{}
			row.initFromEmbed(r)
			fillEmptyFields(&row, "❌")
			tmpRows = append(tmpRows, row)
		}
		if len(tmpRows) == 0 && r.Refused != "" {
			// The table lists what a model does. A refusal is a result the scoreboard
			// keeps, in the provider's own words.
			continue
		}
		markWire(tmpRows, r.Base())
		rows = appendModels(rows, r.Base(), tmpRows)
	}
	for i := range sb.SystemOne {
		r := &sb.SystemOne[i]
		var tmpRows []tableModelRow
		if r.SystemOne != nil {
			row := tableModelRow{}
			row.initFromSystemOne(r)
			fillEmptyFields(&row, "❌")
			tmpRows = append(tmpRows, row)
		}
		if len(tmpRows) == 0 && r.Refused != "" {
			// The table lists what a model does. A refusal is a result the scoreboard
			// keeps, in the provider's own words.
			continue
		}
		markWire(tmpRows, r.Base())
		rows = appendModels(rows, r.Base(), tmpRows)
	}
	for i := range sb.Realtime {
		r := &sb.Realtime[i]
		var tmpRows []tableModelRow
		if r.Realtime != nil {
			row := tableModelRow{}
			row.initFromRealtime(r)
			fillEmptyFields(&row, "❌")
			tmpRows = append(tmpRows, row)
		}
		if len(tmpRows) == 0 && r.Refused != "" {
			// The table lists what a model does. A refusal is a result the scoreboard
			// keeps, in the provider's own words.
			continue
		}
		markWire(tmpRows, r.Base())
		rows = appendModels(rows, r.Base(), tmpRows)
	}
	// A model no section measured is one queue entry, not one per section, and a
	// model measured somewhere is not a queue entry at all.
	measured := map[string]struct{}{}
	for i := range rows {
		if rows[i].Mode != "?" {
			measured[rows[i].Model] = struct{}{}
		}
	}
	queued := map[string]struct{}{}
	rows = slices.DeleteFunc(rows, func(r tableModelRow) bool {
		if r.Mode != "?" {
			return false
		}
		if _, ok := measured[r.Model]; ok {
			return true
		}
		if _, ok := queued[r.Model]; ok {
			return true
		}
		queued[r.Model] = struct{}{}
		return false
	})
	// A measured model comes before an unmeasured one, then the preferred models,
	// then by name, so a section change does not reshuffle the table.
	slices.SortStableFunc(rows, func(a, b tableModelRow) int {
		if c := unmeasuredRank(&a) - unmeasuredRank(&b); c != 0 {
			return c
		}
		if c := medalRank(a.Model) - medalRank(b.Model); c != 0 {
			return c
		}
		return strings.Compare(strings.TrimRight(a.Model, "🥇🥈🥉"), strings.TrimRight(b.Model, "🥇🥈🥉"))
	})
	return rows
}

// printScoreboardTable prints a single scoreboard table.
func printScoreboardTable(p genai.Provider, w io.Writer, sb *scoreboard.Score) {
	rows := tableRows(sb, p.Capabilities())
	printMarkdownTable(w, rows)
	_, _ = io.WriteString(w, legend)
	if len(sb.Warnings) > 0 {
		_, _ = io.WriteString(w, "\n## Warnings\n\n")
		for _, wrn := range sb.Warnings {
			_, _ = fmt.Fprintf(w, "- %s\n", wrn)
		}
	}
}

// markWire names the wire of a row when it is not HTTP, the norm.
func markWire(rows []tableModelRow, r *scoreboard.Row) {
	for i := range rows {
		if r.Wire == scoreboard.WireWebSocket {
			rows[i].Mode += "WS"
		}
	}
}

// unmeasuredRank separates the models a section measured from those it only queues.
func unmeasuredRank(r *tableModelRow) int {
	if r.Mode == "?" {
		return 1
	}
	return 0
}

// medalRank orders the tiers of the table: SOTA, Good, Cheap, then the rest.
func medalRank(model string) int {
	switch {
	case strings.HasSuffix(model, "🥇"):
		return 0
	case strings.HasSuffix(model, "🥈"):
		return 1
	case strings.HasSuffix(model, "🥉"):
		return 2
	}
	return 3
}

// appendModels adds one row per model of r, with the preference medal on the first.
func appendModels(rows []tableModelRow, r *scoreboard.Row, tmpRows []tableModelRow) []tableModelRow {
	if len(tmpRows) == 0 {
		row := tableModelRow{}
		fillEmptyFields(&row, "?")
		tmpRows = append(tmpRows, row)
	}
	for i, m := range r.Models {
		for j := range tmpRows {
			tmpRows[j].Model = m
			if i == 0 {
				if r.SOTA {
					tmpRows[j].Model += "🥇"
				}
				if r.Good {
					tmpRows[j].Model += "🥈"
				}
				if r.Cheap {
					tmpRows[j].Model += "🥉"
				}
			}
			rows = append(rows, tmpRows[j])
		}
	}
	return rows
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
	visitFieldsType(reflect.TypeFor[*T](), func(f reflect.StructField) {
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
	_, _ = fmt.Fprint(w, "|")
	for i, t := range titles {
		_, _ = fmt.Fprintf(w, " %s |", extendString(t, lengths[i]))
	}
	_, _ = fmt.Fprint(w, "\n|")
	for _, l := range lengths {
		_, _ = fmt.Fprintf(w, " %s |", strings.Repeat("-", l))
	}
	_, _ = fmt.Fprintln(w)
	for i := range rows {
		_, _ = fmt.Fprint(w, "|")
		j := 0
		visitFields(reflect.ValueOf(&rows[i]), func(v reflect.Value) {
			_, _ = fmt.Fprintf(w, " %s |", extendString(v.Elem().String(), lengths[j]))
			j++
		})
		_, _ = fmt.Fprintf(w, "\n")
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
	case '🏠', '❌', '💬', '✅', '📄', '🎤', '🤪', '🚩', '💨', '💸', '📸', '🎥', '💥', '🤐', '🧐', '🌐', '🤏', '📡', '🌱', '🪨', '🔀':
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
