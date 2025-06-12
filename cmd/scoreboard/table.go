// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"os"
	"reflect"
	"slices"
	"strings"
	"unicode"
	"unicode/utf8"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers"
)

type column struct {
	Provider   string `title:"Provider"`
	Country    string `title:"Country"`
	Vision     string `title:"➛Vision"`
	PDF        string `title:"➛PDF"`
	Audio      string `title:"➛Audio"`
	Video      string `title:"➛Video"`
	JSON       string `title:"JSON➛"`
	JSONSchema string `title:"JSON+Schema➛"`
	ImageGen   string `title:"Image➛"`
	AudioGen   string `title:"Audio➛"`
	Chat       string `title:"Chat"`
	Streaming  string `title:"Streaming"`
	Doc        string `title:"Doc"`
	Batch      string `title:"Batch"`
	Seed       string `title:"Seed"`
	Tools      string `title:"Tools"`
	Files      string `title:"Files"`
	Citations  string `title:"Citations"`
}

func processOne(c genai.ProviderScoreboard) column {
	sb := c.Scoreboard()
	col := column{}
	resetToNope(&col)
	col.Provider = c.Name()
	if sb.DashboardURL != "" {
		col.Provider = "[" + c.Name() + "](" + sb.DashboardURL + ")"
	}
	country := countryMap[strings.ToLower(sb.Country)]
	if country == "" {
		country = sb.Country
	}
	col.Country = country
	_, isText := c.(genai.ProviderGen)
	_, isDoc := c.(genai.ProviderGenDoc)
	_, isAsync := c.(genai.ProviderGenAsync)
	_, isFiles := c.(genai.ProviderCache)
	for _, s := range sb.Scenarios {
		if _, hasImage := s.In[genai.ModalityImage]; hasImage {
			col.Vision = "✅"
		}
		if _, hasPDF := s.In[genai.ModalityPDF]; hasPDF {
			col.PDF = "✅"
		}
		if _, hasAudio := s.In[genai.ModalityAudio]; hasAudio {
			col.Audio = "✅"
		}
		if _, hasVideo := s.In[genai.ModalityVideo]; hasVideo {
			col.Video = "✅"
		}
		if s.GenSync != nil && s.GenSync.JSON && s.GenStream.JSON {
			col.JSON = "✅"
		}
		if s.GenSync != nil && s.GenSync.JSONSchema && s.GenStream.JSONSchema {
			col.JSONSchema = "✅"
		}
		if _, hasTextIn := s.In[genai.ModalityText]; hasTextIn {
			if _, hasImageOut := s.Out[genai.ModalityImage]; hasImageOut {
				col.ImageGen = "✅"
			}
		}
		if _, hasTextIn := s.In[genai.ModalityText]; hasTextIn {
			if _, hasTextOut := s.Out[genai.ModalityText]; hasTextOut && isText {
				col.Chat = "✅"
				col.Streaming = "✅"
			}
		}
		if isDoc {
			col.Doc = "✅"
		}
		if isAsync {
			col.Batch = "✅"
		}
		if isFiles {
			col.Files = "✅"
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
			} else if s.GenSync.Tools == genai.Flaky || s.GenStream.Tools == genai.Flaky && col.Tools == "" {
				col.Tools = "💨"
				if s.GenSync.BiasedTool == genai.False {
					col.Tools += "⚖️"
				}
				if s.GenSync.IndecisiveTool == genai.True {
					col.Tools += " 🤷"
				}
			}
			if s.GenSync.Citations {
				col.Citations = "✅"
			}
			//if s.GenSync.Seed {
			//	col.Seed = "✅"
			//}
		}
		if _, hasAudioOut := s.Out[genai.ModalityAudio]; hasAudioOut {
			col.AudioGen = "✅"
		}
	}
	return col
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

func resetToNope[T any](c *T) {
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
			if l := VisibleWidth(val.Field(j).String()); l > lengths[j] {
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
		fmt.Fprintf(w, "%-*s | ", lengths[i], t)
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
			fmt.Fprintf(w, "%-*s |", lengths[i], v.Field(i).String())
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

func printTable() error {
	var columns []column
	for name, f := range providers.All {
		c, err := f("", nil)
		if err != nil {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: %v\n", name, err)
			continue
		}
		ps, ok := c.(genai.ProviderScoreboard)
		if !ok {
			fmt.Fprintf(os.Stderr, "ignoring provider %s: doesn't support scoreboard\n", name)
			continue
		}
		columns = append(columns, processOne(ps))
	}
	slices.SortFunc(columns, func(a, b column) int {
		return strings.Compare(a.Provider, b.Provider)
	})

	printMarkdownTable(os.Stdout, columns)
	return nil
}

// A complete piece of crap generated by claude that doesn't work follows. I'll have to figure out either my
// terminal sucks or it's claude.

// VisibleWidth calculates the display width of a string in a terminal
// Properly handles grapheme clusters (composed characters like flag emojis)
func VisibleWidth(s string) int {
	width := 0

	for len(s) > 0 {
		cluster, remaining := nextGraphemeCluster(s)
		width += GraphemeWidth(cluster)
		s = remaining
	}

	return width
}

// nextGraphemeCluster extracts the next grapheme cluster from the string
// Returns the cluster and the remaining string
func nextGraphemeCluster(s string) (string, string) {
	if len(s) == 0 {
		return "", ""
	}

	// Get the first rune
	firstRune, firstSize := utf8.DecodeRuneInString(s)
	if firstRune == utf8.RuneError {
		return s[:1], s[1:]
	}

	pos := firstSize

	// Keep adding runes that should be part of the same grapheme cluster
	for pos < len(s) {
		nextRune, nextSize := utf8.DecodeRuneInString(s[pos:])
		if nextRune == utf8.RuneError {
			break
		}

		// Check if this rune should be part of the current grapheme cluster
		if shouldCombine(firstRune, nextRune, s[:pos], s[pos:pos+nextSize]) {
			pos += nextSize
			continue
		}

		break
	}

	return s[:pos], s[pos:]
}

// shouldCombine determines if two runes should be part of the same grapheme cluster
func shouldCombine(firstRune, nextRune rune, prevText, nextText string) bool {
	// Zero Width Joiner (ZWJ) sequences - common in complex emojis
	if nextRune == 0x200D { // ZWJ
		return true
	}

	// If previous character was ZWJ, combine with next
	if len(prevText) >= 3 {
		lastRune, _ := utf8.DecodeLastRuneInString(prevText)
		if lastRune == 0x200D {
			return true
		}
	}

	// Variation selectors (emoji vs text presentation)
	if nextRune >= 0xFE00 && nextRune <= 0xFE0F {
		return true
	}
	if nextRune >= 0xE0100 && nextRune <= 0xE01EF {
		return true
	}

	// Combining marks
	if unicode.In(nextRune, unicode.Mn, unicode.Me, unicode.Mc) {
		return true
	}

	// Emoji modifiers (skin tone, etc.)
	if nextRune >= 0x1F3FB && nextRune <= 0x1F3FF {
		return true
	}

	// Regional indicator symbols (country flags)
	if isRegionalIndicator(firstRune) && isRegionalIndicator(nextRune) {
		return true
	}

	// Tag sequences (used in flag emojis like England, Scotland)
	if nextRune >= 0xE0020 && nextRune <= 0xE007F {
		return true
	}
	if nextRune == 0xE007F { // Cancel tag
		return true
	}

	return false
}

// isRegionalIndicator checks if a rune is a regional indicator symbol
func isRegionalIndicator(r rune) bool {
	return r >= 0x1F1E6 && r <= 0x1F1FF
}

// GraphemeWidth returns the display width of a grapheme cluster
func GraphemeWidth(cluster string) int {
	if len(cluster) == 0 {
		return 0
	}

	// Get the first rune to determine the base width
	firstRune, _ := utf8.DecodeRuneInString(cluster)

	// Special case for regional indicators (country flags)
	if isRegionalIndicator(firstRune) && utf8.RuneCountInString(cluster) >= 2 {
		return 2 // Country flags are double-width
	}

	// For other grapheme clusters, use the width of the first rune
	return RuneWidth(firstRune)
}

// RuneWidth returns the display width of a single rune
func RuneWidth(r rune) int {
	// Control characters and DEL are zero width
	if r < 32 || r == 127 {
		return 0
	}

	// Combining marks are zero width
	if unicode.In(r, unicode.Mn, unicode.Me, unicode.Mc) {
		return 0
	}

	// East Asian Wide and Fullwidth characters are double width
	if isWideRune(r) {
		return 2
	}

	// Default to single width
	return 1
}

// isWideRune checks if a rune should be displayed as double-width
func isWideRune(r rune) bool {
	// East Asian Wide characters (Unicode category)
	if unicode.In(r,
		unicode.Han,                        // CJK ideographs
		unicode.Hangul,                     // Korean
		unicode.Hiragana, unicode.Katakana, // Japanese
	) {
		return true
	}

	// Emoji and symbol ranges that are typically double-width
	switch {
	case r >= 0x1F300 && r <= 0x1F9FF: // Miscellaneous Symbols and Pictographs, Emoticons, etc.
		return true
	case r >= 0x2600 && r <= 0x26FF: // Miscellaneous Symbols - ambiguous, but often double-width
		return isDoubleWidthSymbol(r)
	case r >= 0x2700 && r <= 0x27BF: // Dingbats - many are ambiguous width
		return isDoubleWidthDingbat(r)
	case r >= 0x1F100 && r <= 0x1F1FF: // Enclosed Alphanumeric Supplement + Regional Indicators
		return true
	case r >= 0x1F200 && r <= 0x1F2FF: // Enclosed Ideographic Supplement
		return true
	case r >= 0x3000 && r <= 0x303F: // CJK Symbols and Punctuation
		return true
	case r >= 0xFF00 && r <= 0xFFEF: // Halfwidth and Fullwidth Forms
		return true
	}

	return false
}

// isDoubleWidthSymbol checks specific symbols in the Miscellaneous Symbols block
func isDoubleWidthSymbol(r rune) bool {
	// Most symbols in this range are single-width in modern terminals
	// Only include the clearly double-width ones
	switch r {
	case 0x26A0, 0x26A1: // Warning sign, High voltage
		return false // These are typically single-width
	default:
		// Conservative approach: treat most as single-width unless clearly emoji-like
		return r >= 0x26BD && r <= 0x26FF // Soccer ball and onwards tend to be emoji-style
	}
}

// isDoubleWidthDingbat checks specific dingbats for width
func isDoubleWidthDingbat(r rune) bool {
	// Many dingbats are ambiguous width or single-width in terminals
	// Be conservative and only mark clearly double-width ones
	switch r {
	case 0x274C: // ❌ Cross mark - typically single-width in most terminals
		return false
	case 0x2705: // ✅ Check mark - typically single-width
		return false
	case 0x2728: // ✨ Sparkles - often double-width (emoji-style)
		return true
	case 0x2744: // ❄️ Snowflake - often double-width
		return true
	default:
		// Conservative default: most dingbats are single-width in terminals
		return false
	}
}
