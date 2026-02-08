// Use a websearch provider.
//
// This requires `PERPLEXITY_API_KEY` (https://www.perplexity.ai/settings/api)
// environment variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/perplexity"
)

func main() {
	ctx := context.Background()
	// Warning: this is surpringly expensive.
	// Other options (as of 2025-09):
	// - anthropic
	// - gemini
	// - groq
	// - openai
	c, err := perplexity.New(ctx, genai.ModelCheap)
	if err != nil {
		log.Fatal(err)
	}

	msgs := genai.Messages{{
		Requests: []genai.Request{
			{Text: "Who holds ultimate power of Canada? Answer succinctly."},
		},
	}}
	// perplexity has websearch enabled by default so this is a no-op. It is needed to enable websearch for
	// anthropic, gemini and openai.
	opts := genai.GenOptionTools{WebSearch: true}
	fragments, finish := c.GenStream(ctx, msgs, &opts)
	firstText := true
	for f := range fragments {
		if !f.Citation.IsZero() {
			fmt.Printf("Sources:\n")
			for i := range f.Citation.Sources {
				switch f.Citation.Sources[i].Type {
				case genai.CitationWeb:
					fmt.Printf("- %s / %s\n", f.Citation.Sources[i].Title, f.Citation.Sources[i].URL)
				case genai.CitationWebImage:
					fmt.Printf("- image: %s\n", f.Citation.Sources[i].URL)
				case genai.CitationWebQuery, genai.CitationDocument, genai.CitationTool:
				}
			}
		}
		if f.Text != "" {
			if firstText {
				if _, err = os.Stdout.WriteString("\n"); err != nil {
					break
				}
				firstText = false
			}
			if _, err = os.Stdout.WriteString(f.Text); err != nil {
				break
			}
		}
	}
	_, _ = os.Stdout.WriteString("\n")
	if _, err = finish(); err != nil {
		log.Fatal(err)
	}
}
