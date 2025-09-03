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
	// Other options (as of 2025-08):
	// - anthropic
	// - gemini
	// - openai
	c, err := perplexity.New(ctx, &genai.ProviderOptions{Model: genai.ModelCheap}, nil)
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
	opts := genai.OptionsTools{WebSearch: true}
	fragments, finish := c.GenStream(ctx, msgs, &opts)
	firstText := true
	for f := range fragments {
		if !f.Citation.IsZero() {
			fmt.Printf("Sources:\n")
			for _, src := range f.Citation.Sources {
				switch src.Type {
				case genai.CitationWeb:
					fmt.Printf("- %s / %s\n", src.Title, src.URL)
				case genai.CitationWebImage:
					fmt.Printf("- image: %s\n", src.URL)
				case genai.CitationWebQuery, genai.CitationDocument, genai.CitationTool:
				}
			}
		}
		if f.TextFragment != "" {
			if firstText {
				if _, err = os.Stdout.WriteString("\n"); err != nil {
					break
				}
				firstText = false
			}
			if _, err = os.Stdout.WriteString(f.TextFragment); err != nil {
				break
			}
		}
	}
	_, _ = os.Stdout.WriteString("\n")
	if _, err = finish(); err != nil {
		log.Fatal(err)
	}
}
