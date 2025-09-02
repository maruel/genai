// Use a websearch provider.
//
// This requires `PERPLEXITY_API_KEY` (https://www.perplexity.ai/settings/api)
// environment variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"

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
	res, err := c.GenSync(ctx, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range res.Replies {
		for _, ci := range r.Citations {
			fmt.Printf("Sources:\n")
			for _, src := range ci.Sources {
				switch src.Type {
				case genai.CitationWeb:
					fmt.Printf("- %s / %s\n", src.Title, src.URL)
				case genai.CitationWebImage:
					fmt.Printf("- image: %s\n", src.URL)
				}
			}
		}
	}
	fmt.Printf("\nAnswer: %s\n", res.String())
}
