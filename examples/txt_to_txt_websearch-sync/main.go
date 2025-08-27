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
	c, err := perplexity.New(ctx, &genai.ProviderOptions{Model: genai.ModelCheap}, nil)
	if err != nil {
		log.Fatal(err)
	}

	msgs := genai.Messages{{
		Requests: []genai.Request{
			{Text: "Who holds ultimate power of Canada? Answer succinctly."},
		},
	}}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range res.Replies {
		for _, ci := range r.Citations {
			fmt.Printf("Sources:\n")
			for _, src := range ci.Sources {
				if src.Type == "web" {
					fmt.Printf("- %s / %s\n", src.Title, src.URL)
				} else {
					fmt.Printf("Image: %s\n", src.URL)
				}
			}
		}
	}
	fmt.Printf("\nAnswer: %s\n", res.String())
}
