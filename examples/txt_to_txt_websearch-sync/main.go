// Use a websearch provider.
//
// This requires `GEMINI_API_KEY` (https://aistudio.google.com/apikey)
// environment variable to authenticate.

package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/gemini"
)

func mainImpl() (err error) {
	ctx := context.Background()
	// Other options:
	// - anthropic
	// - groq
	// - openairesponses
	c, err := gemini.New(ctx, genai.ModelCheap)
	if c != nil {
		defer func() { err = errors.Join(err, c.Close()) }()
	}
	if err != nil {
		return err
	}

	msgs := genai.Messages{{
		Requests: []genai.Request{
			{Text: "Who holds ultimate power of Canada? Answer succinctly."},
		},
	}}
	opts := genai.GenOptionWeb{Search: true}
	res, err := c.GenSync(ctx, msgs, &opts)
	if err != nil {
		return err
	}
	for i := range res.Replies {
		r := &res.Replies[i]
		if !r.Citation.IsZero() {
			fmt.Printf("Sources:\n")
			for j := range r.Citation.Sources {
				src := &r.Citation.Sources[j]
				switch src.Type {
				case genai.CitationWeb:
					fmt.Printf("- %s / %s\n", src.Title, src.URL)
				case genai.CitationWebImage:
					fmt.Printf("- image: %s\n", src.URL)
				case genai.CitationWebQuery, genai.CitationDocument, genai.CitationTool:
				}
			}
		}
	}
	fmt.Printf("\nAnswer: %s\n", res.String())
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
