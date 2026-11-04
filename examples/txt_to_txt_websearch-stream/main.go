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
	"os"

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
		return err
	}
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
