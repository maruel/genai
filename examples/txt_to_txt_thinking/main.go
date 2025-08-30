// This selects a model that output reasoning tokens.
//
// This requires `DEEPSEEK_API_KEY` (https://platform.deepseek.com/api_keys)
// environment variable to authenticate.

package main

import (
	"context"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/deepseek"
)

func main() {
	ctx := context.Background()
	// Most SOTA providers support thinking but do not provide the full tokens.
	// As of 2025-08, qwen-3-235b-a22b-instruct-2507 is quite solid.
	c, err := deepseek.New(ctx, &genai.ProviderOptions{Model: "deepseek-reasoner"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice."),
	}
	fragments, finish := c.GenStream(ctx, msgs)
	reasoning := false
	for f := range fragments {
		if f.ReasoningFragment != "" {
			if !reasoning {
				if _, err = os.Stdout.WriteString("# Reasoning\n"); err != nil {
					break
				}
				reasoning = true
			}
			if _, err = os.Stdout.WriteString(f.ReasoningFragment); err != nil {
				break
			}
		} else if f.TextFragment != "" {
			if reasoning {
				if _, err = os.Stdout.WriteString("\n\n# Answer\n"); err != nil {
					break
				}
				reasoning = false
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
