// Retrieve the logprobs.

package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers"
)

func indentString(s string) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	for i, line := range lines {
		lines[i] = "    " + line
	}
	return strings.Join(lines, "\n")
}

func run(ctx context.Context, c genai.Provider) {
	msgs := genai.Messages{genai.NewTextMessage("Tell a short joke")}
	res, err := c.GenSync(ctx, msgs, &genai.GenOptionsText{TopLogprobs: 5})
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error while trying with %s: %s\n", c.Name(), err)
		return
	}
	fmt.Printf("Provider %s\n  Reply:\n%s\n  Logprobs:\n", c.Name(), indentString(res.String()))
	for _, token := range res.Logprobs {
		for i, l := range token {
			if i != 0 {
				fmt.Printf("      ")
			} else {
				fmt.Printf("    * ")
			}
			fmt.Printf("%+12.6f:", l.Logprob)
			if l.ID != 0 {
				fmt.Printf(" %6d", l.ID)
			}
			if l.Text != "" {
				fmt.Printf(" %q", l.Text)
			}
			fmt.Printf("\n")
		}
	}
}

func main() {
	ctx := context.Background()
	// Providers supporting logprobs (2025-09):
	supported := []string{
		"cerebras",
		"cohere",
		"deepseek",
		"gemini",
		"huggingface",
	}
	seen := false
	for _, name := range supported {
		cfg := providers.All[name]
		if c, err := cfg.Factory(ctx, genai.ModelGood); err == nil {
			if seen {
				fmt.Printf("\n")
			}
			seen = true
			run(ctx, c)
			break
		}
	}
	if !seen {
		fmt.Printf("No supported provider found\n")
	}
}
