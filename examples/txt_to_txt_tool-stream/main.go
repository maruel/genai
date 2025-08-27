// Leverage a thinking model to see the thinking process while trying to use
// tool calls to answer the user's question. This enables keeping the user
// updated to see the progress.
//
// This requires `GROQ_API_KEY` (https://console.groq.com/keys) environment
// variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers/groq"
)

func main() {
	ctx := context.Background()
	// Some explicit chain-of-thoughts model-provider combinations only support GenSync and not GenStream.
	// Refer to the Scoreboard().
	c, err := groq.New(ctx, &genai.ProviderOptions{Model: "openai/gpt-oss-120b"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	p := adapters.WrapThinking(c)
	type math struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	msgs := genai.Messages{
		genai.NewTextMessage("What is 3214 + 5632? Call the tool \"add\" to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.OptionsText{
		Tools: []genai.ToolDef{
			{
				Name:        "add",
				Description: "Add two numbers together and provides the result",
				Callback: func(ctx context.Context, input *math) (string, error) {
					return fmt.Sprintf("%d", input.A+input.B), nil
				},
			},
		},
	}

	// Run the loop.
	fragments, finish := adapters.GenStreamWithToolCallLoop(ctx, p, msgs, &opts)
	mode := ""
	for f := range fragments {
		if f.ThinkingFragment != "" {
			if mode != "thinking" {
				if _, err = os.Stdout.WriteString("# Thinking\n"); err != nil {
					break
				}
				mode = "thinking"
			}
			if _, err = os.Stdout.WriteString(f.ThinkingFragment); err != nil {
				break
			}
		} else if f.TextFragment != "" {
			if mode != "text" {
				if _, err = os.Stdout.WriteString("\n# Answer\n"); err != nil {
					break
				}
				mode = "text"
			}
			if _, err = os.Stdout.WriteString(f.TextFragment); err != nil {
				break
			}
		} else if !f.ToolCall.IsZero() {
			if mode != "tool" {
				if _, err = os.Stdout.WriteString("\n\n# Tool call\n"); err != nil {
					break
				}
				mode = "tool"
			}
			if _, err = fmt.Printf("%s\n", f.ToolCall); err != nil {
				break
			}
		}
	}
	_, _ = os.Stdout.WriteString("\n")
	_, usage, err := finish()
	fmt.Printf("\nTokens usage: %s\n", usage.String())
	if err != nil {
		log.Fatal(err)
	}
}
