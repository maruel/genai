// This selects a good default model based on Anthropic's currently published
// models, sends a prompt and prints the response as a string.
//
// This requires `ANTHROPIC_API_KEY` (https://console.anthropic.com/settings/keys)
// environment variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/anthropic"
)

func main() {
	ctx := context.Background()
	c, err := anthropic.New(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice. Answer succinctly."),
	}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.String())
}
