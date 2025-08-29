// Prints the tokens processed and generated for the request and the remaining quota if the provider supports
// it.
//
// This requires `GROQ_API_KEY` (https://console.groq.com/keys) environment
// variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/groq"
)

func main() {
	ctx := context.Background()
	c, err := groq.New(ctx, &genai.ProviderOptions{Model: "openai/gpt-oss-120b"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Describe poutine as a French person who just arrived in Québec"),
	}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.String())
	fmt.Printf("\nTokens usage: %s\n", res.Usage.String())
}
