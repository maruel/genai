// Prints the tokens processed and generated for the request and the remaining quota if the provider supports
// it.
//
// This requires `GROQ_API_KEY` (https://console.groq.com/keys) environment
// variable to authenticate.

package main

import (
	"context"
	"errors"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/groq"
)

func mainImpl() (err error) {
	ctx := context.Background()
	c, err := groq.New(ctx, genai.ProviderOptionModel("openai/gpt-oss-120b"))
	if c != nil {
		defer func() { err = errors.Join(err, c.Close()) }()
	}
	if err != nil {
		return err
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Describe poutine as a French person who just arrived in Québec"),
	}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		return err
	}
	fmt.Println(res.String())
	fmt.Printf("\nTokens usage: %s\n", res.Usage.String())
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
