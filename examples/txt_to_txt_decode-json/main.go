// Tell the LLM to use a specific Go struct to determine the JSON schema to
// generate the response. This is much more lightweight than tool calling!
//
// It is very useful when we want the LLM to make a choice between values,
// to return a number or a boolean (true/false). Enums are supported.
//
// This requires `OPENAI_API_KEY` (https://platform.openai.com/settings/organization/api-keys)
// environment variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openai"
)

func main() {
	ctx := context.Background()
	c, err := openai.New(ctx, &genai.ProviderOptions{}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Is a circle round? Reply as JSON."),
	}
	var circle struct {
		Round bool `json:"round"`
	}
	opts := genai.OptionsText{DecodeAs: &circle}
	res, err := c.GenSync(ctx, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	if err = res.Decode(&circle); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Round: %v\n", circle.Round)
	fmt.Printf("\nTokens usage: %s\n", res.Usage.String())
}
