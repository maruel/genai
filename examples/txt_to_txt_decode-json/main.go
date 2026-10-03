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
	"errors"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openaichat"
)

func mainImpl() (err error) {
	ctx := context.Background()
	// See ../../docs/MODELS.md to see which providers support this.
	c, err := openaichat.New(ctx, genai.ModelGood)
	if c != nil {
		defer func() { err = errors.Join(err, c.Close()) }()
	}
	if err != nil {
		return err
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Is a circle round? Reply as JSON."),
	}
	var circle struct {
		Round bool `json:"round"`
	}
	opts := genai.GenOptionText{DecodeAs: &circle}
	res, err := c.GenSync(ctx, msgs, &opts)
	if err != nil {
		return err
	}
	if err = res.Decode(&circle); err != nil {
		return err
	}
	fmt.Printf("Round: %v\n", circle.Round)
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
