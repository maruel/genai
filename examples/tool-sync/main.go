// A LLM can both retrieve information and act on its environment through tool
// calling. This unblocks a whole realm of possibilities. Our design enables
// dense strongly typed code that favorably compares to python.
//
// This requires `CEREBRAS_API_KEY` (https://cloud.cerebras.ai/platform/)
// environment variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers/cerebras"
)

func main() {
	ctx := context.Background()
	c, err := cerebras.New(ctx, &genai.ProviderOptions{Model: "qwen-3-235b-a22b-instruct-2507"}, nil)
	if err != nil {
		log.Fatal(err)
	}
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
		// Force the LLM to do a tool call.
		ToolCallRequest: genai.ToolCallRequired,
	}

	// Run the loop.
	resp, _, err := adapters.GenSyncWithToolCallLoop(ctx, c, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	// Print the answer which is the last message generated.
	fmt.Println(resp[len(resp)-1].String())
}
