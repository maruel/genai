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
	c, err := cerebras.New(ctx, genai.ProviderOptionModel("qwen-3-235b-a22b-instruct-2507"))
	if err != nil {
		log.Fatal(err)
	}
	type numbers struct {
		A int `json:"a"`
		B int `json:"b"`
	}
	msgs := genai.Messages{
		genai.NewTextMessage("What is 3214 + 5632? Call the tool \"add\" to tell me the answer. Do not explain. Be terse. Include only the answer."),
	}
	opts := genai.GenOptionTools{
		Tools: []genai.ToolDef{
			{
				Name:        "add",
				Description: "Add two numbers together and provides the result",
				Callback: func(ctx context.Context, input *numbers) (string, error) {
					return fmt.Sprintf("%d", input.A+input.B), nil
				},
			},
		},
		// Force the LLM to do a tool call.
		Force: genai.ToolCallRequired,
	}

	// Run the loop.
	res, _, err := adapters.GenSyncWithToolCallLoop(ctx, c, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}
	// Print the answer which is the last message generated.
	fmt.Println(res[len(res)-1].String())
}
