// A LLM can both retrieve information and act on its environment through tool
// calling. This unblocks a whole realm of possibilities. Our design enables
// dense strongly typed code that favorably compares to python.
//
// This example uses a manual loop to show how tool calling works inside the loop.
//
// This requires `CEREBRAS_API_KEY` (https://cloud.cerebras.ai/platform/)
// environment variable to authenticate.

package main

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/cerebras"
)

func main() {
	ctx := context.Background()
	// While most provider support tool calling in theory, most fail reliability smoke tests.
	// See ../../docs/MODELS.md for the scoreboard and each provider's Scoreboard().
	// This is continuously improving.
	c, err := cerebras.New(ctx, genai.ProviderOptionModel("qwen-3-235b-a22b-instruct-2507"))
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
	opts := genai.GenOptionTools{
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
		Force: genai.ToolCallRequired,
	}
	res, err := c.GenSync(ctx, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}

	// Add the assistant's message to the messages list.
	msgs = append(msgs, res.Message)

	// Process the tool call from the assistant.
	msg, err := res.DoToolCalls(ctx, opts.Tools)
	if err != nil {
		log.Fatalf("Error calling tool: %v", err)
	}
	if msg.IsZero() {
		log.Fatal("Expected a tool call")
	}

	// Add the tool call response to the messages list.
	msgs = append(msgs, msg)

	// Follow up so the LLM can interpret the tool call response. Tell the LLM to not do a tool call this time.
	opts.Force = genai.ToolCallNone
	res, err = c.GenSync(ctx, msgs, &opts)
	if err != nil {
		log.Fatal(err)
	}

	// Print the result.
	fmt.Println(res.String())
}
