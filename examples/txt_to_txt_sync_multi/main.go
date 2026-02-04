// This selects a good default model based on OpenAI's currently published
// models, sends a prompt and prints the response as a string.
//
// This requires `OPENAI_API_KEY` (https://platform.openai.com/api-keys)
// environment variable to authenticate.

package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openaichat"
)

func main() {
	// Set up the OpenAI client
	ctx := context.Background()
	c, err := openaichat.New(ctx, genai.ModelCheap)
	if err != nil {
		log.Fatal(err)
	}

	// Tell the user what we're doing. We'll also use this as a prompt for the
	// LLM itself so everyone plays with the same rules.
	const prompt = "Let's play a word association game. You pick a single word, then I say the first word I think of, then you respond with a word, and so on. You go first.\n> "
	fmt.Print(prompt)
	var msgs genai.Messages

	// Read in the users input until they are done.
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		line := scanner.Text()
		if line == "q" || line == "quit" {
			fmt.Println("Seeya.")
			os.Exit(0)
		}

		// If this is the first time we've gotten input we need to structure it with both
		// our prompt and the users first word.
		if len(msgs) == 0 {
			msgs = append(msgs, genai.NewTextMessage(fmt.Sprintf("%s%s", prompt, line)))
		} else {
			msgs = append(msgs, genai.NewTextMessage(line))
		}

		res, err := c.GenSync(ctx, msgs)
		if err != nil {
			log.Fatal(err)
			os.Exit(1)
		}
		fmt.Printf("ChatGPT: %s\n> ", res.Message.String())
		// Add the response from the LLM to the set of messages so that the context continues
		// to grow as the user continues to engage.
		msgs = append(msgs, res.Message)

	}
}
