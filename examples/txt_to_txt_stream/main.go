// This selects a good default model based on Anthropic's currently published
// models, sends a prompt and prints the response as it is generated.
// This leverages go 1.23 iterators (https://go.dev/blog/range-functions).
// Notice how little difference there is between this and ../txt_to_txt_sync/.
//
// This requires `ANTHROPIC_API_KEY` (https://console.anthropic.com/settings/keys)
// environment variable to authenticate.

package main

import (
	"context"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/anthropic"
)

func main() {
	ctx := context.Background()
	// All providers except image-only-providers (e.g. bfl) support streaming.
	// Streaming may be emulated in some cases, generally in non-text output modalities.
	c, err := anthropic.New(ctx, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice."),
	}
	fragments, finish := c.GenStream(ctx, msgs)
	for f := range fragments {
		if _, err = os.Stdout.WriteString(f.Text); err != nil {
			break
		}
	}
	if _, err = finish(); err != nil {
		log.Fatal(err)
	}
}
