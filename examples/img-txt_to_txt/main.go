// Analyze a picture provided as an URL. The response is streamed out the
// console as the reply is generated.
//
// This requires `MISTRAL_API_KEY` (https://console.mistral.ai/api-keys)
// environment variable to authenticate.

package main

import (
	"context"
	"log"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/mistral"
)

func main() {
	ctx := context.Background()
	// Most SOTA providers support vision. Notable exceptions (as of 2025-08) are cerebras and groq.
	c, err := mistral.New(ctx, genai.ModelGood)
	if err != nil {
		log.Fatal(err)
	}
	// Uses a banana picture from wikipedia directly via URL. Source: https://en.m.wikipedia.org/wiki/File:Banana-Single.jpg
	msgs := genai.Messages{
		genai.Message{Requests: []genai.Request{
			{Text: "Succinctly describe this image."},
			{Doc: genai.Doc{URL: "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Banana-Single.jpg/960px-Banana-Single.jpg"}},
		}},
	}
	fragments, finish := c.GenStream(ctx, msgs)
	for f := range fragments {
		if _, err = os.Stdout.WriteString(f.Text); err != nil {
			break
		}
	}
	_, _ = os.Stdout.WriteString("\n")
	if _, err = finish(); err != nil {
		log.Fatal(err)
	}
}
