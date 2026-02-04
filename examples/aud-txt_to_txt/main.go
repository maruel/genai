// Analyze an audio file.
//
// This requires `OPENAI_API_KEY` (https://platform.openai.com/settings/organization/api-keys)
// environment variable to authenticate.

package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openaichat"
)

func main() {
	ctx := context.Background()
	// Other options (as of 2025-08):
	// - "voxtral-*-latest" from mistral
	// - any "gemini-2-5-*" model from gemini
	c, err := openaichat.New(ctx, genai.ProviderOptionModel("gpt-4o-audio-preview"))
	if err != nil {
		log.Fatal(err)
	}
	// Use an audio file from the test data suite.
	// OpenAI only allows inline audio.
	resp, err := http.Get("https://github.com/maruel/genai/raw/refs/heads/main/scoreboard/testdata/audio.mp3")
	if err != nil || resp.StatusCode != 200 {
		log.Fatal(err)
	}
	b, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.Message{Requests: []genai.Request{
			{Text: "What was the word?"},
			{Doc: genai.Doc{Src: bytes.NewReader(b), Filename: "audio.mp3"}},
		}},
	}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.String())
}
