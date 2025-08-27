// Analyze a video provided as an URL. The response is streamed out the
// console as the reply is generated.
//
// This requires `MISTRAL_API_KEY` (https://console.mistral.ai/api-keys)
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
	"github.com/maruel/genai/providers/gemini"
)

func main() {
	ctx := context.Background()
	c, err := gemini.New(ctx, &genai.ProviderOptions{}, nil)
	if err != nil {
		log.Fatal(err)
	}
	// Use a video from the test data suite.
	resp, err := http.Get("https://github.com/maruel/genai/raw/refs/heads/main/scoreboard/testdata/video.mp4")
	if err != nil {
		log.Fatal(err)
	}
	b, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.Message{Requests: []genai.Request{
			{Text: "Say the word. Say nothing else."},
			{Doc: genai.Doc{Src: bytes.NewReader(b), Filename: "video.mp4"}},
		}},
	}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.String())
}
