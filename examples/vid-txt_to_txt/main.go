// Run vision to analyze a video.
//
// This requires `GEMINI_API_KEY` (https://aistudio.google.com/apikey)
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
	// Other options (as of 2025-08):
	// - None!
	c, err := gemini.New(ctx, genai.ModelCheap)
	if err != nil {
		log.Fatal(err)
	}
	// Use a video from the test data suite.
	// Gemini only allows URL references from files uploaded with its file API. Otherwise we need to send it
	// inline.
	resp, err := http.Get("https://github.com/maruel/genai/raw/refs/heads/main/scoreboard/testdata/video.mp4")
	if err != nil || resp.StatusCode != http.StatusOK {
		log.Fatal(err)
	}
	b, err := io.ReadAll(resp.Body)
	_ = resp.Body.Close()
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
