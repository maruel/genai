// Run vision to analyze a video.
//
// This requires `GEMINI_API_KEY` (https://aistudio.google.com/apikey)
// environment variable to authenticate.

package main

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/gemini"
)

func mainImpl() (err error) {
	ctx := context.Background()
	// Other options (as of 2025-08):
	// - None!
	c, err := gemini.New(ctx, genai.ModelCheap)
	if c != nil {
		defer func() { err = errors.Join(err, c.Close()) }()
	}
	if err != nil {
		return err
	}
	// Use a video from the test data suite.
	// Gemini only allows URL references from files uploaded with its file API. Otherwise we need to send it
	// inline.
	resp, err := http.Get("https://github.com/maruel/genai/raw/refs/heads/main/scoreboard/testdata/video.mp4")
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return errors.Join(fmt.Errorf("unexpected HTTP status %d", resp.StatusCode), resp.Body.Close())
	}
	b, err := io.ReadAll(resp.Body)
	err = errors.Join(err, resp.Body.Close())
	if err != nil {
		return err
	}
	msgs := genai.Messages{
		genai.Message{Requests: []genai.Request{
			{Text: "Say the word. Say nothing else."},
			{Doc: genai.Doc{Src: bytes.NewReader(b), Filename: "video.mp4"}},
		}},
	}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		return err
	}
	fmt.Println(res.String())
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
