// Analyze an audio file.
//
// This requires `OPENAI_API_KEY` (https://platform.openai.com/settings/organization/api-keys)
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
	"github.com/maruel/genai/providers/openaichat"
)

func mainImpl() (err error) {
	ctx := context.Background()
	// Other options (as of 2025-08):
	// - "voxtral-*-latest" from mistral
	// - any "gemini-2-5-*" model from gemini
	c, err := openaichat.New(ctx, genai.ProviderOptionModel("gpt-4o-audio-preview"))
	if c != nil {
		defer func() { err = errors.Join(err, c.Close()) }()
	}
	if err != nil {
		return err
	}
	// Use an audio file from the test data suite.
	// OpenAI only allows inline audio.
	resp, err := http.Get("https://github.com/maruel/genai/raw/refs/heads/main/scoreboard/testdata/audio.mp3")
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
			{Text: "What was the word?"},
			{Doc: genai.Doc{Src: bytes.NewReader(b), Filename: "audio.mp3"}},
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
