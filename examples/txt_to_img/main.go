// Use Together.AI's free (!) image generation albeit with low rate limit.
//
// Some providers return an URL that must be fetched manually within a few
// minutes or hours, some return the data inline. This example handles both cases.
//
// This requires `TOGETHER_API_KEY` (https://api.together.ai/settings/api-keys)
// environment variable to authenticate.

package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/togetherai"
)

func main() {
	ctx := context.Background()
	// Other options (as of 2025-08):
	// - "gpt-image-1" from openai
	// - "imagen-4.0-*" from gemini
	// - pollinations
	c, err := togetherai.New(ctx, genai.ProviderOptionModel("black-forest-labs/FLUX.1-schnell-Free"))
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Carton drawing of a husky playing on the beach."),
	}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	for i := range res.Replies {
		r := &res.Replies[i]
		if r.Doc.IsZero() {
			fmt.Println(r.Text)
			continue
		}
		// The image can be returned as an URL or inline, depending on the provider.
		var src io.Reader
		if r.Doc.URL != "" {
			req, err := c.HTTPClient().Get(r.Doc.URL)
			if err != nil {
				log.Fatal(err)
			} else if req.StatusCode != http.StatusOK {
				log.Fatal(req.StatusCode)
			}
			src = req.Body
			defer func() { _ = req.Body.Close() }()
		} else {
			src = r.Doc.Src
		}
		b, err := io.ReadAll(src)
		if err != nil {
			log.Fatal(err)
		}
		name := r.Doc.GetFilename()
		fmt.Printf("Wrote: %s\n", name)
		if err = os.WriteFile(name, b, 0o644); err != nil {
			log.Fatal(err)
		}
	}
}
