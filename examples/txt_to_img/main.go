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
	c, err := togetherai.New(ctx, &genai.ProviderOptions{Model: "black-forest-labs/FLUX.1-schnell-Free"}, nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Carton drawing of a husky playing on the beach."),
	}
	result, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	for _, r := range result.Replies {
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
			defer req.Body.Close()
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
