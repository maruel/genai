// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Example usage of the OpenCode provider.

package opencode_test

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/opencode"
)

func Example() {
	c, err := opencode.New(context.Background(), genai.ProviderOptionModel("opencode/big-pickle"))
	if c != nil {
		defer func() {
			if err := c.Close(); err != nil {
				log.Printf("Close: %v", err)
			}
		}()
	}
	if err != nil {
		log.Print(err)
		return
	}
	res, err := c.GenSync(
		context.Background(),
		genai.Messages{genai.NewTextMessage("Say hello")},
		&opencode.GenOption{Effort: opencode.EffortXHigh},
	)
	if err != nil {
		log.Print(err)
		return
	}
	fmt.Println(res.Replies[0].Text)
}
