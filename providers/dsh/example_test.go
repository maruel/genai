// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Example usage of the DeepSeek Harness provider.

package dsh_test

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/dsh"
)

func Example() {
	c, err := dsh.New(context.Background(), genai.ProviderOptionModel("deepseek-official/deepseek-v4-flash"))
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
		genai.EffortLow,
	)
	if err != nil {
		log.Print(err)
		return
	}
	fmt.Println(res.String())
}
