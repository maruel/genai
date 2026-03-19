// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package codex_test

import (
	"context"
	"fmt"
	"log"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/codex"
)

func Example() {
	c, err := codex.New(genai.ProviderOptionModel("gpt-5.4"), codex.ReasoningEffortHigh)
	if err != nil {
		log.Fatal(err)
	}
	res, err := c.GenSync(context.Background(), genai.Messages{genai.NewTextMessage("Say hello")})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Replies[0].Text)
}
