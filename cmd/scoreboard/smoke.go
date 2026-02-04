// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers"
	"github.com/maruel/genai/smoke"
)

func smokeModel(ctx context.Context, w io.Writer, provider, model string) error {
	// TODO: We may want to run in strict mode?
	// TODO: We will want to record the HTTP requests, it's wasteful otherwise.
	if _, err := providers.All[provider].Factory(ctx, genai.ProviderOptionModel(model)); err != nil {
		return err
	}
	pf := func(name string) genai.Provider {
		p, _ := providers.All[provider].Factory(ctx, genai.ProviderOptionModel(model))
		return p
	}
	sc, usage, err := smoke.Run(ctx, pf)
	if err != nil {
		return err
	}
	b, _ := json.MarshalIndent(sc, "", "  ")
	fmt.Fprintf(w, "%s\n", string(b))
	fmt.Fprintf(os.Stderr, "Usage: %s\n", usage.String())
	return nil
}
