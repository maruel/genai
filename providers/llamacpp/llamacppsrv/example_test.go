// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacppsrv_test

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
)

func Example() {
	ctx := context.Background()
	srv, err := startServer(ctx, "Qwen", "Qwen2-0.5B-Instruct-GGUF", "qwen2-0_5b-instruct-q2_k.gguf")
	if err != nil {
		log.Print(err)
		return
	}
	defer func() { _ = srv.Close() }()
	c, err := llamacpp.New(ctx, genai.ProviderOptionRemote(srv.URL()))
	if err != nil {
		log.Print(err)
		return
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Say hello. Reply with only one word."),
	}
	opts := genai.GenOptionText{
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.GenSync(ctx, msgs, &opts, genai.GenOptionSeed(1))
	if err != nil {
		log.Print(err)
		return
	}
	log.Printf("Raw response: %#v", resp)
	if resp.Usage.InputTokens != 0 || resp.Usage.OutputTokens != 0 {
		log.Printf("Did I finally start filling the usage fields?")
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Requests[0].Text)), ".!")
	fmt.Printf("Response: %s\n", txt)
	// Disabled because it's slow in CI, especially on Windows.
	// // Output: Response: hello
}

//

func startServer(ctx context.Context, author, repo, modelfile string) (*llamacppsrv.Server, error) {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		return nil, err
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return nil, err
	}
	exe, err := llamacppsrv.DownloadRelease(ctx, cache, llamacppsrv.BuildNumber)
	if err != nil {
		return nil, err
	}
	extraArgs := []string{"-hf", author + "/" + repo, "-hff", modelfile, "--no-warmup", "--jinja", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"}
	l, err := os.Create(filepath.Join(cache, "llama-server.log"))
	if err != nil {
		return nil, err
	}
	defer func() { _ = l.Close() }()
	return llamacppsrv.New(ctx, exe, "", l, "", 0, extraArgs)
}
