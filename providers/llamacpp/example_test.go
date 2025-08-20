// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacpp_test

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
	"github.com/maruel/huggingface"
)

func ExampleClient_GenSync() {
	// Download and start the server.
	ctx := context.Background()
	// Start a server with a minimalist model: Qwen2 0.5B in Q2_K quantization.
	srv, err := startServer(ctx, "Qwen", "Qwen2-0.5B-Instruct-GGUF", "qwen2-0_5b-instruct-q2_k.gguf", "")
	if err != nil {
		log.Print(err)
		return
	}
	defer srv.Close()
	// Connect the provider.
	c, err := llamacpp.New(ctx, &genai.ProviderOptions{Remote: srv.URL(), Model: genai.ModelNone}, nil)
	if err != nil {
		log.Print(err)
		return
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Say hello. Reply with only one word."),
	}
	opts := genai.OptionsText{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.GenSync(ctx, msgs, &opts)
	if err != nil {
		log.Print(err)
		return
	}
	log.Printf("Raw response: %#v", resp)
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.String())), ".!"))
	// Disabled because it's slow in CI, especially on Windows.
	// // Output: Response: hello
}

// startServer starts a server.
func startServer(ctx context.Context, author, repo, modelfile, multimodal string) (*llamacppsrv.Server, error) {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		return nil, err
	}
	if err = os.MkdirAll(cache, 0o755); err != nil {
		return nil, err
	}
	// It's a bit inefficient to download from github every single time.
	exe, err := llamacppsrv.DownloadRelease(ctx, cache, llamacppsrv.BuildNumber)
	if err != nil {
		return nil, err
	}
	// llama.cpp now knows how to pull from huggingface but this was not integrated yet, so pull a model
	// manually.
	hf, err := huggingface.New("")
	if err != nil {
		return nil, err
	}
	modelPath, err := hf.EnsureFile(ctx, huggingface.ModelRef{Author: author, Repo: repo}, "HEAD", modelfile)
	if err != nil {
		return nil, err
	}
	extraArgs := []string{"--no-warmup", "--jinja", "--flash-attn", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"}
	mmPath := ""
	if multimodal != "" {
		if mmPath, err = hf.EnsureFile(ctx, huggingface.ModelRef{Author: author, Repo: repo}, "HEAD", multimodal); err != nil {
			return nil, err
		}
		extraArgs = append(extraArgs, "--mmproj", mmPath)
	}
	l, err := os.Create(filepath.Join(cache, "llama-server.log"))
	if err != nil {
		return nil, err
	}
	defer l.Close()
	return llamacppsrv.New(ctx, exe, modelPath, l, "", 0, extraArgs)
}
