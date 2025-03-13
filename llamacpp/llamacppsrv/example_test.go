// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacppsrv_test

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/genai/llamacpp/llamacppsrv"
	"github.com/maruel/huggingface"
)

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// startServer starts a server with Qwen2 0.5B in Q2_K quantization.
func startServer(ctx context.Context) (*llamacppsrv.Server, error) {
	const buildNumber = 4882
	cache, err := filepath.Abs("tmp")
	if err != nil {
		return nil, err
	}
	if err = os.MkdirAll(cache, 0o755); err != nil {
		return nil, err
	}
	// It's a bit inefficient to download from github every single time.
	exe, err := llamacppsrv.DownloadRelease(ctx, cache, buildNumber)
	if err != nil {
		return nil, err
	}
	port := findFreePort()
	hf, err := huggingface.New("")
	if err != nil {
		return nil, err
	}
	// A really small model.
	modelPath, err := hf.EnsureFile(ctx, huggingface.ModelRef{Author: "Qwen", Repo: "Qwen2-0.5B-Instruct-GGUF"}, "HEAD", "qwen2-0_5b-instruct-q2_k.gguf")
	if err != nil {
		return nil, err
	}
	l, err := os.Create(filepath.Join(cache, "lllama-server.log"))
	if err != nil {
		return nil, err
	}
	defer l.Close()
	return llamacppsrv.NewServer(ctx, exe, modelPath, l, port, 0, nil)
}

func Example() {
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer srv.Close()
	c, err := llamacpp.New(srv.URL(), nil)
	if err != nil {
		log.Fatal(err)
	}
	msgs := []genaiapi.Message{
		{
			Role: genaiapi.User,
			Type: genaiapi.Text,
			Text: "Say hello. Reply with only one word.",
		},
	}
	opts := genaiapi.CompletionOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Completion(ctx, msgs, &opts)
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Text)), ".!")
	fmt.Printf("Response: %s\n", txt)
	// Output: Response: hello
}
