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

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/huggingface"
)

func Example() {
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		log.Print(err)
		return
	}
	defer srv.Close()
	c, err := llamacpp.New(srv.URL(), nil, nil)
	if err != nil {
		log.Print(err)
		return
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Reply with only one word."),
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
	if resp.InputTokens != 0 || resp.OutputTokens != 0 {
		log.Printf("Did I finally start filling the usage fields?")
	}
	if resp.Role != genai.Assistant || len(resp.Contents) != 1 {
		log.Print("Unexpected response")
		return
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	fmt.Printf("Response: %s\n", txt)
	// Disabled because it's slow in CI, especially on Windows.
	// // Output: Response: hello
}

//

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

// startServer starts a server with Qwen2 0.5B in Q2_K quantization.
func startServer(ctx context.Context) (*llamacppsrv.Server, error) {
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
	return llamacppsrv.NewServer(ctx, exe, modelPath, l, fmt.Sprintf("127.0.0.1:%d", port), 0, nil)
}
