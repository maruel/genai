// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollamasrv_test

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/ollama"
	"github.com/maruel/genai/ollama/ollamasrv"
)

// Ollama build to use. Do not specify a version to force running the latest.
const version = ""

// Using small model for testing.
const model = "gemma3:1b"

func Example() {
	// Download and start the server.
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		log.Print(err)
		return
	}
	defer srv.Close()
	// Connect the client.
	c, err := ollama.New(srv.URL(), model)
	if err != nil {
		log.Print(err)
		return
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Reply with only one word."),
	}
	opts := genai.CompletionOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Completion(ctx, msgs, &opts)
	if err != nil {
		log.Print(err)
		return
	}
	log.Printf("Raw response: %#v", resp)
	if resp.InputTokens != 18 || resp.OutputTokens != 3 {
		log.Printf("Unexpected tokens usage: %v", resp.Usage)
	}
	if resp.Role != genai.Assistant || len(resp.Contents) != 1 {
		log.Print("Unexpected response")
		return
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!")
	fmt.Printf("Response: %s\n", txt)
	// Output: Response: hello
}

//

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		panic(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}

func startServer(ctx context.Context) (*ollamasrv.Server, error) {
	cache, err := filepath.Abs("tmp")
	if err != nil {
		return nil, err
	}
	if err = os.MkdirAll(cache, 0o755); err != nil {
		return nil, err
	}
	// It's a bit inefficient to download from github every single time.
	exe, err := ollamasrv.DownloadRelease(ctx, cache, version)
	if err != nil {
		return nil, err
	}
	port := findFreePort()
	l, err := os.Create(filepath.Join(cache, "ollama.log"))
	if err != nil {
		return nil, err
	}
	defer l.Close()
	return ollamasrv.NewServer(ctx, exe, l, port)
}
