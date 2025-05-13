// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacpp_test

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/genai/llamacpp/llamacppsrv"
	"github.com/maruel/huggingface"
)

// buildNumber is the build number to use from
// https://github.com/ggml-org/llama.cpp/releases
const buildNumber = 4882

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

func ExampleClient_Chat() {
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		log.Print(err)
		return
	}
	defer srv.Close()
	c, err := llamacpp.New(srv.URL(), nil)
	if err != nil {
		log.Print(err)
		return
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Reply with only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(ctx, msgs, &opts)
	if err != nil {
		log.Print(err)
		return
	}
	log.Printf("Raw response: %#v", resp)
	if len(resp.Contents) != 1 {
		log.Print("Unexpected response")
		return
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"))
	// // Output: Response: hello
}

func ExampleClient_ChatStream() {
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		log.Print(err)
		return
	}
	defer srv.Close()
	c, err := llamacpp.New(srv.URL(), nil)
	if err != nil {
		log.Print(err)
		return
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Reply with only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	chunks := make(chan genai.MessageFragment)
	end := make(chan genai.Message, 10)
	go func() {
		var pendingMsgs genai.Messages
		defer func() {
			for _, m := range pendingMsgs {
				end <- m
			}
			close(end)
		}()
		for {
			select {
			case <-ctx.Done():
				return
			case pkt, ok := <-chunks:
				if !ok {
					return
				}
				var err2 error
				if pendingMsgs, err2 = pkt.Accumulate(pendingMsgs); err2 != nil {
					end <- genai.NewTextMessage(genai.Assistant, fmt.Sprintf("Error: %v", err2))
					return
				}
			}
		}
	}()
	_, err = c.ChatStream(ctx, msgs, &opts, chunks)
	close(chunks)
	var responses genai.Messages
	for m := range end {
		responses = append(responses, m)
	}
	log.Printf("Raw responses: %#v", responses)
	if err != nil {
		log.Print(err)
		return
	}
	if len(responses) != 1 {
		log.Print("Unexpected responses")
		return
	}
	resp := responses[0]
	if len(resp.Contents) != 1 {
		log.Print("Unexpected response")
		return
	}
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"))
	// // Output: Response: hello
}

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}
