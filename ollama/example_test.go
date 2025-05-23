// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

import (
	"context"
	_ "embed"
	"fmt"
	"log"
	"net"
	"os"
	"path/filepath"

	"github.com/maruel/genai"
	"github.com/maruel/genai/ollama"
	"github.com/maruel/genai/ollama/ollamasrv"
)

// Ollama build to use.
const version = "v0.7.0"

func ExampleClient_ChatStream() {
	// Download and start the server.
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		log.Print(err)
		return
	}
	defer srv.Close()
	// Connect the client.
	c, err := ollama.New(srv.URL(), "gemma3:1b")
	if err != nil {
		log.Print(err)
		return
	}
	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
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
	fmt.Println(resp.AsText())
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
	cache, err := filepath.Abs("testdata/tmp")
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
