// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"log/slog"
	"net"
	"os"
	"path/filepath"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/ollama"
	"github.com/maruel/genai/providers/ollama/ollamasrv"
)

func ExampleClient_GenSync() {
	// Download and start the server.
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		log.Print(err)
		return
	}
	defer srv.Close()
	// Connect the provider.
	c, err := ollama.New(srv.URL(), "gemma3:1b", nil)
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
	// Normalize some of the variance. Obviously many models will still fail this test.
	fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.AsText())), ".!"))
	// Output: Response: hello
}

//

func startServer(ctx context.Context) (*ollamasrv.Server, error) {
	cache, err := filepath.Abs("testdata/tmp")
	if err != nil {
		return nil, err
	}
	if err = os.MkdirAll(cache, 0o755); err != nil {
		return nil, err
	}
	// It's a bit inefficient to download from github every single time.
	exe, err := ollamasrv.DownloadRelease(ctx, cache, ollamasrv.Version)
	if err != nil {
		return nil, err
	}
	port := findFreePort()
	// ollama doesn't log much, so redirect that to the logs instead of to a file. This permits not writing a
	// file, which can cause go test caching issues.
	l := &logWriter{slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			if a.Key == "level" || a.Key == "msg" || a.Key == "time" {
				a = slog.Attr{}
			}
			return a
		},
	}))}
	return ollamasrv.NewServer(ctx, exe, l, port)
}

type logWriter struct {
	logger *slog.Logger
}

func (w *logWriter) Write(p []byte) (n int, err error) {
	lines := bytes.Split(p, []byte("\n"))
	for i, line := range lines {
		// Skip the last empty line if the data ended with \n
		if i == len(lines)-1 && len(line) == 0 {
			continue
		}
		w.logger.Info("ollama", "ollama", string(line))
	}
	return len(p), nil
}

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}
