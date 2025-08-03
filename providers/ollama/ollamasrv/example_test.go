// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollamasrv_test

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

func Example() {
	// Download and start the server.
	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		log.Print(err)
		return
	}
	defer srv.Close()
	// Connect the provider.
	// Using small model for testing.
	c, err := ollama.New(srv.URL(), "qwen2.5:0.5b", nil)
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
	txt := strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.AsText())), ".!")
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
	return ollamasrv.NewServer(ctx, exe, l, port, []string{"OLLAMA_FLASH_ATTENTION=1", "OLLAMA_KV_CACHE_TYPE=q8_0"})
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
