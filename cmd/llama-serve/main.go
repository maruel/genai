// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command llama-serve fetches a model from HuggingFace and runs llama-server.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/maruel/genai/llamacpp/llamacppsrv"
	"github.com/maruel/huggingface"
)

func mainImpl() error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	modelFlag := flag.String("model", "", "HuggingFace model reference (e.g., 'Qwen/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q6_K.gguf')")
	cacheDir := flag.String("cache", "", "Cache directory for models and server (default: ~/.cache/llama-serve)")
	port := flag.Int("port", 8080, "Port to serve on")
	threads := flag.Int("threads", 0, "Number of threads to use (default: CPU count - 2)")
	flag.Parse()
	if *modelFlag == "" {
		return fmt.Errorf("-model flag is required")
	}
	if *cacheDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			return err
		}
		*cacheDir = filepath.Join(home, ".cache", "llama-serve")
	}
	if err := os.MkdirAll(*cacheDir, 0o755); err != nil {
		return err
	}

	parts := strings.Split(*modelFlag, "/")
	if len(parts) < 2 {
		log.Fatalf("Invalid model format. Use 'Author/Repo' or 'Author/Repo/Filename'")
	}
	modelRef := huggingface.ModelRef{Author: parts[0], Repo: parts[1]}
	filename := ""
	if len(parts) > 2 {
		filename = parts[2]
	}

	log.Printf("Ensuring llama-server (build %d)...", llamacppsrv.BuildNumber)
	exe, err := llamacppsrv.DownloadRelease(ctx, *cacheDir, llamacppsrv.BuildNumber)
	if err != nil {
		return err
	}

	hf, err := huggingface.New("")
	if err != nil {
		return err
	}
	log.Printf("Ensuring model %s...", *modelFlag)
	modelPath, err := hf.EnsureFile(ctx, modelRef, "HEAD", filename)
	if err != nil {
		return err
	}

	log.Printf("Starting llama-server on port %d...", *port)
	server, err := llamacppsrv.NewServer(ctx, exe, modelPath, os.Stdout, *port, *threads, flag.Args())
	if err != nil {
		return err
	}
	log.Printf("Server running at %s", server.URL())
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	select {
	case <-sigChan:
		log.Println("Received signal, shutting down...")
		cancel()
	case err := <-server.Done():
		if err != nil {
			return err
		}
		log.Println("Server exited")
	}
	return server.Close()
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "llama-serve: %s\n", err)
		}
		os.Exit(1)
	}
}
