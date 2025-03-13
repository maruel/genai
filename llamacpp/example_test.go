// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacpp_test

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/httpjson"
)

// qwen2.5-0.5b-instruct-q3_k_m.gguf works great and is only 412MiB.
var model = os.Getenv("LLAMACPP_MODEL")

func startServer(ctx context.Context) (string, <-chan error, func()) {
	// Make sure llama-server is in PATH and export LLAMACPP_MODEL to point to a
	// model path.
	if model == "" {
		return "", nil, func() {}
	}
	svr, _ := exec.LookPath("llama-server")
	if svr == "" {
		return "", nil, func() {}
	}
	p := strconv.Itoa(findFreePort())
	log.Printf("Starting llama-server on port %s with model %s", p, model)
	cmd := exec.CommandContext(ctx, svr, "--port", p, "--model", model)
	cmd.Dir = filepath.Dir(svr)
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}
	log.Printf("llama-server pid %d", cmd.Process.Pid)
	done := make(chan error, 1)
	go func() {
		err2 := cmd.Wait()
		log.Print("llama-server exited")
		done <- err2
		done <- nil
	}()
	return "http://localhost:" + p, done, func() {
		cmd.Process.Kill()
		<-done
	}
}

func ExampleClient_Completion() {
	ctx := context.Background()
	baseURL, done, cleanup := startServer(ctx)
	if baseURL == "" {
		// Make the test pass even if skipped.
		fmt.Println("Response: hello")
		return
	}
	defer cleanup()
	c, err := llamacpp.New(baseURL, nil)
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
	for {
		select {
		case <-done:
			return
		default:
		}
		resp, err := c.Completion(ctx, msgs, &opts)
		var v *url.Error
		if errors.As(err, &v) {
			continue
		}
		var h *httpjson.Error
		if errors.As(err, &h) && h.StatusCode == 503 {
			continue
		}
		if err != nil {
			fmt.Printf("Got error %T: %s\n", err, err)
			continue
		}
		log.Printf("Response: %#v", resp)
		txt := resp.Text
		if len(txt) < 2 || len(txt) > 100 {
			log.Fatalf("Unexpected response: %s", txt)
		}
		// Normalize some of the variance. Obviously many models will still fail this test.
		txt = strings.TrimSpace(txt)
		txt = strings.TrimRight(txt, ".!")
		txt = strings.ToLower(txt)
		fmt.Printf("Response: %s\n", txt)
		break
	}
	// Output: Response: hello
}

func ExampleClient_CompletionStream() {
	ctx := context.Background()
	baseURL, done, cleanup := startServer(ctx)
	if baseURL == "" {
		// Make the test pass even if skipped.
		fmt.Println("Response: hello")
		return
	}
	defer cleanup()
	c, err := llamacpp.New(baseURL, nil)
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
	opts := genaiapi.CompletionOptions{Seed: 1}
	for {
		select {
		case <-done:
			return
		default:
		}
		words := make(chan string, 10)
		result := make(chan string)
		go func() {
			resp := ""
			for {
				select {
				case <-ctx.Done():
					goto end
				case w, ok := <-words:
					if !ok {
						goto end
					}
					resp += w
				}
			}
		end:
			result <- resp
			close(result)
		}()
		err := c.CompletionStream(ctx, msgs, &opts, words)
		close(words)
		resp := <-result
		var v *url.Error
		if errors.As(err, &v) {
			continue
		}
		var h *httpjson.Error
		if errors.As(err, &h) && h.StatusCode == 503 {
			continue
		}
		if err != nil {
			fmt.Printf("Got error %T: %s\n", err, err)
		} else if len(resp) < 2 || len(resp) > 100 {
			fmt.Printf("Unexpected response: %s\n", resp)
		} else {
			// Normalize some of the variance. Obviously many models will still fail this test.
			resp = strings.TrimSpace(resp)
			resp = strings.TrimRight(resp, ".!")
			resp = strings.ToLower(resp)
			fmt.Printf("Response: %s\n", resp)
		}
		break
	}
	// Output: Response: hello
}

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}
