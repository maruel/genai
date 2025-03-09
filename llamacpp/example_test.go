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

func ExampleClient_Completion() {
	// Make sure llama-server is in PATH and export LLAMACPP_MODEL to point to a
	// model path.
	if model == "" {
		// Make the test pass even if skipped.
		fmt.Println("Response: hi")
		return
	}
	svr, _ := exec.LookPath("llama-server")
	if svr == "" {
		// Make the test pass even if skipped.
		fmt.Println("Response: hi")
		return
	}
	p := strconv.Itoa(findFreePort())
	cmd := exec.Command(svr, "--port", p, "--model", model)
	cmd.Dir = filepath.Dir(svr)
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()
	c := llamacpp.Client{BaseURL: "http://localhost:" + p}
	ctx := context.Background()
	msgs := []genaiapi.Message{
		{Role: genaiapi.User, Content: "Say hi. Use only one two letters word."},
	}
	opts := genaiapi.CompletionOptions{Seed: 1}
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
		} else if len(resp) < 2 || len(resp) > 100 {
			fmt.Printf("Unexpected response: %s\n", resp)
		} else {
			// Normalize some of the variance. Obviously many models will still fail this test.
			resp = strings.TrimSpace(resp)
			resp = strings.TrimRight(resp, ".")
			resp = strings.ToLower(resp)
			resp = strings.ReplaceAll(resp, "hello", "hi")
			resp = strings.ReplaceAll(resp, "hey", "hi")
			fmt.Printf("Response: %s\n", resp)
		}
		break
	}
	cmd.Process.Kill()
	<-done
	// Output: Response: hi
}

func ExampleClient_CompletionStream() {
	// Make sure llama-server is in PATH and export LLAMACPP_MODEL to point to a
	// model path.
	if model == "" {
		// Make the test pass even if skipped.
		fmt.Println("Response: hi")
		return
	}
	svr, _ := exec.LookPath("llama-server")
	if svr == "" {
		// Make the test pass even if skipped.
		fmt.Println("Response: hi")
		return
	}
	p := strconv.Itoa(findFreePort())
	cmd := exec.Command(svr, "--port", p, "--model", model)
	cmd.Dir = filepath.Dir(svr)
	if err := cmd.Start(); err != nil {
		log.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		done <- cmd.Wait()
	}()
	c := llamacpp.Client{BaseURL: "http://localhost:" + p}
	ctx := context.Background()
	msgs := []genaiapi.Message{
		{Role: genaiapi.User, Content: "Say hi. Use only one word."},
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
			resp = strings.TrimRight(resp, ".")
			resp = strings.ToLower(resp)
			resp = strings.ReplaceAll(resp, "hello", "hi")
			resp = strings.ReplaceAll(resp, "hey", "hi")
			fmt.Printf("Response: %s\n", resp)
		}
		break
	}
	cmd.Process.Kill()
	<-done
	// Output: Response: hi
}

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}
