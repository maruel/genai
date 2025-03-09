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

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/llamacpp"
	"github.com/maruel/httpjson"
)

func ExampleClient_Completion() {
	// Print something so the example runs.
	fmt.Println("Hello, world!")
	// Make sure llama-server is in PATH and export LLAMACPP_MODEL to point to a
	// model path.
	mdl := os.Getenv("LLAMACPP_MODEL")
	if mdl == "" {
		return
	}
	svr, _ := exec.LookPath("llama-server")
	if svr == "" {
		return
	}
	p := strconv.Itoa(findFreePort())
	cmd := exec.Command(svr, "--port", p, "--model", mdl, "-ngl", "9999")
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
		{Role: genaiapi.User, Content: "Say hello. Use only one word."},
	}
	opts := genaiapi.CompletionOptions{}
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
		}
		break
	}
	cmd.Process.Kill()
	<-done
	// Output: Hello, world!
}

func findFreePort() int {
	l, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		log.Fatal(err)
	}
	defer l.Close()
	return l.Addr().(*net.TCPAddr).Port
}
