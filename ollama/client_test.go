// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ollama_test

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/ollama"
	"github.com/maruel/httpjson/roundtrippers"
)

func TestNew(t *testing.T) {
	record := os.Getenv("RECORD") == "1"

	ctx := context.Background()
	srv, err := startServer(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	c, err := ollama.New(srv.URL(), "gemma3:1b")
	if err != nil {
		t.Fatal(err)
	}
	var records []roundtrippers.Record
	ch := make(chan roundtrippers.Record, 10)
	wg := sync.WaitGroup{}
	if record {
		c.Client.Client = &http.Client{Transport: &roundtrippers.Capture{Transport: http.DefaultTransport, C: ch}}
		wg.Add(1)
		defer func() {
			for r := range ch {
				records = append(records, r)
			}
			wg.Done()
		}()
	}

	msgs := genai.Messages{
		genai.NewTextMessage(genai.User, "Say hello. Use only one word."),
	}
	opts := genai.ChatOptions{
		Seed:        1,
		Temperature: 0.01,
		MaxTokens:   50,
	}
	resp, err := c.Chat(ctx, msgs, &opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolCalls) != 0 {
		t.Fatalf("got %d, want 0", len(resp.ToolCalls))
	}
	if len(resp.Contents) != 1 {
		t.Fatalf("got %d, want 1", len(resp.Contents))
	}
	if resp.Contents[0].Text != "Hello." {
		t.Fatalf("got %q, want %q", resp.Contents[0].Text, "Hello.")
	}

	if record && !t.Failed() {
		close(ch)
		wg.Wait()
		name := strings.ReplaceAll(t.Name(), "/", "_")
		b, err := json.Marshal(records)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile("testdata/"+name+".json", b, 0o666); err != nil {
			t.Fatal(err)
		}
	}
}

//
