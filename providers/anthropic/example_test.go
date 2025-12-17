// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package anthropic_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/httprecord"
	"github.com/maruel/genai/providers/anthropic"
	"github.com/maruel/roundtrippers"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func ExampleNew_mCP_client() {
	// Anthropic requires an HTTP header to enable MCP use. See
	// https://docs.anthropic.com/en/docs/agents-and-tools/mcp-connector
	wrapper := func(h http.RoundTripper) http.RoundTripper {
		return &roundtrippers.Header{
			Transport: h,
			Header:    http.Header{"anthropic-beta": []string{"mcp-client-2025-04-04"}},
		}
	}
	ctx := context.Background()
	c, err := anthropic.New(ctx, &genai.ProviderOptions{}, wrapper)
	if err != nil {
		log.Fatal(err)
	}

	msgs := genai.Messages{
		genai.NewTextMessage("Remember that my name is Bob, my mother is Jane and my father is John."),
	}
	// Use raw calls to use the MCP client. It is not yet generalized in genai.
	in := anthropic.ChatRequest{}
	if err = in.Init(msgs, c.ModelID()); err != nil {
		log.Fatal(err)
	}
	// Use your own MCP server. This one runs a modified version of
	// https://github.com/modelcontextprotocol/go-sdk/tree/main/examples/server/memory for testing purposes.
	in.MCPServers = []anthropic.MCPServer{
		{
			Name:              "memory",
			Type:              "url",
			URL:               "https://mcp.maruel.ca",
			ToolConfiguration: anthropic.ToolConfiguration{Enabled: true},
		},
	}
	out := anthropic.ChatResponse{}
	if err = c.GenSyncRaw(ctx, &in, &out); err != nil {
		log.Fatal(err)
	}
	res, err := out.ToResult()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Reply: %s\n", res.String())
}

func ExampleNew_hTTP_record() {
	// Example to do HTTP recording and playback for smoke testing.
	// The example recording is in testdata/example.yaml.
	var rr *recorder.Recorder
	defer func() {
		// In a smoke test, use t.Cleanup().
		if rr != nil {
			if err := rr.Stop(); err != nil {
				log.Printf("Failed saving recordings: %v", err)
			}
		}
	}()

	// Simple trick to force recording via an environment variable.
	mode := recorder.ModeRecordOnce
	if os.Getenv("RECORD") == "1" {
		mode = recorder.ModeRecordOnly
	}
	wrapper := func(h http.RoundTripper) http.RoundTripper {
		var err error
		rr, err = httprecord.New("testdata/example", h, recorder.WithMode(mode))
		if err != nil {
			log.Fatal(err)
		}
		return rr
	}
	// When playing back the smoke test, no API key is needed. Insert a fake API key.
	apiKey := ""
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	ctx := context.Background()
	c, err := anthropic.New(ctx, &genai.ProviderOptions{APIKey: apiKey, Model: genai.ModelNone}, wrapper)
	if err != nil {
		log.Fatal(err)
	}
	models, err := c.ListModels(ctx)
	if err != nil {
		log.Fatal(err)
	}
	if len(models) > 1 {
		fmt.Println("Found multiple models")
	}
	// Output:
	// Found multiple models
}
