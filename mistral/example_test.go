// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	"bytes"
	"context"
	_ "embed"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/maruel/genai/genaiapi"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/httpjson"
)

// See the 1kib banana jpg online at
// https://github.com/maruel/genai/blob/main/mistral/testdata/banana.jpg
//
//go:embed testdata/banana.jpg
var bananaJpg []byte

func ExampleClient_Completion_vision_and_JSON() {
	// This code will run when MISTRAL_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// Require a model which has the "vision" capability.
	// https://docs.mistral.ai/capabilities/vision/
	if c, err := mistral.New("", "mistral-small-latest"); err == nil {
		msgs := genaiapi.Messages{
			{
				Role: genaiapi.User,
				Contents: []genaiapi.Content{
					{Text: "Is it a banana? Reply as JSON."},
					{Filename: "banana.jpg", Document: bytes.NewReader(bananaJpg)},
				},
			},
		}
		var got struct {
			Banana bool `json:"banana"`
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
			DecodeAs:    &got,
		}
		var resp genaiapi.CompletionResult
		for i := range 3 {
			// Mistral has a very good rate limiting implementation.
			if resp, err = c.Completion(context.Background(), msgs, &opts); err != nil && i != 2 {
				var herr *httpjson.Error
				if errors.As(err, &herr) {
					if herr.StatusCode == http.StatusTooManyRequests {
						fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
						time.Sleep(2 * time.Second)
						continue
					}
				}
				log.Fatal(err)
			}
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 67 || resp.OutputTokens != 9 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		if err := resp.Contents[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Banana: %v\n", got.Banana)
	} else {
		// Print something so the example runs.
		fmt.Println("Banana: true")
	}
	// Output: Banana: true
}

func ExampleClient_Completion_pDF() {
	// This code will run when MISTRAL_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// Require a model which has the "OCR" or the "Document understanding"
	// capability. There's a subtle difference between the two; from what I
	// understand, the document understanding will only parse the text, while the
	// OCR will try to understand the pictures.
	// https://docs.mistral.ai/capabilities/document/
	// https://docs.mistral.ai/capabilities/vision/
	if c, err := mistral.New("", "mistral-small-latest"); err == nil {
		msgs := genaiapi.Messages{
			{
				Role: genaiapi.User,
				Contents: []genaiapi.Content{
					{Text: "What is the word? Reply with only the word."},
					{URL: "https://raw.githubusercontent.com/maruel/genai/refs/heads/main/mistral/testdata/hidden_word.pdf"},
				},
			},
		}
		opts := genaiapi.CompletionOptions{
			Temperature: 0.01,
			MaxTokens:   50,
		}
		var resp genaiapi.CompletionResult
		for i := range 3 {
			// Mistral has a very good rate limiting implementation.
			if resp, err = c.Completion(context.Background(), msgs, &opts); err != nil && i != 2 {
				var herr *httpjson.Error
				if errors.As(err, &herr) {
					if herr.StatusCode == http.StatusTooManyRequests {
						fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
						time.Sleep(2 * time.Second)
						continue
					}
				}
				log.Fatal(err)
			}
			break
		}
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("Raw response: %#v", resp)
		// Mistral is super efficient with tokens for PDFs.
		if resp.InputTokens != 28 || resp.OutputTokens != 1 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.Contents) != 1 {
			log.Fatal("Unexpected response")
		}
		fmt.Printf("Hidden word in PDF: %v\n", strings.ToLower(resp.Contents[0].Text))
	} else {
		// Print something so the example runs.
		fmt.Println("Hidden word in PDF: orange")
	}
	// Output: Hidden word in PDF: orange
}

func ExampleClient_Completion_tool_use() {
	// This code will run when MISTRAL_API_KEY is set.
	//
	// As of March 2025, you can try it out for free.
	//
	// Require a model which has the tool capability. See
	// https://docs.mistral.ai/capabilities/function_calling/
	if c, err := mistral.New("", "ministral-3b-latest"); err == nil {
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "I wonder if Canada is a better country than the US? Call the tool best_country to tell me which country is the best one."),
		}
		var got struct {
			Country string `json:"country" jsonschema:"enum=Canada,enum=USA"`
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   200,
			Tools: []genaiapi.ToolDef{
				{
					Name:        "best_country",
					Description: "A tool to determine the best country",
					InputsAs:    &got,
				},
			},
		}
		var resp genaiapi.CompletionResult
		for i := range 3 {
			// Mistral has a very good rate limiting implementation.
			if resp, err = c.Completion(context.Background(), msgs, &opts); err != nil && i != 2 {
				var herr *httpjson.Error
				if errors.As(err, &herr) {
					if herr.StatusCode == http.StatusTooManyRequests {
						fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
						time.Sleep(2 * time.Second)
						continue
					}
				}
				log.Fatal(err)
			}
			break
		}
		log.Printf("Raw response: %#v", resp)
		if resp.InputTokens != 129 || resp.OutputTokens != 19 {
			log.Printf("Unexpected tokens usage: %v", resp.Usage)
		}
		if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].Name != "best_country" {
			log.Fatal("Unexpected response")
		}
		if err := resp.ToolCalls[0].Decode(&got); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("Best: %v\n", got.Country)
	} else {
		// Print something so the example runs.
		fmt.Println("Best: Canada")
	}
	// Output: Best: Canada
}

func ExampleClient_CompletionStream() {
	// This code will run when MISTRAL_API_KEY is set.
	// As of March 2025, you can try it out for free.
	// Using very small model for testing.
	// See https://docs.mistral.ai/getting-started/models/models_overview/
	if c, err := mistral.New("", "ministral-3b-latest"); err == nil {
		ctx := context.Background()
		msgs := genaiapi.Messages{
			genaiapi.NewTextMessage(genaiapi.User, "Say hello. Use only one word."),
		}
		opts := genaiapi.CompletionOptions{
			Seed:        1,
			Temperature: 0.01,
			MaxTokens:   50,
		}
		for i := range 3 {
			chunks := make(chan genaiapi.MessageFragment)
			end := make(chan genaiapi.Message, 10)
			go func() {
				var pendingMsgs genaiapi.Messages
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
						if pendingMsgs, err = pkt.Accumulate(pendingMsgs); err != nil {
							end <- genaiapi.NewTextMessage(genaiapi.Assistant, fmt.Sprintf("Error: %v", err))
							return
						}
					}
				}
			}()
			err := c.CompletionStream(ctx, msgs, &opts, chunks)
			close(chunks)
			var responses genaiapi.Messages
			for m := range end {
				responses = append(responses, m)
			}
			if err != nil && i != 2 {
				// Mistral has a very good rate limiting implementation.
				var herr *httpjson.Error
				if errors.As(err, &herr) {
					if herr.StatusCode == http.StatusTooManyRequests {
						fmt.Fprintf(os.Stderr, "Rate limited, waiting 2s\n")
						time.Sleep(2 * time.Second)
						continue
					}
				}
				log.Fatal(err)
			}
			log.Printf("Raw responses: %#v", responses)
			if err != nil {
				log.Fatal(err)
			}
			if len(responses) != 1 {
				log.Fatal("Unexpected response")
			}
			resp := responses[0]
			if len(resp.Contents) != 1 {
				log.Fatal("Unexpected response")
			}
			// Normalize some of the variance. Obviously many models will still fail this test.
			fmt.Printf("Response: %s\n", strings.TrimRight(strings.TrimSpace(strings.ToLower(resp.Contents[0].Text)), ".!"))
			break
		}
	} else {
		// Print something so the example runs.
		fmt.Println("Response: hello")
	}
	// Output: Response: hello
}

func ExampleClient_ListModels() {
	// Print something so the example runs.
	fmt.Println("Got models")
	if c, err := mistral.New("", ""); err == nil {
		models, err := c.ListModels(context.Background())
		if err != nil {
			fmt.Printf("Failed to get models: %v\n", err)
			return
		}
		for _, model := range models {
			// The list of models will change over time. Print them to stderr so the
			// test doesn't capture them.
			fmt.Fprintf(os.Stderr, "- %s\n", model)
		}
	}
	// Output: Got models
}
