// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package huggingface_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/maruel/genai"
	"github.com/maruel/genai/httprecord"
	"github.com/maruel/genai/providers/huggingface"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

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
	apiKey, err := getAPIKey()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	c, err := huggingface.New(ctx, genai.ProviderOptionTransportWrapper(wrapper), genai.ProviderOptionAPIKey(apiKey))
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

func getAPIKey() (string, error) {
	if v := os.Getenv("HUGGINGFACE_API_KEY"); v != "" {
		return v, nil
	}
	// Fallback to loading from the python client's cache.
	h, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("can't find home directory: %w", err)
	}
	tokenPath := filepath.Join(h, ".cache", "huggingface", "token")
	if data, err := os.ReadFile(tokenPath); err == nil {
		return string(data), nil
	}
	return "<insert_api_key_here>", nil
}
