// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/httprecord"
	"github.com/maruel/genai/providers/gemini"
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
	// When playing back the smoke test, no API key is needed. Insert a fake API key.
	opts := []genai.ProviderOption{genai.ProviderOptionModel(genai.ModelNone)}
	if os.Getenv("GEMINI_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	ctx := context.Background()
	c, err := gemini.New(ctx, append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(wrapper)}, opts...)...)
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
