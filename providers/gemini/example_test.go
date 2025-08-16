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
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/gemini"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
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

	wrapper := func(h http.RoundTripper) http.RoundTripper {
		// Simple trick to force recording via an environment variable.
		mode := recorder.ModeRecordOnce
		if os.Getenv("RECORD") == "1" {
			mode = recorder.ModeRecordOnly
		}
		// Remove API key when matching the request, so the playback doesn't need to have access to the API key.
		// Gemini is more complicated because we also need to mutate the URL.
		var err error
		rr, err = recorder.New("testdata/example",
			recorder.WithHook(trimRecording, recorder.AfterCaptureHook),
			recorder.WithMode(mode),
			recorder.WithSkipRequestLatency(true),
			recorder.WithRealTransport(h),
			recorder.WithMatcher(defaultMatcher),
		)
		if err != nil {
			log.Fatal(err)
		}
		return rr
	}
	// When playing back the smoke test, no API key is needed. Insert a fake API key.
	apiKey := ""
	if os.Getenv("GEMINI_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := gemini.New(&genai.OptionsProvider{APIKey: apiKey, Model: genai.ModelNone}, wrapper)
	if err != nil {
		log.Fatal(err)
	}
	models, err := c.ListModels(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Found %d models\n", len(models))
	// Output:
	// Found 57 models
}

// trimRecording trims API key and noise from the recording.
func trimRecording(i *cassette.Interaction) error {
	// Do not record API key.
	i.Request.Headers.Del("X-Goog-Api-Key")
	// OMG What are they thinking? This happens on HTTP 302 redirect when fetching Veo generated videos:
	i.Response.Headers.Del("X-Goog-Api-Key")
	// Reduce noise.
	i.Request.Headers.Del("X-Request-Id")
	i.Response.Headers.Del("Date")
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
}

var defaultMatcher = cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Goog-Api-Key", "X-Request-Id"))
