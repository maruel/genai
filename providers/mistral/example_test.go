// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package mistral_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/mistral"
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
		m := cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Request-Id"))
		var err error
		rr, err = recorder.New("testdata/example",
			recorder.WithHook(trimResponseHeaders, recorder.AfterCaptureHook),
			recorder.WithMode(mode),
			recorder.WithSkipRequestLatency(true),
			recorder.WithRealTransport(h),
			recorder.WithMatcher(m),
		)
		if err != nil {
			log.Fatal(err)
		}
		return rr
	}
	// When playing back the smoke test, no API key is needed. Insert a fake API key.
	apiKey := ""
	if os.Getenv("MISTRAL_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	ctx := context.Background()
	c, err := mistral.New(ctx, &genai.ProviderOptions{APIKey: apiKey, Model: genai.ModelNone}, wrapper)
	if err != nil {
		log.Fatal(err)
	}
	models, err := c.ListModels(ctx)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Found %d models\n", len(models))
	// Output:
	// Found 65 models
}

// trimResponseHeaders trims API key and noise from the recording.
func trimResponseHeaders(i *cassette.Interaction) error {
	// Do not save the API key in the recording.
	i.Request.Headers.Del("Authorization")
	i.Request.Headers.Del("X-Request-Id")
	i.Response.Headers.Del("Set-Cookie")
	// Reduce noise.
	i.Response.Headers.Del("Date")
	i.Response.Headers.Del("X-Kong-Request-Id")
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
}
