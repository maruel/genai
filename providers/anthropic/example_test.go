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
	"time"

	"github.com/maruel/genai/providers/anthropic"
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
		m := cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("X-Api-Key"))
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
	c, err := anthropic.New("", "", wrapper)
	if err != nil {
		log.Fatal(err)
	}
	models, err := c.ListModels(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Found %d models\n", len(models))
	// Output:
	// Found 11 models
}

// trimResponseHeaders trims API key and noise from the recording.
func trimResponseHeaders(i *cassette.Interaction) error {
	// Do not save the API key in the recording.
	i.Request.Headers.Del("X-Api-Key")
	// Do not save the organization ID.
	i.Response.Headers.Del("Anthropic-Organization-Id")
	// Reduce noise.
	i.Response.Headers.Del("Date")
	i.Response.Headers.Del("Request-Id")
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
}
