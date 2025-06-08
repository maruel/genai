// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package cloudflare_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"regexp"
	"time"

	"github.com/maruel/genai/providers/cloudflare"
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
		// Cloudflare is more complicated because we also need to mutate the URL.
		var err error
		rr, err = recorder.New("testdata/example",
			recorder.WithHook(trimRecording, recorder.AfterCaptureHook),
			recorder.WithMode(mode),
			recorder.WithSkipRequestLatency(true),
			recorder.WithRealTransport(h),
			recorder.WithMatcher(matchCassette),
		)
		if err != nil {
			log.Fatal(err)
		}
		return rr
	}
	// When playing back the smoke test, no API key is needed. Insert a fake API key.
	accountID := ""
	if os.Getenv("CLOUDFLARE_ACCOUNT_ID") == "" {
		accountID = "ACCOUNT_ID"
	}
	apiKey := ""
	if os.Getenv("CLOUDFLARE_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := cloudflare.New(accountID, apiKey, "", wrapper)
	if err != nil {
		log.Fatal(err)
	}
	models, err := c.ListModels(context.Background())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("Found %d models\n", len(models))
	// Output:
	// Found 68 models
}

var reAccount = regexp.MustCompile(`/accounts/[0-9a-fA-F]{32}/`)

// trimRecording trims API key and noise from the recording.
func trimRecording(i *cassette.Interaction) error {
	// Zap the account ID from the URL path before saving.
	i.Request.URL = reAccount.ReplaceAllString(i.Request.URL, "/accounts/ACCOUNT_ID/")
	// Do not save the API key in the recording.
	i.Request.Headers.Del("Authorization")
	i.Request.Headers.Del("X-Request-Id")
	i.Response.Headers.Del("Set-Cookie")
	// Reduce noise.
	i.Response.Headers.Del("Date")
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
}

func matchCassette(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	// When matching, ignore the account ID from the URL path.
	r.URL.Path = reAccount.ReplaceAllString(r.URL.Path, "/accounts/ACCOUNT_ID/")
	return defaultMatcher(r, i)
}

var defaultMatcher = cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Request-Id"))
