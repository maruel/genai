// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package gemini_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
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
			recorder.WithMatcher(matchCassette),
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
	c, err := gemini.New(&genai.OptionsProvider{APIKey: apiKey}, wrapper)
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
	// Gemini pass the API key as a query argument (!) so zap it before recording.
	u, err := url.Parse(i.Request.URL)
	if err != nil {
		return err
	}
	u.RawQuery = removeKeyFromQuery(u.RawQuery, "key")
	i.Request.URL = u.String()
	i.Request.Form.Del("key")
	// Reduce noise.
	i.Request.Headers.Del("X-Request-Id")
	i.Response.Headers.Del("Date")
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
}

// matchCassette matches the cassette ignoring the query argument
func matchCassette(r *http.Request, i cassette.Request) bool {
	// Gemini pass the API key as a query argument (!) so zap it before matching.
	r = r.Clone(r.Context())
	r.URL.RawQuery = removeKeyFromQuery(r.URL.RawQuery, "key")
	_ = r.ParseForm()
	return defaultMatcher(r, i)
}

var defaultMatcher = cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Request-Id"))

// removeKeyFromQuery remove a key from a raw query.
// Using url.URL.Query() then Encode() reorders the keys, which makes it non-deterministic. Do it manually.
func removeKeyFromQuery(query, keyToRemove string) string {
	b := strings.Builder{}
	for _, part := range strings.Split(query, "&") {
		if part != "" {
			if k := strings.SplitN(part, "=", 2)[0]; k == keyToRemove {
				continue
			}
		}
		if b.Len() != 0 {
			b.WriteByte('&')
		}
		b.WriteString(part)
	}
	return b.String()
}
