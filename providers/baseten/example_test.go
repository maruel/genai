// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package baseten_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"

	"github.com/maruel/genai"
	"github.com/maruel/genai/httprecord"
	"github.com/maruel/genai/providers/baseten"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

func ExampleNew_hTTP_record() {
	// Example to do HTTP recording and playback for smoke testing.
	// The example recording is in testdata/example.yaml.
	var rr *recorder.Recorder
	defer func() {
		if rr != nil {
			if err := rr.Stop(); err != nil {
				log.Printf("Failed saving recordings: %v", err)
			}
		}
	}()

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
	var opts []genai.ProviderOption
	if os.Getenv("BASETEN_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	ctx := context.Background()
	c, err := baseten.New(ctx, append([]genai.ProviderOption{genai.ProviderOptionTransportWrapper(wrapper)}, opts...)...)
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
