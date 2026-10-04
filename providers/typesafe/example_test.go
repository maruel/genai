// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Example usage of the TypeSafe provider.

package typesafe_test

import (
	"context"
	"fmt"
	"log"
	"maps"
	"net/http"
	"os"
	"slices"

	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"

	"github.com/maruel/genai"
	"github.com/maruel/genai/httprecord"
	"github.com/maruel/genai/providers/typesafe"
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
	if os.Getenv("RECORD") == "all" {
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
	if os.Getenv("TYPESAFE_API_KEY") == "" {
		opts = append(opts, genai.ProviderOptionAPIKey("<insert_api_key_here>"))
	}
	ctx := context.Background()
	c, err := typesafe.New(ctx, append([]genai.ProviderOption{
		genai.ProviderOptionModel("jev-latest"),
		genai.ProviderOptionTransportWrapper(wrapper),
	}, opts...)...)
	if c != nil {
		defer func() {
			if err := c.Close(); err != nil {
				log.Printf("Close: %v", err)
			}
		}()
	}
	if err != nil {
		log.Print(err)
		return
	}

	// Ask three kinds of typed questions about the same state.
	q := genai.Questions{
		"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this request about billing?")},
		"tone": {Type: genai.QuestionChoice, Instructions: genai.Text("What is the tone of the customer?"), Choice: map[string]genai.DecisionContent{
			"calm":       genai.Text("the customer is calm"),
			"frustrated": genai.Text("annoyed but polite"),
			"angry":      genai.Text("openly hostile"),
		}},
		"urgency": {Type: genai.QuestionScore, Instructions: genai.Text("How soon does this need to be handled?"), Score: []genai.DecisionContent{
			genai.Text("can wait"), genai.Text("this week"), genai.Text("today"), genai.Text("right now"),
		}},
	}
	res, err := c.SystemOne(ctx, &genai.SystemOneRequest{State: genai.Text("I was charged twice for order A-104, please refund the duplicate."), Questions: q})
	if err != nil {
		log.Print(err)
		return
	}

	// Every answer keeps the confidence and the probability of each option or level.
	fmt.Printf("billing: %.2f likely to be a yes\n", res.Answers["billing"].Noul)
	fmt.Printf("tone: %s with %.0f%% confidence\n", res.Answers["tone"].Choice, 100*res.Answers["tone"].Confidence)
	for _, name := range slices.Sorted(maps.Keys(res.Answers["tone"].Probabilities)) {
		fmt.Printf("  %s: %.2f\n", name, res.Answers["tone"].Probabilities[name])
	}
	fmt.Printf("urgency: %.2f over %d levels with %.0f%% confidence\n", res.Answers["urgency"].Score, len(res.Answers["urgency"].Legend), 100*res.Answers["urgency"].Confidence)
	for _, level := range slices.Sorted(maps.Keys(res.Answers["urgency"].Probabilities)) {
		fmt.Printf("  %s (%v): %.2f\n", level, res.Answers["urgency"].Legend[level], res.Answers["urgency"].Probabilities[level])
	}
	// Output:
	// billing: 0.99 likely to be a yes
	// tone: calm with 84% confidence
	//   angry: 0.00
	//   calm: 0.89
	//   frustrated: 0.11
	// urgency: 1.80 over 4 levels with 61% confidence
	//   0 (can wait): 0.01
	//   1 (this week): 0.28
	//   2 (today): 0.62
	//   3 (right now): 0.09
}
