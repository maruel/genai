// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Example usage of the TypeSafe provider.

package typesafe_test

import (
	"context"
	"encoding/json"
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
	if err != nil {
		log.Fatal(err)
	}

	// Ask three different kinds of questions about the same state. Each field is a question, the name the
	// answer comes back under is its json tag.
	q := struct {
		Billing typesafe.Noul   `json:"billing"`
		Tone    typesafe.Choice `json:"tone"`
		Urgency typesafe.Score  `json:"urgency"`
	}{
		Billing: typesafe.Noul{
			Instructions: typesafe.Text("Is this request about billing?"),
		},
		Tone: typesafe.Choice{
			Instructions: typesafe.Text("What is the tone of the customer?"),
			Criteria: map[string]typesafe.Content{
				"calm":       typesafe.Text("the customer is calm"),
				"frustrated": typesafe.Text("annoyed but polite"),
				"angry":      typesafe.Text("openly hostile"),
			},
		},
		Urgency: typesafe.Score{
			Instructions: typesafe.Text("How soon does this need to be handled?"),
			Criteria: []typesafe.Content{
				typesafe.Text("can wait"), typesafe.Text("this week"), typesafe.Text("today"), typesafe.Text("right now"),
			},
		},
	}
	res, err := c.GenSync(ctx, genai.Messages{genai.NewTextMessage("I was charged twice for order A-104, please refund the duplicate.")}, &genai.GenOptionText{DecodeAs: &q})
	if err != nil {
		log.Fatal(err)
	}
	// The same struct holds the answers.
	if err = res.Decode(&q); err != nil {
		log.Fatal(err)
	}
	// Every answer keeps the confidence and the probability of each option or level.
	fmt.Printf("billing: %.2f likely to be a yes\n", q.Billing.Probability)
	fmt.Printf("tone: %s with %.0f%% confidence\n", q.Tone.Label, 100*q.Tone.Confidence)
	for _, name := range slices.Sorted(maps.Keys(q.Tone.Probabilities)) {
		fmt.Printf("  %s: %.2f\n", name, q.Tone.Probabilities[name])
	}
	fmt.Printf("urgency: %.2f over %d levels with %.0f%% confidence\n", q.Urgency.Value, len(q.Urgency.Legend), 100*q.Urgency.Confidence)
	for _, level := range slices.Sorted(maps.Keys(q.Urgency.Probabilities)) {
		fmt.Printf("  %s (%v): %.2f\n", level, q.Urgency.Legend[level], q.Urgency.Probabilities[level])
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

// ExampleQuestionsFrom shows the questions a questionnaire struct declares, without sending a request.
func ExampleQuestionsFrom() {
	var q struct {
		Billing typesafe.Noul   `json:"billing"`
		Tone    typesafe.Choice `json:"tone"`
	}
	q.Billing.Instructions = typesafe.Text("Is this request about billing?")
	q.Tone.Instructions = typesafe.Text("What is the tone of the customer?")
	q.Tone.Criteria = map[string]typesafe.Content{"calm": nil, "angry": nil}
	questions, err := typesafe.QuestionsFrom(&q)
	if err != nil {
		log.Fatal(err)
	}
	raw, err := json.MarshalIndent(questions, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s\n", raw)
	// Output:
	// {
	//   "billing": {
	//     "type": "noul",
	//     "instructions": "Is this request about billing?"
	//   },
	//   "tone": {
	//     "type": "choice",
	//     "instructions": "What is the tone of the customer?",
	//     "criteria": {
	//       "angry": null,
	//       "calm": null
	//     }
	//   }
	// }
}
