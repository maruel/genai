// Ask a yes/no, a choice and a score question about one state at once.
//
// This requires `TYPESAFE_API_KEY` (https://console.typesafe.ai/settings/keys) environment variable to
// authenticate.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"maps"
	"slices"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/typesafe"
)

func main() {
	ctx := context.Background()
	c, err := typesafe.New(ctx, genai.ModelGood)
	if err != nil {
		log.Fatal(err)
	}
	// The state is the material to judge, the ticket and its subject. It is passed as a document with a
	// JSON media type; text messages are accepted as is.
	ticket := map[string]string{
		"subject": "Charged twice this month",
		"body":    "Hi, I see two charges of $49 on my card for August. I only have one account. Please fix this ASAP, I'm pretty frustrated.",
	}
	raw, err := json.Marshal(ticket)
	if err != nil {
		log.Fatal(err)
	}
	// Each field is a question, the name the answer comes back under is its json tag.
	q := struct {
		// A noul is a yes/no question, answered by the probability that the answer is yes.
		Billing typesafe.Noul `json:"billing"`
		// A choice picks one of the options, each mapped to its description, or nil when the name is self
		// explanatory.
		Tone typesafe.Choice `json:"tone"`
		// A score rates the state along an ordered rubric, the levels start at 0.
		Urgency typesafe.Score `json:"urgency"`
	}{
		Billing: typesafe.Noul{
			Instructions: typesafe.Text("Is this request about billing?"),
			Criteria: &typesafe.NoulCriteria{
				True:  typesafe.Text("the customer is asking about a charge or an invoice"),
				False: typesafe.Text("the customer is asking about anything else"),
			},
		},
		Tone: typesafe.Choice{
			Instructions: typesafe.Text("What is the tone of the customer?"),
			Criteria: map[string]typesafe.Content{
				"calm":       nil,
				"frustrated": typesafe.Text("annoyed but polite"),
				"angry":      typesafe.Text("openly hostile"),
			},
		},
		Urgency: typesafe.Score{
			Instructions: typesafe.Text("How soon does this need to be handled?"),
			Criteria: []typesafe.Content{
				typesafe.Text("can wait"),
				typesafe.Text("this week"),
				typesafe.Text("today"),
				typesafe.Text("right now"),
			},
		},
	}
	// Print the state being judged. QuestionsFrom(&q) returns the questions these fields declare as
	// typesafe.Questions, to review or tune them.
	pretty, err := json.MarshalIndent(ticket, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("%s:\n%s\n", "State", pretty)
	start := time.Now()
	res, err := c.GenSync(ctx, genai.Messages{genai.Message{Requests: []genai.Request{{
		Doc: genai.Doc{Filename: "ticket.json", Src: bytes.NewReader(raw)},
	}}}}, &genai.GenOptionText{DecodeAs: &q})
	if err != nil {
		log.Fatal(err)
	}
	elapsed := time.Since(start)
	if err = res.Decode(&q); err != nil {
		log.Fatal(err)
	}
	// The answers are typed by the question they reply to, and keep the probability of each option and
	// each level.
	fmt.Println("Answers:")
	fmt.Printf("- billing: %.2f likely to be a yes\n", q.Billing.Probability)
	fmt.Printf("- tone:    %s with %.0f%% confidence\n", q.Tone.Label, 100*q.Tone.Confidence)
	for _, name := range slices.Sorted(maps.Keys(q.Tone.Probabilities)) {
		fmt.Printf("    %s: %.2f\n", name, q.Tone.Probabilities[name])
	}
	fmt.Printf("- urgency: %.2f over %d levels with %.0f%% confidence\n", q.Urgency.Value, len(q.Urgency.Legend), 100*q.Urgency.Confidence)
	for _, level := range slices.Sorted(maps.Keys(q.Urgency.Probabilities)) {
		fmt.Printf("    %s (%v): %.2f\n", level, q.Urgency.Legend[level], q.Urgency.Probabilities[level])
	}
	fmt.Printf("in: %d, out: %d, total: %d\n", res.Usage.InputTokens, res.Usage.OutputTokens, res.Usage.TotalTokens)
	fmt.Printf("took %s\n", elapsed.Round(time.Millisecond))
}
