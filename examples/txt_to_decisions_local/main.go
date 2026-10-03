// Ask typed decision questions about a ticket using a local Kev-4B model.
//
// Downloads and starts llama-server, then stops it on exit. No API key is required.

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"maps"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"syscall"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/llamacpp"
	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
)

func mainImpl() (err error) {
	subject := flag.String("subject", "Charged twice this month", "Ticket subject")
	text := flag.String("text", "Hi, I see two charges of $49 on my card for August. I only have one account. Please fix this ASAP, I'm pretty frustrated.", "Ticket text")
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	cache, err := os.UserCacheDir()
	if err != nil {
		return err
	}
	// TODO: Remove this override and use llamacppsrv.Version once a stable release newer than v0.5.0
	// is available.
	const version = "b11361"
	cache = filepath.Join(cache, "llama-server", version)
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return err
	}
	exe, err := llamacppsrv.DownloadVersion(ctx, cache, version)
	if err != nil {
		return err
	}
	// Only the Go client needs access; grant no browser origins cross-origin access.
	srv, err := llamacppsrv.New(ctx, exe, "", os.Stderr, "localhost:0", 0, []string{"-hf", "ggml-org/Kev-4B-GGUF", "--no-warmup", "--cors-origins", "", "--no-cors-credentials", "--no-ui", "--no-slots", "--parallel", "4", "--kv-unified"})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, srv.Close()) }()
	c, err := llamacpp.New(ctx, genai.ProviderOptionRemote(srv.URL()), &llamacpp.ProviderOption{SystemOne: true})
	if c != nil {
		defer func() { err = errors.Join(err, c.Close()) }()
	}
	if err != nil {
		return err
	}
	// The state is the material to judge, the ticket and its subject. It is passed as a document with a
	// JSON media type; text messages are accepted as is.
	ticket := map[string]string{
		"subject": *subject,
		"body":    *text,
	}
	raw, err := json.Marshal(ticket)
	if err != nil {
		return err
	}
	// Each field is a question, the name the answer comes back under is its json tag.
	q := struct {
		// A noul is a yes/no question, answered by the probability that the answer is yes.
		Billing llamacpp.Noul `json:"billing"`
		// A choice picks one of the options, each mapped to its description, or nil when the name is self
		// explanatory.
		Tone llamacpp.Choice `json:"tone"`
		// A score rates the state along an ordered rubric, the levels start at 0.
		Urgency llamacpp.Score `json:"urgency"`
	}{
		Billing: llamacpp.Noul{
			Instructions: llamacpp.Text("Is this request about billing?"),
			Criteria: &llamacpp.NoulCriteria{
				True:  llamacpp.Text("the customer is asking about a charge or an invoice"),
				False: llamacpp.Text("the customer is asking about anything else"),
			},
		},
		Tone: llamacpp.Choice{
			Instructions: llamacpp.Text("What is the tone of the customer?"),
			Criteria: map[string]llamacpp.DecisionContent{
				"calm":       nil,
				"frustrated": llamacpp.Text("annoyed but polite"),
				"angry":      llamacpp.Text("openly hostile"),
			},
		},
		Urgency: llamacpp.Score{
			Instructions: llamacpp.Text("How soon does this need to be handled?"),
			Criteria: []llamacpp.DecisionContent{
				llamacpp.Text("can wait"),
				llamacpp.Text("this week"),
				llamacpp.Text("today"),
				llamacpp.Text("right now"),
			},
		},
	}
	// Print the state being judged. QuestionsFrom(&q) returns the questions these fields declare as
	// llamacpp.Questions, to review or tune them.
	pretty, err := json.MarshalIndent(ticket, "", "  ")
	if err != nil {
		return err
	}
	fmt.Printf("%s:\n%s\n", "State", pretty)
	start := time.Now()
	res, err := c.GenSync(ctx, genai.Messages{genai.Message{Requests: []genai.Request{{
		Doc: genai.Doc{Filename: "ticket.json", Src: bytes.NewReader(raw)},
	}}}}, &genai.GenOptionText{DecodeAs: &q})
	if err != nil {
		return err
	}
	elapsed := time.Since(start)
	if err = res.Decode(&q); err != nil {
		return err
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
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
