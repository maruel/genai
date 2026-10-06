// Ask typed decision questions about a ticket using a local Kev-4B model.
//
// Downloads and starts llama-server, then stops it on exit. No API key is required.

package main

import (
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
	cache = filepath.Join(cache, "llama-server", llamacppsrv.Version)
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return err
	}
	exe, err := llamacppsrv.DownloadVersion(ctx, cache, llamacppsrv.Version)
	if err != nil {
		return err
	}
	// Only the Go client needs access; grant no browser origins cross-origin access.
	srv, err := llamacppsrv.New(ctx, exe, "", os.Stderr, "localhost:0", 0, []string{"-hf", "ggml-org/Kev-4B-GGUF", "--no-warmup", "--cors-origins", "", "--no-cors-credentials", "--no-ui", "--no-slots", "--parallel", "4", "--kv-unified"})
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, srv.Close()) }()
	c, err := llamacpp.New(ctx, genai.ProviderOptionRemote(srv.URL()))
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
	// Question names also identify the returned answers.
	q := genai.Questions{
		"billing": {Type: genai.QuestionNoul, Instructions: genai.Text("Is this request about billing?"), Noul: &genai.NoulCriteria{
			True:  genai.Text("the customer is asking about a charge or an invoice"),
			False: genai.Text("the customer is asking about anything else"),
		}},
		"tone": {Type: genai.QuestionChoice, Instructions: genai.Text("What is the tone of the customer?"), Choice: map[string]genai.DecisionContent{
			"calm":       nil,
			"frustrated": genai.Text("annoyed but polite"),
			"angry":      genai.Text("openly hostile"),
		}},
		"urgency": {Type: genai.QuestionScore, Instructions: genai.Text("How soon does this need to be handled?"), Score: []genai.DecisionContent{
			genai.Text("can wait"),
			genai.Text("this week"),
			genai.Text("today"),
			genai.Text("right now"),
		}},
	}
	pretty, err := json.MarshalIndent(ticket, "", "  ")
	if err != nil {
		return err
	}
	fmt.Printf("%s:\n%s\n", "State", pretty)
	start := time.Now()
	state, err := genai.ParseDecisionContent(raw)
	if err != nil {
		return err
	}
	res, err := c.SystemOne(ctx, &genai.SystemOneRequest{State: state, Questions: q})
	if err != nil {
		return err
	}
	elapsed := time.Since(start)

	// The answers are typed by the question they reply to, and keep the probability of each option and
	// each level.
	fmt.Println("Answers:")
	fmt.Printf("- billing: %.2f likely to be a yes\n", res.Answers["billing"].Noul)
	fmt.Printf("- tone:    %s with %.0f%% confidence\n", res.Answers["tone"].Choice, 100*res.Answers["tone"].Confidence)
	for _, name := range slices.Sorted(maps.Keys(res.Answers["tone"].Probabilities)) {
		fmt.Printf("    %s: %.2f\n", name, res.Answers["tone"].Probabilities[name])
	}
	fmt.Printf("- urgency: %.2f over %d levels with %.0f%% confidence\n", res.Answers["urgency"].Score, len(res.Answers["urgency"].Legend), 100*res.Answers["urgency"].Confidence)
	for _, level := range slices.Sorted(maps.Keys(res.Answers["urgency"].Probabilities)) {
		fmt.Printf("    %s (%v): %.2f\n", level, res.Answers["urgency"].Legend[level], res.Answers["urgency"].Probabilities[level])
	}
	fmt.Printf("in: %d, out: %d, total: %d\n", res.Usage.InputTokens, res.Usage.OutputTokens, res.Usage.InputTokens+res.Usage.OutputTokens)
	fmt.Printf("took %s\n", elapsed.Round(time.Millisecond))
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
