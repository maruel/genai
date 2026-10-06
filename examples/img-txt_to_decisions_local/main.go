// Ask typed decision questions about an image and its text context.
//
// Downloads and starts a local OpenJev vision decision model, then stops the server on exit.
//
// Uses the bundled sample image by default, or pass -image with a local image filename.

package main

import (
	"bytes"
	"context"
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
	"github.com/maruel/genai/scoreboard"
)

func mainImpl() (err error) {
	image := flag.String("image", "", "Local image to evaluate (default: bundled sample image)")
	text := flag.String("text", "Review this image before attaching it to a customer support ticket.", "Text context for the image")
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	var doc genai.Doc
	name := *image
	if name == "" {
		data, err := scoreboard.TestdataFiles.ReadFile("testdata/image.png")
		if err != nil {
			return err
		}
		doc = genai.Doc{Filename: "image.png", Src: bytes.NewReader(data)}
		name = "bundled image.png"
	} else {
		f, err := os.Open(name)
		if err != nil {
			return err
		}
		defer func() { err = errors.Join(err, f.Close()) }()
		doc = genai.Doc{Src: f}
	}
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
	// The HuggingFace download also selects the model's multimodal projector.
	// Only the Go client needs access; grant no browser origins cross-origin access.
	srv, err := llamacppsrv.New(ctx, exe, "", os.Stderr, "localhost:0", 0, []string{"-hf", "ggml-org/OpenJev-GGUF", "--no-warmup", "--cors-origins", "", "--no-cors-credentials", "--no-ui", "--no-slots", "--parallel", "4", "--kv-unified"})
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
	q := genai.Questions{
		"has_text": {Type: genai.QuestionNoul, Instructions: genai.Text("Does the image contain readable text?")},
		"kind": {Type: genai.QuestionChoice, Instructions: genai.Text("What kind of image is this?"), Choice: map[string]genai.DecisionContent{
			"document":     genai.Text("a scanned or photographed document"),
			"illustration": genai.Text("a drawing or rendered illustration"),
			"photograph":   genai.Text("a photograph of a real scene or object"),
			"screenshot":   genai.Text("a screenshot of an application or website"),
		}},
		"quality": {Type: genai.QuestionScore, Instructions: genai.Text("How useful is the image for the purpose described in the text context?"), Score: []genai.DecisionContent{
			genai.Text("unusable: the relevant content cannot be identified"),
			genai.Text("poor: important details are unclear or missing"),
			genai.Text("adequate: the relevant content can be understood"),
			genai.Text("clear: the relevant details are easy to identify"),
		}},
	}
	fmt.Printf("Image: %s\nState: %s\n", name, *text)
	start := time.Now()
	req := genai.SystemOneRequest{State: genai.Text(*text), Docs: []genai.Doc{doc}, Questions: q}

	res, err := c.SystemOne(ctx, &req)
	if err != nil {
		return err
	}
	elapsed := time.Since(start)

	fmt.Println("Answers:")
	fmt.Printf("- has_text: %.2f likely to be a yes\n", res.Answers["has_text"].Noul)
	fmt.Printf("- kind:     %s with %.0f%% confidence\n", res.Answers["kind"].Choice, 100*res.Answers["kind"].Confidence)
	for _, name := range slices.Sorted(maps.Keys(res.Answers["kind"].Probabilities)) {
		fmt.Printf("    %s: %.2f\n", name, res.Answers["kind"].Probabilities[name])
	}
	fmt.Printf("- quality:  %.2f over %d levels with %.0f%% confidence\n", res.Answers["quality"].Score, len(res.Answers["quality"].Legend), 100*res.Answers["quality"].Confidence)
	for _, level := range slices.Sorted(maps.Keys(res.Answers["quality"].Probabilities)) {
		fmt.Printf("    %s (%v): %.2f\n", level, res.Answers["quality"].Legend[level], res.Answers["quality"].Probabilities[level])
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
