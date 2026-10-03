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
	// The HuggingFace download also selects the model's multimodal projector.
	// Only the Go client needs access; grant no browser origins cross-origin access.
	srv, err := llamacppsrv.New(ctx, exe, "", os.Stderr, "localhost:0", 0, []string{"-hf", "ggml-org/OpenJev-GGUF", "--no-warmup", "--cors-origins", "", "--no-cors-credentials", "--no-ui", "--no-slots", "--parallel", "4", "--kv-unified"})
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
	q := struct {
		HasText llamacpp.Noul   `json:"has_text"`
		Kind    llamacpp.Choice `json:"kind"`
		Quality llamacpp.Score  `json:"quality"`
	}{
		HasText: llamacpp.Noul{Instructions: llamacpp.Text("Does the image contain readable text?")},
		Kind: llamacpp.Choice{
			Instructions: llamacpp.Text("What kind of image is this?"),
			Criteria: map[string]llamacpp.DecisionContent{
				"document":    llamacpp.Text("a scanned or photographed document"),
				"illustration": llamacpp.Text("a drawing or rendered illustration"),
				"photograph":  llamacpp.Text("a photograph of a real scene or object"),
				"screenshot":  llamacpp.Text("a screenshot of an application or website"),
			},
		},
		Quality: llamacpp.Score{
			Instructions: llamacpp.Text("How useful is the image for the purpose described in the text context?"),
			Criteria: []llamacpp.DecisionContent{
				llamacpp.Text("unusable: the relevant content cannot be identified"),
				llamacpp.Text("poor: important details are unclear or missing"),
				llamacpp.Text("adequate: the relevant content can be understood"),
				llamacpp.Text("clear: the relevant details are easy to identify"),
			},
		},
	}
	msgs := genai.Messages{
		genai.Message{Requests: []genai.Request{
			{Text: *text},
			{Doc: doc},
		}},
	}
	fmt.Printf("Image: %s\nState: %s\n", name, *text)
	start := time.Now()
	res, err := c.GenSync(ctx, msgs, &genai.GenOptionText{DecodeAs: &q})
	if err != nil {
		return err
	}
	elapsed := time.Since(start)
	if err := res.Decode(&q); err != nil {
		return err
	}
	fmt.Println("Answers:")
	fmt.Printf("- has_text: %.2f likely to be a yes\n", q.HasText.Probability)
	fmt.Printf("- kind:     %s with %.0f%% confidence\n", q.Kind.Label, 100*q.Kind.Confidence)
	for _, name := range slices.Sorted(maps.Keys(q.Kind.Probabilities)) {
		fmt.Printf("    %s: %.2f\n", name, q.Kind.Probabilities[name])
	}
	fmt.Printf("- quality:  %.2f over %d levels with %.0f%% confidence\n", q.Quality.Value, len(q.Quality.Legend), 100*q.Quality.Confidence)
	for _, level := range slices.Sorted(maps.Keys(q.Quality.Probabilities)) {
		fmt.Printf("    %s (%v): %.2f\n", level, q.Quality.Legend[level], q.Quality.Probabilities[level])
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
