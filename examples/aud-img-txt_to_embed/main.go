// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Embed text, an image and audio into a shared vector space.
//
// Requires GEMINI_API_KEY. Uses Gemini Embedding 2 and bundled media by default.

package main

import (
	"bytes"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/gemini"
	"github.com/maruel/genai/scoreboard"
)

func mainImpl() (err error) {
	model := flag.String("model", "gemini-embedding-2", "Embedding model")
	text := flag.String("text", "A picture and a spoken word.", "Text to embed")
	image := flag.String("image", "", "Local image (default: bundled image.jpg)")
	audio := flag.String("audio", "", "Local audio (default: bundled audio.wav)")
	dimensions := flag.Int("dimensions", 128, "Vector dimensions (0: model default)")
	flag.Parse()
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()
	img, imgFile, err := loadDoc(*image, "image.jpg")
	if err != nil {
		return err
	}
	if imgFile != nil {
		defer func() { err = errors.Join(err, imgFile.Close()) }()
	}
	aud, audFile, err := loadDoc(*audio, "audio.wav")
	if err != nil {
		return err
	}
	if audFile != nil {
		defer func() { err = errors.Join(err, audFile.Close()) }()
	}
	c, err := gemini.New(ctx, genai.ProviderOptionModel(*model))
	if c != nil {
		defer func() { err = errors.Join(err, c.Close()) }()
	}
	if err != nil {
		return err
	}
	out, err := c.Embed(ctx, &genai.EmbeddingRequest{Inputs: []genai.Request{{Text: *text}, {Doc: img}, {Doc: aud}}, Dimensions: *dimensions})
	if err != nil {
		return err
	}
	for i, name := range []string{"text", "image", "audio"} {
		v := out.Embeddings[i]
		fmt.Printf("%s: %d dimensions, first values %v\n", name, len(v), v[:min(6, len(v))])
	}
	fmt.Printf("Cosine similarity: text/image %.4f, text/audio %.4f, image/audio %.4f\n", cosine(out.Embeddings[0], out.Embeddings[1]), cosine(out.Embeddings[0], out.Embeddings[2]), cosine(out.Embeddings[1], out.Embeddings[2]))
	fmt.Printf("Tokens: %d input, %d total\n", out.Usage.InputTokens, out.Usage.TotalTokens)
	return nil
}

func loadDoc(name, bundled string) (genai.Doc, *os.File, error) {
	if name != "" {
		f, err := os.Open(name)
		if err != nil {
			return genai.Doc{}, nil, err
		}
		return genai.Doc{Filename: filepath.Base(name), Src: f}, f, nil
	}
	data, err := scoreboard.TestdataFiles.ReadFile("testdata/" + bundled)
	if err != nil {
		return genai.Doc{}, nil, err
	}
	return genai.Doc{Filename: bundled, Src: bytes.NewReader(data)}, nil, nil
}

func cosine(a, b []float32) float64 {
	var dot, x, y float64
	for i, v := range a {
		av, bv := float64(v), float64(b[i])
		dot += av * bv
		x += av * av
		y += bv * bv
	}
	return dot / math.Sqrt(x*y)
}

func main() {
	if err := mainImpl(); err != nil {
		log.Fatal(err)
	}
}
