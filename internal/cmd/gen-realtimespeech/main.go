// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command gen-realtimespeech regenerates the realtime speech FLAC fixtures with
// llama-tts.
//
// It downloads llama-tts and the TTS model on first use. Regeneration changes
// the client audio bytes, so re-record realtime smoke fixtures with
// -update-scoreboard afterwards.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/maruel/genai/providers/llamacpp/llamacppsrv"
	"github.com/maruel/genai/smoke"
)

const (
	modelRepo = "ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF"
	outDir    = "scoreboard/testdata/speech"
)

type phrase struct {
	name string
	text string
}

var phrases = []phrase{
	{name: "orange", text: "Please say the word orange."},
	{name: "code", text: "What is the access code? Call the lookup code tool, then reply with only the code."},
}

func mainImpl() error {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	cache, err := os.UserCacheDir()
	if err != nil {
		return err
	}
	cache = filepath.Join(cache, "genai", "llama-tts")
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}
	llamaTTS, err := llamacppsrv.DownloadTTSVersion(ctx, cache, llamacppsrv.Version)
	if err != nil {
		return err
	}
	for _, p := range phrases {
		if err := generate(ctx, llamaTTS, p); err != nil {
			return fmt.Errorf("%s: %w", p.name, err)
		}
	}
	return nil
}

func generate(ctx context.Context, exe string, p phrase) error {
	f, err := os.CreateTemp("", "gen-realtimespeech-*.wav")
	if err != nil {
		return err
	}
	tmp := f.Name()
	if err := f.Close(); err != nil {
		return err
	}
	defer func() {
		_ = os.Remove(tmp)
	}()
	cmd := exec.CommandContext(ctx, exe, // #nosec G204 -- exe is the downloaded llama-tts binary.
		"-hf", modelRepo,
		"-p", p.text,
		"--output", tmp,
		"--seed", "0",
		"--temp", "0",
	)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("llama-tts: %w", err)
	}
	raw, err := os.ReadFile(tmp)
	if err != nil {
		return err
	}
	pcm, rate, err := smoke.DecodePCM16WAV(raw)
	if err != nil {
		return err
	}
	if rate != 24000 {
		pcm, err = smoke.ResamplePCM16(pcm, rate, 24000)
		if err != nil {
			return err
		}
		rate = 24000
	}
	raw, err = smoke.EncodePCM16FLAC(pcm, rate)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(outDir, p.name+"-24000.flac"), raw, 0o644)
}

func main() {
	if err := mainImpl(); err != nil {
		if !errors.Is(err, context.Canceled) {
			_, _ = fmt.Fprintf(os.Stderr, "gen-realtimespeech: %s\n", err)
		}
		os.Exit(1)
	}
}
