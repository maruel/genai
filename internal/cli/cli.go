// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package cli provides process helpers for CLI-based providers.
package cli

import (
	"context"
	"fmt"
	"io"
	"os/exec"

	"github.com/maruel/genai"
)

// NewStarter returns a genai.Starter that runs bin in dir. The caller owns dir
// and its cleanup.
func NewStarter(bin, dir string) genai.Starter {
	return func(ctx context.Context, args []string) (io.WriteCloser, io.ReadCloser, func() error, error) {
		cmd := exec.CommandContext(ctx, bin, args...)
		cmd.Dir = dir
		stdin, err := cmd.StdinPipe()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("stdin pipe: %w", err)
		}
		stdout, err := cmd.StdoutPipe()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("stdout pipe: %w", err)
		}
		if err := cmd.Start(); err != nil {
			return nil, nil, nil, fmt.Errorf("start %s: %w", bin, err)
		}
		return stdin, stdout, cmd.Wait, nil
	}
}
