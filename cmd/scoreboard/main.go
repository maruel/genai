// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command scoreboard generates a scoreboard for every providers supported.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
)

func mainImpl() error {
	ctx := context.Background()
	table := flag.Bool("table", false, "output a markdown table")
	provider := flag.String("provider", "", "output a table only for one provider")
	model := flag.String("model", "", "run a smoke test on the model and output the JSON")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("unexpected arguments")
	}
	if *model != "" {
		if *provider == "" {
			return errors.New("-model requires -provider")
		}
		return smokeModel(ctx, os.Stdout, *provider, *model)
	}
	if *table {
		return printTable(ctx, os.Stdout, *provider)
	}
	if *provider != "" {
		// TODO: Add it to list too.
		return errors.New("-provider requires -table")
	}
	return printList(ctx, os.Stdout)
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "scoreboard: %s\n", err)
		}
		os.Exit(1)
	}
}
