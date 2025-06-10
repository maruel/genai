// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command cache-mgr fetches and prints out the list of files stored on the selected provider.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"slices"
	"sort"
	"strings"
	"syscall"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/providers"
	"golang.org/x/sync/errgroup"
)

func listCache(ctx context.Context, c genai.ProviderCache) error {
	entries, err := c.CacheList(ctx)
	if err != nil {
		return err
	}
	for _, f := range entries {
		fmt.Printf("%s %s expires: %s\n", f.GetID(), f.GetDisplayName(), f.GetExpiry().Format("2006-01-02 15:04 MST"))
		// fmt.Printf("%s %s %d bytes purpose:%s created:%s expires:%s\n", f.ID, f.Filename, f.Bytes, f.Purpose, f.CreatedAt.AsTime().Format("2006-01-02 15:04:05"), f.ExpiresAt.AsTime().Format("2006-01-02 15:04 MST"))
	}
	return nil
}

func mainImpl() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()

	names := make([]string, 0, len(providers.All))
	for name := range providers.All {
		if c, _ := providers.All[name]("", nil); c != nil {
			if _, ok := c.(genai.ProviderCache); ok {
				names = append(names, name)
			}
		}
	}
	sort.Strings(names)
	provider := flag.String("provider", "", "backend to use: "+strings.Join(names, ", "))
	strict := flag.Bool("strict", false, "assert no unknown fields in the APIs are found")
	flag.Parse()
	if *strict {
		internal.BeLenient = false
	}
	if *provider == "" {
		return errors.New("-provider is required")
	}
	if !slices.Contains(names, *provider) {
		return fmt.Errorf("unknown backend %q", *provider)
	}
	c, err := providers.All[*provider]("", nil)
	if err != nil {
		return err
	}
	l := c.(genai.ProviderCache)

	if flag.NArg() == 0 {
		return listCache(ctx, l)
	}

	switch flag.Arg(0) {
	case "delete":
		files := flag.Args()[1:]
		if len(files) == 0 {
			return errors.New("delete <ids...>")
		}
		eg, ctx2 := errgroup.WithContext(ctx)
		for _, id := range files {
			eg.Go(func() error {
				return l.CacheDelete(ctx2, id)
			})
		}
		return eg.Wait()
	default:
		return fmt.Errorf("unknown command %q", flag.Arg(0))
	}
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "cache-mgr: %s\n", err)
		}
		os.Exit(1)
	}
}
