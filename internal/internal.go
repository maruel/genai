// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal is awesome sauce.
package internal

import (
	"context"
	"fmt"
	"io/fs"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

//go:generate go run update_readme.go

// BeLenient is used by all clients to enable or disable httpjson.Client.Lenient.
//
// It is true by default. Tests must manually set it to false.
var BeLenient = true

// DefaultMatcher ignores authentication via API keys.
var DefaultMatcher = cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Api-Key", "X-Key", "X-Request-Id"))

// Records represents HTTP recordings.
type Records struct {
	root        string
	mu          sync.Mutex
	preexisting map[string]struct{}
	recorded    map[string]struct{}
}

func NewRecords(root string) (*Records, error) {
	r := &Records{root: root, preexisting: make(map[string]struct{}), recorded: make(map[string]struct{})}
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err == nil && !d.IsDir() && strings.HasSuffix(path, ".yaml") {
			r.preexisting[path[len(root)+1:]] = struct{}{}
		}
		return err
	})
	if os.IsNotExist(err) {
		return r, nil
	}
	return r, err
}

func (r *Records) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for f := range r.recorded {
		delete(r.preexisting, f)
	}
	if len(r.preexisting) != 0 {
		names := make([]string, 0, len(r.preexisting))
		for f := range r.preexisting {
			names = append(names, f)
		}
		sort.Strings(names)
		return &orphanedError{root: r.root, name: names}
	}
	return nil
}

func (r *Records) Signal(name string) {
	r.mu.Lock()
	_, ok := r.recorded[name+".yaml"]
	r.recorded[name+".yaml"] = struct{}{}
	r.mu.Unlock()
	if ok {
		panic(fmt.Sprintf("refusing duplicate %q", name))
	}
}

// Record records and replays HTTP requests.
//
// When the environment variable RECORD=1 is set, it forcibly re-record the
// cassettes and save in <root>/<testname>.yaml.
//
// It ignores the port number in the URL both for recording and playback so it
// works with local services like ollama and llama-server.
func (r *Records) Record(name string, h http.RoundTripper, opts ...recorder.Option) (*recorder.Recorder, error) {
	name = strings.ReplaceAll(strings.ReplaceAll(name, "/", string(os.PathSeparator)), ":", "-")
	if d := filepath.Dir(name); d != "." {
		if err := os.MkdirAll(filepath.Join(r.root, d), 0o755); err != nil {
			return nil, err
		}
	}
	mode := recorder.ModeRecordOnce
	if os.Getenv("RECORD") == "1" {
		mode = recorder.ModeRecordOnly
	}
	args := []recorder.Option{
		recorder.WithHook(trimResponseHeaders, recorder.AfterCaptureHook),
		recorder.WithMode(mode),
		recorder.WithSkipRequestLatency(true),
		recorder.WithRealTransport(h),
		recorder.WithMatcher(DefaultMatcher),
	}
	r.Signal(name)
	// Don't forget to call Stop()!
	return recorder.New(filepath.Join(r.root, name), append(args, opts...)...)
}

type contextKey struct{}

func Logger(ctx context.Context) *slog.Logger {
	v := ctx.Value(contextKey{})
	switch v := v.(type) {
	case *slog.Logger:
		return v
	default:
		return slog.Default()
	}
}

func WithLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, contextKey{}, logger)
}

//

type orphanedError struct {
	root string
	name []string
}

func (e *orphanedError) Error() string {
	return fmt.Sprintf("Found orphaned recordings in %s:\n- %s", e.root, strings.Join(e.name, "\n- "))
}

func trimResponseHeaders(i *cassette.Interaction) error {
	// Authentication via API keys.
	i.Request.Headers.Del("Authorization")
	i.Request.Headers.Del("X-Api-Key")
	i.Request.Headers.Del("X-Key")
	// Noise.
	i.Request.Headers.Del("X-Request-Id")
	i.Response.Headers.Del("Date")
	i.Response.Headers.Del("Request-Id")
	// Remove this here since it also happens in openaicompatible.
	i.Response.Headers.Del("Anthropic-Organization-Id")
	// The cookie may be used for authentication?
	i.Response.Headers.Del("Set-Cookie")
	// Noise.
	i.Response.Duration = i.Response.Duration.Round(time.Millisecond)
	return nil
}
