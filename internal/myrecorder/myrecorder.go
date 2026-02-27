// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package myrecorder has HTTP recording logic. It is in a separate package because I only want tests to
// depend on it so that genai users do not link with go-vcr by default but it is not using testing.T so it
// can't be in internaltest
package myrecorder

import (
	"fmt"
	"io"
	"io/fs"
	"maps"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"

	"github.com/maruel/genai/httprecord"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

// Records represents HTTP recordings.
type Records struct {
	root        string
	mu          sync.Mutex
	preexisting map[string]struct{}
	recorded    map[string]struct{}
}

// NewRecords creates a new recorder.
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

// Close finalizes all recordings.
func (r *Records) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	for f := range r.recorded {
		delete(r.preexisting, f)
	}
	if len(r.preexisting) != 0 {
		return &orphanedError{root: r.root, name: slices.Sorted(maps.Keys(r.preexisting))}
	}
	return nil
}

// Signal writes a signal marker to the recording.
func (r *Records) Signal(name string) error {
	if name == "" {
		return nil
	}
	r.mu.Lock()
	_, ok := r.recorded[name+".yaml"]
	r.recorded[name+".yaml"] = struct{}{}
	r.mu.Unlock()
	if ok {
		return fmt.Errorf("refusing duplicate %q", name)
	}
	return nil
}

// Record records and replays HTTP requests.
//
// When the environment variable RECORD is set, it controls recording behavior:
//   - "all": forcibly re-record all cassettes into <root>/<testname>.yaml.
//   - "failure_only": use ModeRecordOnce (same as default); the caller is
//     expected to delete cassettes for failed tests so the next run re-records
//     only those.
//
// It ignores the port number in the URL both for recording and playback so it
// works with local services like ollama and llama-server.
//
// Don't forget to call Stop()!
func (r *Records) Record(name string, h http.RoundTripper, opts ...recorder.Option) (*Recorder, error) {
	name = strings.ReplaceAll(strings.ReplaceAll(name, "/", string(os.PathSeparator)), ":", "-")
	mode := recorder.ModeRecordOnce
	if v := os.Getenv("RECORD"); v == "all" {
		mode = recorder.ModeRecordOnly
	} else if v != "" && v != "failure_only" {
		return nil, fmt.Errorf("invalid RECORD value %q; expected \"all\" or \"failure_only\"", v)
	}
	args := []recorder.Option{
		recorder.WithMode(mode),
	}
	if err := r.Signal(name); err != nil {
		return nil, err
	}
	rec, err := httprecord.New(filepath.Join(r.root, name), h, append(args, opts...)...)
	if err != nil {
		return nil, err
	}
	return &Recorder{Recorder: rec, name: name + ".yaml", root: r.root}, nil
}

type orphanedError struct {
	root string
	name []string
}

func (e *orphanedError) Error() string {
	return fmt.Sprintf("Found orphaned recordings in %s:\n- %s", e.root, strings.Join(e.name, "\n- "))
}

//

// Recorder wraps the POST body in the error message.
//
// It is a http.RoundTripper.
type Recorder struct {
	*recorder.Recorder
	name string
	root string
}

// CassettePath returns the absolute path to the cassette file.
func (r *Recorder) CassettePath() string {
	return filepath.Join(r.root, r.name)
}

// RoundTrip implements http.RoundTripper.
func (r *Recorder) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := r.Recorder.RoundTrip(req)
	if err != nil && req.GetBody != nil {
		if body, _ := req.GetBody(); body != nil {
			defer func() { _ = body.Close() }()
			b, _ := io.ReadAll(body)
			err = fmt.Errorf("%w; cassette %q; body:\n %s", err, r.name, string(b))
		}
	}
	return resp, err
}

func (r *Recorder) Unwrap() http.RoundTripper {
	return r.Recorder
}
