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
		return &orphanedError{root: r.root, name: slices.Sorted(maps.Keys(r.preexisting))}
	}
	return nil
}

func (r *Records) Signal(name string) {
	if name == "" {
		return
	}
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
//
// Don't forget to call Stop()!
func (r *Records) Record(name string, h http.RoundTripper, opts ...recorder.Option) (*Recorder, error) {
	name = strings.ReplaceAll(strings.ReplaceAll(name, "/", string(os.PathSeparator)), ":", "-")
	if d := filepath.Dir(name); d != "." {
		if err := os.MkdirAll(filepath.Join(r.root, d), 0o755); err != nil {
			return nil, err
		}
	}
	mode := recorder.ModeRecordOnce
	if v := os.Getenv("RECORD"); v == "1" {
		mode = recorder.ModeRecordOnly
	} else if v == "2" {
		mode = recorder.ModeReplayWithNewEpisodes
	}
	args := []recorder.Option{
		recorder.WithMode(mode),
	}
	r.Signal(name)
	rec, err := httprecord.New(filepath.Join(r.root, name), h, append(args, opts...)...)
	if err != nil {
		return nil, err
	}
	return &Recorder{Recorder: rec, name: name + ".yaml"}, nil
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
type Recorder struct {
	*recorder.Recorder
	name string
}

func (r *Recorder) RoundTrip(req *http.Request) (*http.Response, error) {
	resp, err := r.Recorder.RoundTrip(req)
	if err != nil && req.GetBody != nil {
		if body, _ := req.GetBody(); body != nil {
			defer body.Close()
			b, _ := io.ReadAll(body)
			err = fmt.Errorf("%w; cassette %q; body:\n %s", err, r.name, string(b))
		}
	}
	return resp, err
}
