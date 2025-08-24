// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package myrecorder has HTTP recording logic. It is in a separate package because I only want tests to
// depend on it so that genai users do not link with go-vcr by default but it is not using testing.T so it
// can't be in internaltest
package myrecorder

import (
	"bytes"
	"fmt"
	"io"
	"io/fs"
	"maps"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

// DefaultMatcher ignores authentication via API keys.
var DefaultMatcher = cassette.NewDefaultMatcher(cassette.WithIgnoreHeaders("Authorization", "X-Api-Key", "X-Goog-Api-Key", "X-Key", "X-Request-Id"))

type Recorder interface {
	http.RoundTripper
	Stop() error
	IsNewCassette() bool
}

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
func (r *Records) Record(name string, h http.RoundTripper, opts ...recorder.Option) (Recorder, error) {
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
		recorder.WithHook(trimResponseHeaders, recorder.AfterCaptureHook),
		recorder.WithHook(trimRecordingCloudflare, recorder.AfterCaptureHook),
		recorder.WithHook(trimRecordingHostPort, recorder.AfterCaptureHook),
		recorder.WithMode(mode),
		recorder.WithSkipRequestLatency(true),
		recorder.WithRealTransport(h),
		recorder.WithMatcher(matchIgnorePort),
		recorder.WithFS(&skipEmptyFS{FS: cassette.NewDiskFS()}),
	}
	r.Signal(name)
	// Don't forget to call Stop()!
	rec, err := recorder.New(filepath.Join(r.root, name), append(args, opts...)...)
	if err != nil {
		return nil, err
	}
	return &recorderWithBody{Recorder: rec, name: name + ".yaml"}, nil
}

type skipEmptyFS struct {
	cassette.FS
}

func (c *skipEmptyFS) WriteFile(name string, data []byte) error {
	if bytes.Contains(data, []byte("interactions: []")) {
		// Do not save files without an interaction.
		return nil
	}
	return c.FS.WriteFile(name, data)
}

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
	i.Request.Headers.Del("X-Goog-Api-Key")
	i.Request.Headers.Del("X-Key")
	// OMG What are they thinking? This happens on HTTP 302 redirect when fetching Veo generated videos:
	i.Response.Headers.Del("X-Goog-Api-Key")
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

var reCloudflareAccount = regexp.MustCompile(`/accounts/[0-9a-fA-F]{32}/`)

func trimRecordingCloudflare(i *cassette.Interaction) error {
	// Zap the account ID from the URL path before saving.
	i.Request.URL = reCloudflareAccount.ReplaceAllString(i.Request.URL, "/accounts/ACCOUNT_ID/")
	return nil
}

func matchCassetteCloudflare(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	// When matching, ignore the account ID from the URL path.
	r.URL.Path = reCloudflareAccount.ReplaceAllString(r.URL.Path, "/accounts/ACCOUNT_ID/")
	return DefaultMatcher(r, i)
}

// trimRecordingHostPort is a recorder.HookFunc to remove the port number when recording.
func trimRecordingHostPort(i *cassette.Interaction) error {
	i.Request.Host = strings.Split(i.Request.Host, ":")[0]
	u, err := url.Parse(i.Request.URL)
	if err != nil {
		return err
	}
	u.Host = strings.Split(u.Host, ":")[0]
	i.Request.URL = u.String()
	return nil
}

// matchIgnorePort is a recorder.MatcherFunc that ignore the host port number. This is useful for locally
// hosted LLM providers like llamacpp and ollama.
func matchIgnorePort(r *http.Request, i cassette.Request) bool {
	r = r.Clone(r.Context())
	r.URL.Host = strings.Split(r.URL.Host, ":")[0]
	r.Host = strings.Split(r.Host, ":")[0]
	return matchCassetteCloudflare(r, i)
}

// recorderWithBody wraps the POST body in the error message.
type recorderWithBody struct {
	*recorder.Recorder
	name string
}

func (r *recorderWithBody) RoundTrip(req *http.Request) (*http.Response, error) {
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
