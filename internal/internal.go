// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal is awesome sauce.
package internal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"mime"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/httpjson"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/cassette"
	"gopkg.in/dnaeon/go-vcr.v4/pkg/recorder"
)

//go:generate go run update_readme.go

// BeLenient is used by all clients to enable or disable httpjson.Client.Lenient.
//
// It is true by default. Tests must manually set it to false.
var BeLenient = true

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
	if os.Getenv("RECORD") == "1" {
		mode = recorder.ModeRecordOnly
	}
	args := []recorder.Option{
		recorder.WithHook(trimResponseHeaders, recorder.AfterCaptureHook),
		recorder.WithHook(trimRecordingCloudflare, recorder.AfterCaptureHook),
		recorder.WithHook(trimRecordingHostPort, recorder.AfterCaptureHook),
		recorder.WithHook(trimRecordingGemini, recorder.AfterCaptureHook),
		recorder.WithMode(mode),
		recorder.WithSkipRequestLatency(true),
		recorder.WithRealTransport(h),
		recorder.WithMatcher(matchIgnorePort),
	}
	r.Signal(name)
	// Don't forget to call Stop()!
	rec, err := recorder.New(filepath.Join(r.root, name), append(args, opts...)...)
	if err != nil {
		return nil, err
	}
	return &recorderWithBody{Recorder: rec, name: name + ".yaml"}, nil
}

// Logger retrieves a slog.Logger from the context if any, otherwise returns slog.Default().
func Logger(ctx context.Context) *slog.Logger {
	v := ctx.Value(contextKey{})
	switch v := v.(type) {
	case *slog.Logger:
		return v
	default:
		return slog.Default()
	}
}

// WithLogger injects a slog.Logger into the context. It can be retrieved with Logger().
func WithLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, contextKey{}, logger)
}

// DecodeJSON is duplicate from httpjson.go in https://github.com/maruel/httpjson.
func DecodeJSON(d *json.Decoder, out any, r io.ReadSeeker) (bool, error) {
	d.UseNumber()
	if err := d.Decode(out); err != nil {
		// decode.object() in encoding/json.go does not return a structured error
		// when an unknown field is found or when the type is wrong. Process it manually.
		if r != nil {
			if s := err.Error(); strings.Contains(s, "json: unknown field ") || strings.Contains(s, "json: cannot unmarshal ") {
				// Decode again but this time capture all errors. Try first as a map (JSON object), then as a slice
				// (JSON list).
				for _, t := range []any{map[string]any{}, []any{}} {
					if _, err2 := r.Seek(0, 0); err2 != nil {
						// Unexpected.
						return false, err2
					}
					d = json.NewDecoder(r)
					d.UseNumber()
					if err2 := d.Decode(&t); err2 == nil {
						if err2 = errors.Join(httpjson.FindExtraKeys(reflect.TypeOf(out), t)...); err2 != nil {
							return true, err2
						}
					}
				}
			}
		}
		return false, err
	}
	return false, nil
}

//

type contextKey struct{}

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

func matchCassetteGemini(r *http.Request, i cassette.Request) bool {
	// Gemini pass the API key as a query argument (!) so zap it before matching.
	r = r.Clone(r.Context())
	r.URL.RawQuery = removeKeyFromQuery(r.URL.RawQuery, "key")
	_ = r.ParseForm()
	return matchCassetteCloudflare(r, i)
}

func trimRecordingGemini(i *cassette.Interaction) error {
	// Gemini pass the API key as a query argument (!) so zap it before recording.
	u, err := url.Parse(i.Request.URL)
	if err != nil {
		return err
	}
	u.RawQuery = removeKeyFromQuery(u.RawQuery, "key")
	i.Request.URL = u.String()
	i.Request.Form.Del("key")
	return nil
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
	return matchCassetteGemini(r, i)
}

func removeKeyFromQuery(query, keyToRemove string) string {
	// Using url.URL.Query() then Encode() reorders the keys, which makes it non-deterministic. Do it manually.
	b := strings.Builder{}
	for part := range strings.SplitSeq(query, "&") {
		if part != "" {
			if k := strings.SplitN(part, "=", 2)[0]; k == keyToRemove {
				continue
			}
		}
		if b.Len() != 0 {
			b.WriteByte('&')
		}
		b.WriteString(part)
	}
	return b.String()
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

// MimeByExt wraps mime.TypeByExtension.
//
// It overrides audio entries because they vary surprisingly a lot across OSes!
func MimeByExt(ext string) string {
	switch ext {
	case ".aac":
		return "audio/aac"
	case ".flac":
		return "audio/flac"
	case ".wav":
		return "audio/wav"
	default:
		return mime.TypeByExtension(ext)
	}
}

// JSONSchemaFor returns the JSON schema for the given type.
//
// Many providers (including OpenAI) struggle with $ref that jsonschema package uses by default.
func JSONSchemaFor(t reflect.Type) *jsonschema.Schema {
	// No need to set an ID on the struct, it's unnecessary data that may confuse the tool.
	r := jsonschema.Reflector{Anonymous: true, DoNotReference: true}
	return r.ReflectFromType(t)
}
