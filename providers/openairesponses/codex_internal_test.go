// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests Codex file credentials, renewal, confinement, and terminal events.
package openairesponses

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/internaltest"
)

func fakeToken(exp int64) string { return userToken(exp, "test-account", "test-user") }

func writeCredentials(t *testing.T, exp int64) string {
	dir := t.TempDir()
	b, err := json.Marshal(map[string]any{"auth_mode": "chatgpt", "future_field": "preserved", "tokens": map[string]any{"access_token": fakeToken(exp), "account_id": "test-account", "refresh_token": "test-refresh", "id_token": fakeToken(exp), "future_token_field": "preserved"}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "auth.json"), b, 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

type transportFunc func(*http.Request) (*http.Response, error)

func (f transportFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestCredentialStore(t *testing.T) {
	t.Run("load", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, raw := range []string{`null`, `{`, `{"auth_mode":"apikey"}`, `{"tokens":{}}`, `{"tokens":{"access_token":"bad","account_id":"account"}}`, `{"tokens":{"access_token":"header.e30.signature","account_id":"account"}}`} {
				t.Run(fmt.Sprintf("case-%d", len(raw)), func(t *testing.T) {
					dir := t.TempDir()
					if err := os.WriteFile(filepath.Join(dir, "auth.json"), []byte(raw), 0o600); err != nil {
						t.Fatal(err)
					}
					s := credentialStore{path: filepath.Join(dir, "auth.json")}
					if _, err := s.load(); err == nil {
						t.Fatal("accepted invalid credentials")
					}
				})
			}
			s := credentialStore{path: filepath.Join(t.TempDir(), "missing")}
			if _, err := s.load(); err == nil {
				t.Fatal("accepted missing credentials")
			}
		})
	})
	t.Run("current", func(t *testing.T) {
		t.Run("renewal", func(t *testing.T) {
			dir := writeCredentials(t, 1)
			var requests atomic.Int64
			s := credentialStore{path: filepath.Join(dir, "auth.json"), client: http.Client{Transport: transportFunc(func(r *http.Request) (*http.Response, error) {
				requests.Add(1)
				if r.URL.String() != authURL {
					t.Errorf("refresh URL %s", r.URL)
				}
				var in refreshRequest
				if err := json.NewDecoder(r.Body).Decode(&in); err != nil {
					t.Error(err)
				}
				if in.ClientID != clientID || in.RefreshToken != "test-refresh" || in.GrantType != "refresh_token" {
					t.Error("wrong renewal grant")
				}
				b, _ := json.Marshal(refreshResponse{AccessToken: fakeToken(4102444800), RefreshToken: "rotated-refresh", IDToken: fakeToken(4102444800)})
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(string(b))), Header: make(http.Header)}, nil
			})}}
			var wg sync.WaitGroup
			for range 8 {
				wg.Go(func() {
					if _, err := s.current(t.Context()); err != nil {
						t.Error(err)
					}
				})
			}
			wg.Wait()
			if requests.Load() != 1 {
				t.Fatalf("refreshes %d", requests.Load())
			}
			d, err := s.load()
			if err != nil {
				t.Fatal(err)
			}
			if d.tokens.RefreshToken != "rotated-refresh" || d.tokens.IDToken != fakeToken(4102444800) || string(d.fields["future_field"]) != `"preserved"` {
				t.Fatal("renewal failed to preserve or update credential data")
			}
			var fields map[string]json.RawMessage
			if err := json.Unmarshal(d.fields["tokens"], &fields); err != nil {
				t.Fatal(err)
			}
			if string(fields["future_token_field"]) != `"preserved"` {
				t.Fatal("lost unknown token field")
			}
			info, err := os.Stat(s.path)
			if err != nil {
				t.Fatal(err)
			}
			// Windows does not expose Unix permission bits through os.FileMode.
			if runtime.GOOS != "windows" && info.Mode().Perm() != 0o600 {
				t.Fatalf("permissions %v", info.Mode())
			}
		})
		t.Run("rejection", func(t *testing.T) {
			dir := writeCredentials(t, 1)
			s := credentialStore{path: filepath.Join(dir, "auth.json"), client: http.Client{Transport: transportFunc(func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: http.StatusUnauthorized, Body: io.NopCloser(strings.NewReader("test-refresh"))}, nil
			})}}
			if _, err := s.current(t.Context()); err == nil || strings.Contains(err.Error(), "test-refresh") {
				t.Fatalf("unsafe error %v", err)
			}
			d, err := s.load()
			if err != nil {
				t.Fatal(err)
			}
			if d.tokens.RefreshToken != "test-refresh" {
				t.Fatal("changed rejected grant")
			}
		})
	})
}

func TestAuthTransport(t *testing.T) {
	t.Run("confinement", func(t *testing.T) {
		transport := authTransport{transport: transportFunc(func(*http.Request) (*http.Response, error) {
			t.Fatal("credential reached foreign endpoint")
			return nil, errors.New("unexpected foreign request")
		})}
		req, err := http.NewRequestWithContext(t.Context(), http.MethodGet, "https://example.com/backend-api/codex/models", http.NoBody)
		if err != nil {
			t.Fatal(err)
		}
		if resp, err := transport.RoundTrip(req); err == nil {
			if resp != nil {
				_ = resp.Body.Close()
			}
			t.Fatal("accepted foreign origin")
		}
	})
	t.Run("cancel", func(t *testing.T) {
		dir := writeCredentials(t, 1)
		s := credentialStore{path: filepath.Join(dir, "auth.json")}
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		if _, err := s.current(ctx); !errors.Is(err, context.Canceled) {
			t.Fatalf("error %v", err)
		}
	})
}

func TestCodexCredentialIdentity(t *testing.T) {
	dir := writeCredentials(t, 4102444800)
	t.Setenv("CODEX_HOME", dir)
	calls := 0
	c, err := New(t.Context(), &ProviderOption{Auth: AuthCodex}, genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
		return transportFunc(func(*http.Request) (*http.Response, error) {
			calls++
			return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(`{"models":[{"slug":"gpt-6-luna"}]}`))}, nil
		})
	}))
	if err != nil {
		t.Fatal(err)
	}
	internaltest.CleanupCloser(t, c)
	var d map[string]json.RawMessage
	raw, err := os.ReadFile(filepath.Join(dir, "auth.json"))
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, &d); err != nil {
		t.Fatal(err)
	}
	var tok map[string]any
	if err := json.Unmarshal(d["tokens"], &tok); err != nil {
		t.Fatal(err)
	}
	switched := userToken(4102444800, "test-account", "another-user")
	tok["access_token"] = switched
	tok["id_token"] = switched
	d["tokens"], err = json.Marshal(tok)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = json.Marshal(d)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "auth.json"), raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := c.Ping(t.Context()); err == nil {
		t.Fatal("accepted another user in the same workspace")
	}
	if calls != 0 {
		t.Fatal("switched user credentials reached the network")
	}
}

func TestCodexRawStreamCompletion(t *testing.T) {
	t.Setenv("CODEX_HOME", writeCredentials(t, 4102444800))
	for _, tc := range []struct{ name, event, want string }{
		{"missing", "", "codex subscription stream ended without response.completed"},
		{"failed", `{"type":"response.failed","response":{"error":{"code":"server_error","message":"denied"}}}`, "server_error: denied"},
		{"error", `{"type":"error","error":{"type":"invalid_request_error","message":"denied"}}`, "invalid_request_error: denied"},
		{"incomplete", `{"type":"response.incomplete","response":{"incomplete_details":{"reason":"budget"}}}`, "codex response incomplete: budget"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stream := "data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}\n\n"
			packets := 1
			if tc.event != "" {
				stream += "data: " + tc.event + "\n\n"
				packets++
			}
			c, err := New(t.Context(), &ProviderOption{Auth: AuthCodex}, genai.ProviderOptionModel("gpt-6-luna"), genai.ProviderOptionTransportWrapper(func(http.RoundTripper) http.RoundTripper {
				return transportFunc(func(*http.Request) (*http.Response, error) {
					return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(stream))}, nil
				})
			}))
			if err != nil {
				t.Fatal(err)
			}
			internaltest.CleanupCloser(t, c)
			in := Response{}
			if err := in.Init(genai.Messages{genai.NewTextMessage("hello")}, "gpt-6-luna"); err != nil {
				t.Fatal(err)
			}
			chunks, finish := c.GenStreamRaw(t.Context(), &in)
			count := 0
			for range chunks {
				count++
			}
			if err := finish(); err == nil || err.Error() != tc.want {
				t.Fatalf("error %v", err)
			}
			if count != packets {
				t.Fatalf("packets %d want %d", count, packets)
			}
		})
	}
}

func TestCodexPingIgnoresModelCache(t *testing.T) {
	dir := writeCredentials(t, 4102444800)
	t.Setenv("CODEX_HOME", dir)
	c, err := New(t.Context(), &ProviderOption{Auth: AuthCodex}, genai.ProviderOptionPreloadedModels{&codexModel{Slug: "gpt-6-luna"}})
	if err != nil {
		t.Fatal(err)
	}
	internaltest.CleanupCloser(t, c)
	if err := os.Remove(filepath.Join(dir, "auth.json")); err != nil {
		t.Fatal(err)
	}
	if err := c.Ping(t.Context()); err == nil {
		t.Fatal("cached models authorized a missing login")
	}
}

func userToken(exp int64, account, user string) string {
	b, _ := json.Marshal(map[string]any{"exp": exp, "https://api.openai.com/auth": map[string]string{"chatgpt_account_id": account, "chatgpt_user_id": user}})
	return "header." + base64.RawURLEncoding.EncodeToString(b) + ".signature"
}

func TestCodexCredentialIdentityMismatch(t *testing.T) {
	for _, tc := range []struct{ name, account, accessUser, idUser string }{
		{"token_mismatch", "test-account", "test-user", "another-user"},
		{"renewed_user", "test-account", "another-user", "another-user"},
		{"renewed_workspace", "another-account", "test-user", "test-user"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := writeCredentials(t, 1)
			s := credentialStore{path: filepath.Join(dir, "auth.json"), client: http.Client{Transport: transportFunc(func(*http.Request) (*http.Response, error) {
				body, err := json.Marshal(refreshResponse{AccessToken: userToken(4102444800, tc.account, tc.accessUser), IDToken: userToken(4102444800, tc.account, tc.idUser), RefreshToken: "rotated"})
				if err != nil {
					return nil, err
				}
				return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(string(body)))}, nil
			})}}
			if _, err := s.current(t.Context()); err == nil {
				t.Fatal("accepted a different user during renewal")
			}
			d, err := s.load()
			if err != nil {
				t.Fatal(err)
			}
			if d.tokens.RefreshToken != "test-refresh" {
				t.Fatal("overwrote original credentials")
			}
		})
	}
}
