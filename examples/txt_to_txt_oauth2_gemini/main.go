// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Demonstrates a full OAuth2 Authorization Code flow with PKCE to
// authenticate against the Google Gemini API.
//
// Tokens are cached in the user config directory and refreshed automatically.
//
// # Prerequisites
//
// 1. Create a Google Cloud project (or use an existing one) at
// https://console.cloud.google.com/
//
// 2. Enable the "Generative Language API" for your project at
// https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com
//
// 3. Configure the OAuth consent screen at
// https://console.cloud.google.com/apis/credentials/consent
//   - Choose "External" user type (unless you have a Workspace org)
//   - Fill in the required app name and support email
//   - Add scopes: "https://www.googleapis.com/auth/cloud-platform",
//     "https://www.googleapis.com/auth/generative-language.tuning",
//     and "https://www.googleapis.com/auth/generative-language.retriever"
//   - Add your Google account as a test user (required while the app is in
//     "Testing" publishing status)
//
// 4. Create OAuth2 credentials at
// https://console.cloud.google.com/apis/credentials
//   - Click "+ CREATE CREDENTIALS" → "OAuth client ID"
//   - Application type: "Desktop app"
//   - Name: any descriptive name (e.g. "genai CLI")
//   - Click "Create", then copy the Client ID and Client secret
//
// # Usage
//
//	go run . -client-id=YOUR_CLIENT_ID -client-secret=YOUR_CLIENT_SECRET
//
// Or set the environment variables GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.
//
// Works on Linux, macOS and Windows.

package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/gemini"
)

const (
	googleAuthURL  = "https://accounts.google.com/o/oauth2/v2/auth"
	googleTokenURL = "https://oauth2.googleapis.com/token"
	// The generativelanguage.googleapis.com API does not declare OAuth2 scopes
	// on generateContent / models.list in its discovery doc. Google's cookbook
	// (Authentication_with_OAuth.ipynb) uses all three scopes below. The
	// cloud-platform scope provides the broad access needed for content
	// generation; the two narrower scopes cover tuning and semantic-retrieval.
	googleScope = "https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/generative-language.tuning https://www.googleapis.com/auth/generative-language.retriever"
)

func main() {
	clientID := flag.String("client-id", os.Getenv("GOOGLE_CLIENT_ID"), "OAuth2 client ID (or GOOGLE_CLIENT_ID env)")
	clientSecret := flag.String("client-secret", os.Getenv("GOOGLE_CLIENT_SECRET"), "OAuth2 client secret (or GOOGLE_CLIENT_SECRET env)")
	flag.Parse()
	if *clientID == "" || *clientSecret == "" {
		log.Fatal("-client-id and -client-secret are required (or set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET); use http://localhost:8080/callback as Authorized redirect URI")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	tok, err := getTokens(ctx, *clientID, *clientSecret)
	if err != nil {
		log.Fatal(err)
	}

	wrapper := genai.ProviderOptionTransportWrapper(func(t http.RoundTripper) http.RoundTripper {
		return &roundtrippers.Header{
			Header:    http.Header{"Authorization": {"Bearer " + tok.AccessToken}},
			Transport: t,
		}
	})
	c, err := gemini.New(ctx, genai.ModelGood, wrapper)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice. Answer succinctly."),
	}
	res, err := c.GenSync(ctx, msgs)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.String())
}

// Token cache.

// cachedTokens is the on-disk format for cached OAuth2 tokens.
type cachedTokens struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token,omitempty"`
}

// cachePath returns the path to the token cache file.
func cachePath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "genai-oauth2-example", "gemini-tokens.json"), nil
}

// load reads cached tokens from disk.
func (t *cachedTokens) load() error {
	p, err := cachePath()
	if err != nil {
		return err
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return err
	}
	return json.Unmarshal(b, t)
}

// save writes tokens to disk.
func (t *cachedTokens) save() error {
	p, err := cachePath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(p), 0o700); err != nil {
		return err
	}
	b, err := json.Marshal(t)
	if err != nil {
		return err
	}
	return os.WriteFile(p, b, 0o600)
}

// OAuth2 flow.

// getTokens returns valid tokens, using cached tokens if available and
// refreshing if needed. Falls back to a full browser login.
func getTokens(ctx context.Context, clientID, clientSecret string) (*cachedTokens, error) {
	if p, err := cachePath(); err == nil {
		fmt.Fprintf(os.Stderr, "Token cache: %s\n", p)
	}
	var tok cachedTokens
	if tok.load() == nil && tok.RefreshToken != "" {
		if err := tok.refresh(ctx, clientID, clientSecret); err == nil {
			fmt.Fprintf(os.Stderr, "Token refreshed.\n")
			if err := tok.save(); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: could not save refreshed token: %v\n", err)
			}
			return &tok, nil
		}
		fmt.Fprintf(os.Stderr, "Token refresh failed, re-authenticating...\n")
	}

	if err := tok.doBrowserLogin(ctx, clientID, clientSecret); err != nil {
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "Login successful!\n")
	if err := tok.save(); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not save token: %v\n", err)
	}
	return &tok, nil
}

// doBrowserLogin runs the full Authorization Code flow with PKCE.
func (t *cachedTokens) doBrowserLogin(ctx context.Context, clientID, clientSecret string) error {
	verifier, challenge, err := generatePKCE()
	if err != nil {
		return fmt.Errorf("generating PKCE: %w", err)
	}
	redirectURI, code, err := waitForCallback(ctx, googleAuthURL, url.Values{
		"client_id":             {clientID},
		"response_type":         {"code"},
		"scope":                 {googleScope},
		"code_challenge":        {challenge},
		"code_challenge_method": {"S256"},
		"access_type":           {"offline"},
	})
	if err != nil {
		return err
	}
	return t.exchangeCode(ctx, clientID, clientSecret, redirectURI, verifier, code)
}

// exchangeCode exchanges an authorization code for Google OAuth2 tokens.
func (t *cachedTokens) exchangeCode(ctx context.Context, clientID, clientSecret, redirectURI, verifier, code string) error {
	form := url.Values{
		"client_id":     {clientID},
		"client_secret": {clientSecret},
		"code":          {code},
		"code_verifier": {verifier},
		"grant_type":    {"authorization_code"},
		"redirect_uri":  {redirectURI},
	}
	req, err := http.NewRequestWithContext(ctx, "POST", googleTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("exchanging code for token: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("token exchange returned %s: %s", resp.Status, b)
	}
	return json.NewDecoder(resp.Body).Decode(t)
}

// refresh uses a refresh token to obtain a new access token.
func (t *cachedTokens) refresh(ctx context.Context, clientID, clientSecret string) error {
	form := url.Values{
		"client_id":     {clientID},
		"client_secret": {clientSecret},
		"grant_type":    {"refresh_token"},
		"refresh_token": {t.RefreshToken},
	}
	req, err := http.NewRequestWithContext(ctx, "POST", googleTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("refreshing token: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("refresh returned %s: %s", resp.Status, b)
	}
	old := t.RefreshToken
	if err := json.NewDecoder(resp.Body).Decode(t); err != nil {
		return fmt.Errorf("decoding refresh response: %w", err)
	}
	// Keep old refresh token if the server didn't rotate it.
	if t.RefreshToken == "" {
		t.RefreshToken = old
	}
	return nil
}

// Browser callback server.

// callbackResult holds the authorization code or error from the OAuth2 callback.
type callbackResult struct {
	code string
	err  error
}

// waitForCallback starts a local HTTP server, opens the browser, and waits
// for the authorization code callback.
func waitForCallback(ctx context.Context, authEndpoint string, params url.Values) (redirectURI, code string, err error) {
	state, err := randomString(32)
	if err != nil {
		return "", "", fmt.Errorf("generating state: %w", err)
	}

	ln, err := net.Listen("tcp", "localhost:8080")
	if err != nil {
		return "", "", fmt.Errorf("listening: %w", err)
	}
	defer ln.Close()
	port := ln.Addr().(*net.TCPAddr).Port
	redirectURI = fmt.Sprintf("http://localhost:%d/callback", port)

	ch := make(chan callbackResult, 1)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /callback", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("state") != state {
			ch <- callbackResult{err: fmt.Errorf("state mismatch")}
			http.Error(w, "State mismatch", http.StatusBadRequest)
			return
		}
		if e := r.URL.Query().Get("error"); e != "" {
			ch <- callbackResult{err: fmt.Errorf("authorization error: %s: %s", e, r.URL.Query().Get("error_description"))}
			http.Error(w, "Authorization failed: "+e, http.StatusBadRequest)
			return
		}
		c := r.URL.Query().Get("code")
		if c == "" {
			ch <- callbackResult{err: fmt.Errorf("no code in callback")}
			http.Error(w, "Missing code", http.StatusBadRequest)
			return
		}
		fmt.Fprint(w, "<html><body><h1>Authorization successful!</h1><p>You can close this tab.</p></body></html>")
		ch <- callbackResult{code: c}
	})
	srv := &http.Server{Handler: mux}
	go srv.Serve(ln)
	defer srv.Shutdown(context.WithoutCancel(ctx))

	params.Set("state", state)
	params.Set("redirect_uri", redirectURI)
	authURL := authEndpoint + "?" + params.Encode()

	fmt.Fprintf(os.Stderr, "Opening browser for Google login...\n")
	fmt.Fprintf(os.Stderr, "If the browser does not open, visit:\n%s\n", authURL)
	if err := openBrowser(authURL); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not open browser: %v\n", err)
	}

	var res callbackResult
	select {
	case res = <-ch:
	case <-ctx.Done():
		return "", "", ctx.Err()
	}
	if res.err != nil {
		return "", "", res.err
	}
	return redirectURI, res.code, nil
}

// Helpers.

func generatePKCE() (verifier, challenge string, err error) {
	b := make([]byte, 64)
	if _, err := rand.Read(b); err != nil {
		return "", "", err
	}
	verifier = base64.RawURLEncoding.EncodeToString(b)
	h := sha256.Sum256([]byte(verifier))
	challenge = base64.RawURLEncoding.EncodeToString(h[:])
	return verifier, challenge, nil
}

func randomString(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

func openBrowser(u string) error {
	switch runtime.GOOS {
	case "linux":
		return exec.Command("xdg-open", u).Start()
	case "darwin":
		return exec.Command("open", u).Start()
	case "windows":
		return exec.Command("cmd", "/c", "start", strings.ReplaceAll(u, "&", "^&")).Start()
	default:
		return fmt.Errorf("unsupported platform %s", runtime.GOOS)
	}
}
