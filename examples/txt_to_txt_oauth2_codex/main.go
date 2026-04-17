// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Demonstrates a full OAuth2 Authorization Code flow with PKCE to
// authenticate against the ChatGPT backend API, using the same mechanism as
// Codex CLI (https://github.com/openai/codex).
//
// Tokens are cached in the user config directory and refreshed automatically.
//
// Usage:
//
//	go run .
//
// Works on Linux, macOS and Windows.

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
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

	"github.com/maruel/genai"
	"github.com/maruel/genai/providers/openairesponses"
	"github.com/maruel/roundtrippers"
)

const (
	// OpenAI OAuth2 issuer, as used by Codex CLI.
	openAIIssuer = "https://auth.openai.com"
	// Public client ID used by Codex CLI for the OAuth2 flow.
	openAIClientID = "app_EMoamEEZ73f0CkXaXp7hrann"
	// Scopes requested during the OAuth2 flow.
	openAIScopes = "openid profile email offline_access api.connectors.read api.connectors.invoke"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	tok, err := getTokens(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// The ChatGPT backend requires the ChatGPT-Account-ID header alongside
	// the Bearer token.
	var opts []genai.ProviderOption
	opts = append(opts,
		genai.ProviderOptionModel("gpt-5.4-mini"),
		genai.ProviderOptionRemote("https://chatgpt.com/backend-api/codex"),
		genai.ProviderOptionAPIKey(tok.AccessToken),
	)
	if tok.AccountID != "" {
		opts = append(opts, genai.ProviderOptionTransportWrapper(func(t http.RoundTripper) http.RoundTripper {
			return &roundtrippers.Header{
				Header:    http.Header{"ChatGPT-Account-ID": {tok.AccountID}},
				Transport: t,
			}
		}))
	}
	c, err := openairesponses.New(ctx, opts...)
	if err != nil {
		log.Fatal(err)
	}
	msgs := genai.Messages{
		genai.NewTextMessage("Give me a life advice that sounds good but is a bad idea in practice. Answer succinctly."),
	}
	// The ChatGPT backend requires streaming; GenSync is not supported.
	sp := genai.GenOptionText{SystemPrompt: "You are a helpful assistant."}
	chunks, usage := c.GenStream(ctx, msgs, &sp)
	for chunk := range chunks {
		fmt.Print(chunk.Text)
	}
	fmt.Println()
	if _, err := usage(); err != nil {
		log.Fatal(err)
	}
}

// Token cache.

// cachedTokens is the on-disk format for cached OAuth2 tokens.
type cachedTokens struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	AccountID    string `json:"account_id,omitempty"`
}

// cachePath returns the path to the token cache file.
func cachePath() (string, error) {
	dir, err := os.UserConfigDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(dir, "genai-oauth2-example", "codex-tokens.json"), nil
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
func getTokens(ctx context.Context) (*cachedTokens, error) {
	if p, err := cachePath(); err == nil {
		fmt.Fprintf(os.Stderr, "Token cache: %s\n", p)
	}
	var tok cachedTokens
	if tok.load() == nil && tok.RefreshToken != "" {
		if refreshed, err := refreshToken(ctx, &tok); err == nil {
			fmt.Fprintf(os.Stderr, "Token refreshed.\n")
			if err := refreshed.save(); err != nil {
				fmt.Fprintf(os.Stderr, "Warning: could not save refreshed token: %v\n", err)
			}
			return refreshed, nil
		}
		fmt.Fprintf(os.Stderr, "Token refresh failed, re-authenticating...\n")
	}

	result, err := doBrowserLogin(ctx)
	if err != nil {
		return nil, err
	}
	fmt.Fprintf(os.Stderr, "Login successful!\n")
	if err := result.save(); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: could not save token: %v\n", err)
	}
	return result, nil
}

// doBrowserLogin runs the full Authorization Code flow with PKCE.
func doBrowserLogin(ctx context.Context) (*cachedTokens, error) {
	verifier, challenge, err := generatePKCE()
	if err != nil {
		return nil, fmt.Errorf("generating PKCE: %w", err)
	}
	redirectURI, code, err := waitForCallback(ctx, openAIIssuer+"/oauth/authorize", url.Values{
		"response_type":              {"code"},
		"client_id":                  {openAIClientID},
		"scope":                      {openAIScopes},
		"code_challenge":             {challenge},
		"code_challenge_method":      {"S256"},
		"codex_cli_simplified_flow":  {"true"},
		"id_token_add_organizations": {"true"},
		"originator":                 {"codex_cli_rs"},
	})
	if err != nil {
		return nil, err
	}
	tokens, err := exchangeCode(ctx, redirectURI, verifier, code)
	if err != nil {
		return nil, err
	}
	return &cachedTokens{
		AccessToken:  tokens.AccessToken,
		RefreshToken: tokens.RefreshToken,
		AccountID:    extractAccountID(tokens.IDToken),
	}, nil
}

// codeExchangeResponse is the JSON response from the authorization code exchange.
type codeExchangeResponse struct {
	IDToken      string `json:"id_token"`
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
}

// exchangeCode exchanges an authorization code for tokens.
func exchangeCode(ctx context.Context, redirectURI, verifier, code string) (*codeExchangeResponse, error) {
	form := url.Values{
		"grant_type":    {"authorization_code"},
		"client_id":     {openAIClientID},
		"code":          {code},
		"redirect_uri":  {redirectURI},
		"code_verifier": {verifier},
	}
	req, err := http.NewRequestWithContext(ctx, "POST", openAIIssuer+"/oauth/token", strings.NewReader(form.Encode()))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("exchanging code for tokens: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("token exchange returned %s: %s", resp.Status, b)
	}
	var tok codeExchangeResponse
	if err := json.NewDecoder(resp.Body).Decode(&tok); err != nil {
		return nil, fmt.Errorf("decoding token response: %w", err)
	}
	return &tok, nil
}

// refreshToken uses a refresh token to obtain a new access token.
func refreshToken(ctx context.Context, old *cachedTokens) (*cachedTokens, error) {
	body, err := json.Marshal(map[string]string{
		"client_id":     openAIClientID,
		"grant_type":    "refresh_token",
		"refresh_token": old.RefreshToken,
	})
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, "POST", openAIIssuer+"/oauth/token", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("refreshing token: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("refresh returned %s: %s", resp.Status, b)
	}
	var tok struct {
		IDToken      string `json:"id_token"`
		AccessToken  string `json:"access_token"`
		RefreshToken string `json:"refresh_token"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tok); err != nil {
		return nil, fmt.Errorf("decoding refresh response: %w", err)
	}
	result := &cachedTokens{
		AccessToken:  tok.AccessToken,
		RefreshToken: tok.RefreshToken,
		AccountID:    old.AccountID,
	}
	// Keep old refresh token if the server didn't rotate it.
	if result.RefreshToken == "" {
		result.RefreshToken = old.RefreshToken
	}
	// Update account ID if a new id_token was returned.
	if tok.IDToken != "" {
		if id := extractAccountID(tok.IDToken); id != "" {
			result.AccountID = id
		}
	}
	return result, nil
}

// extractAccountID decodes a JWT id_token and extracts the
// chatgpt_account_id from the https://api.openai.com/auth claim.
func extractAccountID(idToken string) string {
	parts := strings.SplitN(idToken, ".", 3)
	if len(parts) < 2 {
		fmt.Fprintf(os.Stderr, "Warning: id_token is not a valid JWT\n")
		return ""
	}
	// JWT payload is base64url-encoded without padding.
	b, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to decode id_token payload: %v\n", err)
		return ""
	}
	var claims struct {
		Auth struct {
			AccountID string `json:"chatgpt_account_id"`
		} `json:"https://api.openai.com/auth"`
	}
	if err := json.Unmarshal(b, &claims); err != nil {
		fmt.Fprintf(os.Stderr, "Warning: failed to parse id_token claims: %v\n", err)
		return ""
	}
	return claims.Auth.AccountID
}

// Browser callback server.

// callbackResult holds the authorization code or error from the OAuth2 callback.
type callbackResult struct {
	code string
	err  error
}

// waitForCallback starts a local HTTP server on port 1455 (matching the
// redirect URI registered for the Codex CLI client ID), opens the browser,
// and waits for the authorization code callback.
func waitForCallback(ctx context.Context, authEndpoint string, params url.Values) (redirectURI, code string, err error) {
	state, err := randomString(32)
	if err != nil {
		return "", "", fmt.Errorf("generating state: %w", err)
	}

	ln, err := net.Listen("tcp", "localhost:1455")
	if err != nil {
		return "", "", fmt.Errorf("listening on localhost:1455: %w", err)
	}
	defer ln.Close()
	redirectURI = "http://localhost:1455/auth/callback"

	ch := make(chan callbackResult, 1)

	mux := http.NewServeMux()
	mux.HandleFunc("GET /auth/callback", func(w http.ResponseWriter, r *http.Request) {
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

	fmt.Fprintf(os.Stderr, "Opening browser for OpenAI login...\n")
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
