// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// File-backed Codex credential loading and OAuth renewal.
// Source: openai/codex 49f0c6022596c361fe676a5c4738fb2be34e3a93,
// codex-rs/login/src/auth/{storage.rs,manager.rs} and token_data.rs.

package openairesponses

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gofrs/flock"
	"github.com/maruel/genai/base"
)

const authURL = "https://auth.openai.com/oauth/token"
const clientID = "app_EMoamEEZ73f0CkXaXp7hrann"
const maxCredentialBytes = 1 << 20
const loginAdvice = "sign in with codex login using cli_auth_credentials_store = \"file\" in Codex config.toml"

type oauthProvenance struct {
	Issuer   string `json:"issuer"`
	ClientID string `json:"client_id"`
}
type tokens struct {
	AccessToken     string           `json:"access_token"`
	RefreshToken    string           `json:"refresh_token"`
	IDToken         string           `json:"id_token"`
	AccountID       string           `json:"account_id"`
	OAuthProvenance *oauthProvenance `json:"oauth_provenance,omitzero"`
}
type credentialDocument struct {
	fields map[string]json.RawMessage
	tokens tokens
	userID string
}
type tokenClaims struct {
	Expires int64        `json:"exp"`
	Auth    tokenAccount `json:"https://api.openai.com/auth"`
}
type tokenAccount struct {
	AccountID    string `json:"chatgpt_account_id"`
	UserID       string `json:"chatgpt_user_id"`
	LegacyUserID string `json:"user_id"`
}
type credentialStore struct {
	accountID string
	userID    string
	path      string
	client    http.Client
}

func newCredentialStore() (*credentialStore, error) {
	dir := os.Getenv("CODEX_HOME")
	if dir == "" {
		dir = "~/.codex"
	}
	if dir == "~" || strings.HasPrefix(dir, "~/") {
		h, err := os.UserHomeDir()
		if err != nil {
			return nil, fmt.Errorf("find Codex credentials: %w", err)
		}
		if dir == "~" {
			dir = h
		} else {
			dir = filepath.Join(h, dir[2:])
		}
	}
	return &credentialStore{path: filepath.Join(dir, "auth.json"), client: http.Client{Transport: base.DefaultTransport, Timeout: 45 * time.Second, CheckRedirect: func(*http.Request, []*http.Request) error { return errors.New("refusing OAuth token redirect") }}}, nil
}

func (s *credentialStore) load() (*credentialDocument, error) {
	f, err := os.Open(s.path)
	if err != nil {
		return nil, fmt.Errorf("read Codex credentials: %w; %s", err, loginAdvice)
	}
	defer func() { _ = f.Close() }()
	b, err := io.ReadAll(io.LimitReader(f, maxCredentialBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read Codex credentials: %w", err)
	}
	if len(b) > maxCredentialBytes {
		return nil, errors.New("codex credential file exceeds 1 MiB")
	}
	var d credentialDocument
	if json.Unmarshal(b, &d.fields) != nil || d.fields == nil {
		return nil, errors.New("invalid Codex credential document")
	}
	var mode string
	if raw, ok := d.fields["auth_mode"]; ok && string(raw) != "null" {
		if json.Unmarshal(raw, &mode) != nil {
			return nil, errors.New("invalid Codex authentication mode")
		}
	}
	if mode != "" && mode != "chatgpt" {
		return nil, fmt.Errorf("codex credentials are not a ChatGPT subscription; %s", loginAdvice)
	}
	if json.Unmarshal(d.fields["tokens"], &d.tokens) != nil || d.tokens.AccessToken == "" || d.tokens.AccountID == "" {
		return nil, fmt.Errorf("codex subscription tokens or account ID missing; %s", loginAdvice)
	}
	if s.accountID != "" && s.accountID != d.tokens.AccountID {
		return nil, errors.New("codex account changed; create a new subscription client")
	}
	identity, err := credentialIdentity(&d.tokens)
	if err != nil {
		return nil, err
	}
	if s.userID != "" && s.userID != identity {
		return nil, errors.New("codex user changed; create a new subscription client")
	}
	d.userID = identity

	if p := d.tokens.OAuthProvenance; p != nil && (p.Issuer != "https://auth.openai.com" || p.ClientID == "") {
		return nil, errors.New("unsupported Codex OAuth authority")
	}
	return &d, nil
}

// credentialIdentity reads stable user identity from locally trusted credentials.
// The claims identify the login generation; bearer validity is checked upstream.
func credentialIdentity(t *tokens) (string, error) {
	access, err := parseClaims(t.AccessToken)
	if err != nil {
		return "", err
	}
	user := access.Auth.UserID
	if user == "" {
		user = access.Auth.LegacyUserID
	}
	if t.IDToken != "" {
		id, err := parseClaims(t.IDToken)
		if err != nil {
			return "", errors.New("invalid Codex ID token")
		}
		idUser := id.Auth.UserID
		if idUser == "" {
			idUser = id.Auth.LegacyUserID
		}
		if user != "" && idUser != "" && user != idUser {
			return "", errors.New("codex access and ID tokens belong to different users")
		}
		if user == "" {
			user = idUser
		}
	}
	if user == "" {
		return "", fmt.Errorf("codex credentials have no stable user identity; %s", loginAdvice)
	}
	return user, nil
}

func parseClaims(token string) (*tokenClaims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, errors.New("invalid Codex access token")
	}
	b, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil, errors.New("invalid Codex access token payload")
	}
	var c tokenClaims
	if json.Unmarshal(b, &c) != nil || c.Expires <= 0 {
		return nil, errors.New("codex access token has no valid expiry")
	}
	return &c, nil
}

func (s *credentialStore) current(ctx context.Context) (_ *tokens, err error) {
	d, err := s.load()
	if err != nil {
		return nil, err
	}
	claims, err := parseClaims(d.tokens.AccessToken)
	if err != nil {
		return nil, err
	}
	if time.Until(time.Unix(claims.Expires, 0)) > time.Minute {
		return &d.tokens, nil
	}
	// Codex uses an exclusive filesystem lock on auth.lock for credential commits.
	// Hold it across reloading and renewal so our clients cannot rotate the same
	// refresh token concurrently; compare again before committing alongside Codex.
	lock := flock.New(filepath.Join(filepath.Dir(s.path), "auth.lock"))
	ok, err := lock.TryLockContext(ctx, 10*time.Millisecond)
	if err != nil {
		return nil, fmt.Errorf("lock Codex credentials: %w", err)
	}
	if !ok {
		return nil, ctx.Err()
	}
	defer func() { err = errors.Join(err, lock.Close()) }()
	d, err = s.load()
	if err != nil {
		return nil, err
	}
	claims, err = parseClaims(d.tokens.AccessToken)
	if err != nil {
		return nil, err
	}
	if time.Until(time.Unix(claims.Expires, 0)) > time.Minute {
		return &d.tokens, nil
	}
	if d.tokens.RefreshToken == "" {
		return nil, fmt.Errorf("codex token expired without a refresh token; %s", loginAdvice)
	}
	id := clientID
	if d.tokens.OAuthProvenance != nil {
		id = d.tokens.OAuthProvenance.ClientID
	}
	body, err := json.Marshal(refreshRequest{GrantType: "refresh_token", ClientID: id, RefreshToken: d.tokens.RefreshToken}) //nolint:gosec // OAuth requires sending this credential only to the fixed token endpoint.
	if err != nil {
		return nil, errors.New("encode Codex token renewal")
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, authURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("renew Codex token: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("renew Codex token: HTTP %d; %s", resp.StatusCode, loginAdvice)
	}
	b, err := io.ReadAll(io.LimitReader(resp.Body, maxCredentialBytes+1))
	if err != nil {
		return nil, fmt.Errorf("read Codex token renewal: %w", err)
	}
	var next refreshResponse
	if len(b) > maxCredentialBytes || json.Unmarshal(b, &next) != nil || next.AccessToken == "" {
		return nil, errors.New("invalid Codex token renewal response")
	}
	c, err := parseClaims(next.AccessToken)
	if err != nil {
		return nil, err
	}
	if c.Expires <= time.Now().Unix() || c.Auth.AccountID != "" && c.Auth.AccountID != d.tokens.AccountID {
		return nil, errors.New("codex token renewal returned an expired token or a different account")
	}
	candidate := d.tokens
	candidate.AccessToken = next.AccessToken
	if next.IDToken != "" {
		candidate.IDToken = next.IDToken
	}
	identity, err := credentialIdentity(&candidate)
	if err != nil {
		return nil, err
	}
	if identity != d.userID {
		return nil, errors.New("codex token renewal returned a different user")
	}
	current, err := s.load()
	if err != nil {
		return nil, err
	}
	if !equalDocuments(d, current) {
		return nil, errors.New("codex credentials changed during token renewal; retry with the current login")
	}
	var fields map[string]json.RawMessage
	if json.Unmarshal(d.fields["tokens"], &fields) != nil {
		return nil, errors.New("invalid Codex token document")
	}
	updates := map[string]string{"access_token": next.AccessToken}
	if next.RefreshToken != "" {
		updates["refresh_token"] = next.RefreshToken
	}
	if next.IDToken != "" {
		updates["id_token"] = next.IDToken
	}
	for k, v := range updates {
		fields[k], err = json.Marshal(v)
		if err != nil {
			return nil, err
		}
	}
	d.fields["tokens"], err = json.Marshal(fields)
	if err != nil {
		return nil, err
	}
	d.fields["last_refresh"], err = json.Marshal(time.Now().UTC().Format(time.RFC3339))
	if err != nil {
		return nil, err
	}
	if err := s.save(d); err != nil {
		return nil, err
	}
	d, err = s.load()
	if err != nil {
		return nil, err
	}
	return &d.tokens, nil
}

func equalDocuments(a, b *credentialDocument) bool {
	x, err := json.Marshal(a.fields)
	if err != nil {
		return false
	}
	y, err := json.Marshal(b.fields)
	if err != nil {
		return false
	}
	return bytes.Equal(x, y)
}

func (s *credentialStore) save(d *credentialDocument) (err error) {
	// Resolve the target so replacing credentials preserves a symlinked auth.json.
	target, err := filepath.EvalSymlinks(s.path)
	if err != nil {
		return fmt.Errorf("resolve Codex credential file: %w", err)
	}
	b, err := json.MarshalIndent(d.fields, "", "  ")
	if err != nil {
		return errors.New("encode Codex credentials")
	}
	f, err := os.CreateTemp(filepath.Dir(target), ".codex-auth-*")
	if err != nil {
		return fmt.Errorf("create Codex credential replacement: %w", err)
	}
	defer func() {
		if e := os.Remove(f.Name()); e != nil && !os.IsNotExist(e) {
			err = errors.Join(err, e)
		}
	}()
	if _, err = f.Write(b); err != nil {
		return errors.Join(err, f.Close())
	}
	if err = f.Sync(); err != nil {
		return errors.Join(err, f.Close())
	}
	if err := f.Close(); err != nil {
		return err
	}
	if err = os.Rename(f.Name(), target); err != nil {
		return fmt.Errorf("commit renewed Codex credentials: %w", err)
	}
	return nil
}

type refreshRequest struct {
	GrantType    string `json:"grant_type"`
	ClientID     string `json:"client_id"`
	RefreshToken string `json:"refresh_token"`
}
type refreshResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
}
type authTransport struct {
	store     *credentialStore
	transport http.RoundTripper
	appName   string
}

func (t *authTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if req.URL.Scheme != "https" || req.URL.Host != "chatgpt.com" || req.URL.User != nil || (req.URL.Path != "/backend-api/codex/responses" && req.URL.Path != "/backend-api/codex/models") {
		return nil, errors.New("refusing to send Codex credentials outside the subscription API")
	}
	tok, err := t.store.current(req.Context())
	if err != nil {
		return nil, err
	}
	r := req.Clone(req.Context())
	r.Header.Set("Authorization", "Bearer "+tok.AccessToken)
	r.Header.Set("Chatgpt-Account-Id", tok.AccountID)
	r.Header.Set("Originator", t.appName)
	r.Header.Set("User-Agent", t.appName)
	return t.transport.RoundTrip(r)
}
