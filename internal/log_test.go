// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the internal logging utilities.

package internal

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// mockRoundTripper implements http.RoundTripper for testing.
type mockRoundTripper struct {
	responseToSend *http.Response
}

func (m *mockRoundTripper) RoundTrip(*http.Request) (*http.Response, error) {
	return m.responseToSend, nil
}

// TestLogTransport verifies that LogTransport creates a wrapper that correctly
// passes requests to the underlying transport and allows response body to be read.
func TestLogTransport(t *testing.T) {
	// Create a test server that will respond with known data
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Read the request body
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("Failed to read request body: %v", err)
		}

		// Echo the request body and add some extra data
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(append(body, []byte(" - response")...))
	}))
	defer server.Close()

	// Create a client with the LogTransport
	c := &http.Client{
		Transport: LogTransport(http.DefaultTransport),
	}

	// Create a request with a body
	reqBody := "test data"
	req, err := http.NewRequest(http.MethodPost, server.URL, strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	// Add GetBody for replayability (which LogTransport uses)
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader(reqBody)), nil
	}

	// Send the request
	res, err := c.Do(req)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer func() { _ = res.Body.Close() }()

	// Read the response
	respBody, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("Failed to read response: %v", err)
	}

	// Verify the response body is as expected
	expectedResponse := "test data - response"
	if string(respBody) != expectedResponse {
		t.Errorf("Expected response body %q, got %q", expectedResponse, string(respBody))
	}

	t.Run("empty_body", func(t *testing.T) {
		// Create a mock response with empty body
		mockResp := &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(bytes.NewReader(nil)),
			Header:     make(http.Header),
		}

		mock := &mockRoundTripper{responseToSend: mockResp}
		loggingTransport := LogTransport(mock)
		req, err := http.NewRequest(http.MethodGet, "http://example.com", http.NoBody)
		if err != nil {
			t.Fatalf("Failed to create request: %v", err)
		}

		// Use the transport directly
		res, err := loggingTransport.RoundTrip(req)
		if err != nil {
			t.Fatalf("RoundTrip failed: %v", err)
		}

		// Read the response body
		body, err := io.ReadAll(res.Body)
		if err != nil {
			t.Fatalf("Failed to read response body: %v", err)
		}
		_ = res.Body.Close()
		if len(body) != 0 {
			t.Errorf("Expected empty body, got %q", string(body))
		}
	})
}
