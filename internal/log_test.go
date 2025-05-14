// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

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
	t              *testing.T
	calledWith     *http.Request
	responseToSend *http.Response
	errorToReturn  error
}

func (m *mockRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	m.calledWith = req
	return m.responseToSend, m.errorToReturn
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
		w.Write(append(body, []byte(" - response")...))
	}))
	defer server.Close()

	// Create a client with the LogTransport
	client := &http.Client{
		Transport: LogTransport(http.DefaultTransport),
	}

	// Create a request with a body
	reqBody := "test data"
	req, err := http.NewRequest("POST", server.URL, strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	// Add GetBody for replayability (which LogTransport uses)
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader(reqBody)), nil
	}

	// Send the request
	res, err := client.Do(req)
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer res.Body.Close()

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
}

// TestLogTransportWithMockTransport tests the LogTransport directly with a mock transport
// to verify it correctly forwards calls to the underlying transport.
func TestLogTransportWithMockTransport(t *testing.T) {
	// Create mock response
	respBody := "mock response data"
	mockResp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(strings.NewReader(respBody)),
		Header:     make(http.Header),
	}

	mock := &mockRoundTripper{t: t, responseToSend: mockResp}
	loggingTransport := LogTransport(mock)
	reqBody := "test request"
	req, err := http.NewRequest("POST", "http://example.com", strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}
	req.GetBody = func() (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader(reqBody)), nil
	}

	// Use the transport directly
	res, err := loggingTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	// Verify the transport was called with our request
	if mock.calledWith == nil {
		t.Fatal("Underlying transport was not called")
	}
	if mock.calledWith.URL.String() != req.URL.String() {
		t.Errorf("Expected URL %q, got %q", req.URL.String(), mock.calledWith.URL.String())
	}
	if mock.calledWith.Method != req.Method {
		t.Errorf("Expected method %q, got %q", req.Method, mock.calledWith.Method)
	}

	// Read the response body
	actualRespBody, err := io.ReadAll(res.Body)
	if err != nil {
		t.Fatalf("Failed to read response body: %v", err)
	}
	res.Body.Close()
	if string(actualRespBody) != respBody {
		t.Errorf("Expected response body %q, got %q", respBody, string(actualRespBody))
	}
}

// TestLogTransportWithNilResponse tests that LogTransport properly handles
// a nil response body without panicking.
func TestLogTransportWithNilResponse(t *testing.T) {
	// Create a mock response with nil body
	mockResp := &http.Response{
		StatusCode: http.StatusNoContent,
		Body:       nil,
		Header:     make(http.Header),
	}

	mock := &mockRoundTripper{t: t, responseToSend: mockResp}
	loggingTransport := LogTransport(mock)
	req, err := http.NewRequest("GET", "http://example.com", nil)
	if err != nil {
		t.Fatalf("Failed to create request: %v", err)
	}

	// Use the transport directly - this should not panic
	res, err := loggingTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("RoundTrip failed: %v", err)
	}

	// Verify we received the correct status code
	if res.StatusCode != http.StatusNoContent {
		t.Errorf("Expected status %d, got %d", http.StatusNoContent, res.StatusCode)
	}
}

// TestLogTransportWithEmptyBody tests that LogTransport properly handles
// a non-nil but empty response body.
func TestLogTransportWithEmptyBody(t *testing.T) {
	// Create a mock response with empty body
	mockResp := &http.Response{
		StatusCode: http.StatusOK,
		Body:       io.NopCloser(bytes.NewReader(nil)),
		Header:     make(http.Header),
	}

	mock := &mockRoundTripper{t: t, responseToSend: mockResp}
	loggingTransport := LogTransport(mock)
	req, err := http.NewRequest("GET", "http://example.com", nil)
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
	res.Body.Close()
	if len(body) != 0 {
		t.Errorf("Expected empty body, got %q", string(body))
	}
}
