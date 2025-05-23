package sse

import (
	"errors"
	"strings"
	"testing"
)

type testResponse struct {
	Text string `json:"text"`
}

func TestProcessSSE(t *testing.T) {
	tests := []struct {
		name          string
		input         string
		expectedMsgs  []testResponse
		expectedError bool
	}{
		{
			name:  "basic processing",
			input: "data: {\"text\":\"message 1\"}\n\ndata: {\"text\":\"message 2\"}\n\ndata: [DONE]\n\n",
			expectedMsgs: []testResponse{
				{Text: "message 1"},
				{Text: "message 2"},
			},
			expectedError: false,
		},
		{
			name:          "invalid json",
			input:         "data: {invalid json}\n\n",
			expectedMsgs:  nil,
			expectedError: true,
		},
		{
			name:  "with keep-alive",
			input: "data: {\"text\":\"message 1\"}\n\n: keep-alive\n\ndata: {\"text\":\"message 2\"}\n\n",
			expectedMsgs: []testResponse{
				{Text: "message 1"},
				{Text: "message 2"},
			},
			expectedError: false,
		},
		{
			name:          "unexpected format",
			input:         "unexpected: {\"text\":\"message\"}\n\n",
			expectedMsgs:  nil,
			expectedError: true,
		},
		{
			name:          "event prefix is ignored",
			input:         "event: message\n\ndata: {\"text\":\"message\"}\n\n",
			expectedMsgs:  []testResponse{{Text: "message"}},
			expectedError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reader := strings.NewReader(tt.input)
			outChan := make(chan testResponse, len(tt.expectedMsgs)+1)

			err := Process(reader, outChan, nil)
			close(outChan)

			if (err != nil) != tt.expectedError {
				t.Errorf("ProcessSSE() error = %v, expectedError %v", err, tt.expectedError)
				return
			}

			var receivedMsgs []testResponse
			for msg := range outChan {
				receivedMsgs = append(receivedMsgs, msg)
			}

			if len(receivedMsgs) != len(tt.expectedMsgs) {
				t.Errorf("ProcessSSE() received %d messages, expected %d",
					len(receivedMsgs), len(tt.expectedMsgs))
				return
			}

			for i, expected := range tt.expectedMsgs {
				if receivedMsgs[i].Text != expected.Text {
					t.Errorf("ProcessSSE() message[%d] = %v, expected %v",
						i, receivedMsgs[i], expected)
				}
			}
		})
	}
}

func TestProcessSSEWithReader(t *testing.T) {
	// Test with a reader that returns an error
	errorReader := &errorReaderMock{err: errors.New("read error")}
	outChan := make(chan testResponse)

	err := Process(errorReader, outChan, nil)
	if err == nil {
		t.Errorf("ProcessSSE() with error reader should return an error")
	}
}

// Mock implementation of io.Reader that returns an error
type errorReaderMock struct {
	err error
}

func (e *errorReaderMock) Read(p []byte) (n int, err error) {
	return 0, e.err
}
