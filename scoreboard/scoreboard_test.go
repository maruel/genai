// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package scoreboard_test

import (
	"net/http"
	"os"
	"testing"

	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/internaltest"
	"github.com/maruel/genai/providers/cerebras"
	"github.com/maruel/genai/scoreboard"
)

func TestCreateScenario(t *testing.T) {
	models, err2 := getClient(t, "").ListModels(t.Context())
	if err2 != nil {
		t.Fatal(err2)
	}
	for _, m := range models {
		t.Run(m.GetID(), func(t *testing.T) {
			c := getClient(t, m.GetID())
			_, err := scoreboard.CreateScenario(c)
			if err != nil || err.Error() != "implement me" {
				t.Skip("implement me")
			} else {
				t.Fatalf("progress? %v", err)
			}
		})
	}
}

func getClient(t *testing.T, m string) *cerebras.Client {
	testRecorder.Signal(t)
	t.Parallel()
	return getClientInner(t, "", m)
}

func getClientInner(t *testing.T, apiKey, m string) *cerebras.Client {
	if apiKey == "" && os.Getenv("CEREBRAS_API_KEY") == "" {
		apiKey = "<insert_api_key_here>"
	}
	c, err := cerebras.New(apiKey, m, func(h http.RoundTripper) http.RoundTripper { return testRecorder.Record(t, h) })
	if err != nil {
		t.Fatal(err)
	}
	return c
}

var testRecorder *internaltest.Records

func TestMain(m *testing.M) {
	testRecorder = internaltest.NewRecords()
	code := m.Run()
	os.Exit(max(code, testRecorder.Close()))
}

func init() {
	internal.BeLenient = false
}
