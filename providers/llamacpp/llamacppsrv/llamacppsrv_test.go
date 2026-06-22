// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for llamacppsrv.

package llamacppsrv

import (
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"testing"
)

func TestMain(m *testing.M) {
	if os.Getenv("LLAMACPP_TEST_HELPER") == "1" {
		os.Exit(runDownloadReleaseHelper())
	}
	os.Exit(m.Run())
}

func TestDownloadRelease(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		cache := t.TempDir()
		exe := installHelperExecutable(t, cache)
		want := 1234
		t.Setenv("LLAMACPP_TEST_HELPER", "1")
		t.Setenv("LLAMACPP_TEST_CACHE", cache)
		t.Setenv("LLAMACPP_TEST_VERSION", strconv.Itoa(want))

		oldTransport := http.DefaultTransport
		http.DefaultTransport = forbidRoundTrip{t: t}
		t.Cleanup(func() { http.DefaultTransport = oldTransport })

		got, err := DownloadRelease(t.Context(), cache, want)
		if err != nil {
			t.Fatal(err)
		}
		if got != exe {
			t.Fatalf("expected %q, got %q", exe, got)
		}
	})
	t.Run("validExactVersion", func(t *testing.T) {
		cache := t.TempDir()
		exe := installHelperExecutable(t, cache)
		t.Setenv("LLAMACPP_TEST_HELPER", "1")
		t.Setenv("LLAMACPP_TEST_CACHE", cache)
		t.Setenv("LLAMACPP_TEST_VERSION", "1234")
		t.Setenv("LLAMACPP_TEST_VERSION_OUTPUT", "version: 1234\n")

		oldTransport := http.DefaultTransport
		http.DefaultTransport = forbidRoundTrip{t: t}
		t.Cleanup(func() { http.DefaultTransport = oldTransport })

		got, err := DownloadRelease(t.Context(), cache, 1234)
		if err != nil {
			t.Fatal(err)
		}
		if got != exe {
			t.Fatalf("expected %q, got %q", exe, got)
		}
	})
}

func installHelperExecutable(t *testing.T, cache string) string {
	src, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	suffix := ""
	if runtime.GOOS == "windows" {
		suffix = ".exe"
	}
	dst := filepath.Join(cache, "llama-server"+suffix)
	b, err := os.ReadFile(src)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dst, b, 0o755); err != nil {
		t.Fatal(err)
	}
	return dst
}

func runDownloadReleaseHelper() int {
	if !slices.Contains(os.Args[1:], "--version") {
		return 2
	}
	cache := os.Getenv("LLAMACPP_TEST_CACHE")
	switch runtime.GOOS {
	case "darwin":
		if !slices.Contains(filepath.SplitList(os.Getenv("DYLD_LIBRARY_PATH")), cache) {
			return 2
		}
	case "windows":
	default:
		if !slices.Contains(filepath.SplitList(os.Getenv("LD_LIBRARY_PATH")), cache) {
			return 2
		}
	}
	out := os.Getenv("LLAMACPP_TEST_VERSION_OUTPUT")
	if out == "" {
		out = fmt.Sprintf("version: %s test\n", os.Getenv("LLAMACPP_TEST_VERSION"))
	}
	if _, err := fmt.Print(out); err != nil {
		return 2
	}
	return 0
}

func TestParseBuildNumber(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			out  string
			want int
		}{
			{name: "plain", out: "version: 9383\n", want: 9383},
			{name: "suffix", out: "version: 9383 (bb771cbd2)\n", want: 9383},
			{name: "tag", out: "version: b9383\n", want: 9383},
		} {
			t.Run(tc.name, func(t *testing.T) {
				got, ok := parseBuildNumber([]byte(tc.out))
				if !ok {
					t.Fatal("failed to parse build number")
				}
				if got != tc.want {
					t.Fatalf("expected %d, got %d", tc.want, got)
				}
			})
		}
	})
	t.Run("error", func(t *testing.T) {
		if got, ok := parseBuildNumber([]byte("no version\n")); ok {
			t.Fatalf("expected parse failure, got %d", got)
		}
	})
}

var errUnexpectedHTTPRequest = errors.New("unexpected HTTP request")

type forbidRoundTrip struct {
	t *testing.T
}

func (f forbidRoundTrip) RoundTrip(*http.Request) (*http.Response, error) {
	f.t.Fatal(errUnexpectedHTTPRequest)
	return nil, errUnexpectedHTTPRequest
}
