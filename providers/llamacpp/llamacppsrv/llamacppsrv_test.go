// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package llamacppsrv

import (
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"testing"
)

func TestDownloadRelease(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		if runtime.GOOS == "windows" {
			t.Skip("test uses a POSIX shell script")
		}
		cache := t.TempDir()
		exe := filepath.Join(cache, "llama-server")
		want := 1234
		envName := "LD_LIBRARY_PATH"
		if runtime.GOOS == "darwin" {
			envName = "DYLD_LIBRARY_PATH"
		}
		script := `#!/bin/sh
case ":$` + envName + `:" in
*:` + cache + `:*) ;;
*) exit 2 ;;
esac
printf 'version: ` + strconv.Itoa(want) + ` test\n'
`
		if err := os.WriteFile(exe, []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}

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
		if runtime.GOOS == "windows" {
			t.Skip("test uses a POSIX shell script")
		}
		cache := t.TempDir()
		exe := filepath.Join(cache, "llama-server")
		if err := os.WriteFile(exe, []byte("#!/bin/sh\nprintf 'version: 1234\\n'\n"), 0o755); err != nil {
			t.Fatal(err)
		}

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
