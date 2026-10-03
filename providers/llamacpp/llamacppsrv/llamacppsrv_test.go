// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for llamacppsrv.

package llamacppsrv

import (
	"archive/tar"
	"archive/zip"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/maruel/genai/internal/ghrelease"
)

func TestMain(m *testing.M) {
	if os.Getenv("LLAMACPP_TEST_HELPER") == "1" {
		os.Exit(runDownloadReleaseHelper())
	}
	os.Exit(m.Run())
}

func TestDownloadRelease(t *testing.T) {
	t.Run("extraction", testDownloadReleaseExtraction)
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
			{name: "versioned", out: "version: 0.5.0-dev (build 11146, commit 7fe450e19)\n", want: 11146},
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
		for _, out := range []string{"no version\n", "version: 0.5.0-dev\n", "version: 0.5.0\n"} {
			if got, ok := parseBuildNumber([]byte(out)); ok {
				t.Fatalf("expected parse failure for %q, got %d", out, got)
			}
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

func TestReleaseAssets(t *testing.T) {
	// Names from the v0.5.0 binary release. The runtime asset comes first in
	// GitHub's response, which previously selected an archive without a server.
	rel := &ghrelease.Release{TagName: "b11146"}
	for _, name := range []string{
		"cudart-llama-bin-win-cuda-12.4-x64.zip",
		"llama-b11146-bin-macos-arm64.tar.gz",
		"llama-b11146-bin-macos-x64.tar.gz",
		"llama-b11146-bin-ubuntu-arm64.tar.gz",
		"llama-b11146-bin-ubuntu-x64.tar.gz",
		"llama-b11146-bin-win-cpu-arm64.zip",
		"llama-b11146-bin-win-cpu-x64.zip",
		"llama-b11146-bin-win-cuda-12.4-x64.zip",
	} {
		rel.Assets = append(rel.Assets, ghrelease.Asset{Name: name})
	}
	t.Run("valid", func(t *testing.T) {
		for _, tc := range []struct {
			os, arch, name string
			cuda           bool
		}{
			{"darwin", "amd64", "llama-b11146-bin-macos-x64.tar.gz", false},
			{"darwin", "arm64", "llama-b11146-bin-macos-arm64.tar.gz", false},
			{"linux", "amd64", "llama-b11146-bin-ubuntu-x64.tar.gz", false},
			{"linux", "arm64", "llama-b11146-bin-ubuntu-arm64.tar.gz", false},
			{"windows", "amd64", "llama-b11146-bin-win-cpu-x64.zip", false},
			{"windows", "arm64", "llama-b11146-bin-win-cpu-arm64.zip", false},
			{"windows", "amd64", "llama-b11146-bin-win-cuda-12.4-x64.zip", true},
		} {
			t.Run(tc.os+"/"+tc.arch+"/"+strconv.FormatBool(tc.cuda), func(t *testing.T) {
				got, err := releaseAssets(rel, tc.os, tc.arch, tc.cuda)
				if err != nil {
					t.Fatal(err)
				}
				if got[0].Name != tc.name {
					t.Fatalf("binary = %q, want %q", got[0].Name, tc.name)
				}
				if tc.cuda {
					if len(got) != 2 || got[1].Name != "cudart-llama-bin-win-cuda-12.4-x64.zip" {
						t.Fatalf("missing matching CUDA runtime: %v", got)
					}
				} else if len(got) != 1 {
					t.Fatalf("unexpected extra assets: %v", got)
				}
			})
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, tc := range []struct{ os, arch string }{{"freebsd", "amd64"}, {"darwin", "386"}} {
			if _, err := releaseAssets(rel, tc.os, tc.arch, false); err == nil {
				t.Fatalf("accepted unsupported platform %s/%s", tc.os, tc.arch)
			}
		}
		empty := &ghrelease.Release{TagName: "b11146"}
		if _, err := releaseAssets(empty, "linux", "arm64", false); err == nil {
			t.Fatal("accepted missing binary archive")
		}
		empty.Assets = []ghrelease.Asset{{Name: "llama-b11146-bin-win-cuda-12.4-x64.zip"}}
		if _, err := releaseAssets(empty, "windows", "amd64", true); err == nil {
			t.Fatal("accepted missing CUDA runtime")
		}
	})
}

func TestDownloadVersion(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		cache := t.TempDir()
		exe := installHelperExecutable(t, cache)
		t.Setenv("LLAMACPP_TEST_HELPER", "1")
		t.Setenv("LLAMACPP_TEST_CACHE", cache)
		t.Setenv("LLAMACPP_TEST_VERSION_OUTPUT", "version: 0.5.0-dev (build 11146, commit 7fe450e19)\n")
		old := http.DefaultTransport
		n := 0
		http.DefaultTransport = releaseRoundTrip{fn: func(r *http.Request) (*http.Response, error) {
			n++
			body := ""
			switch r.URL.Path {
			case "/repos/ggml-org/llama.cpp/releases/tags/v0.5.0":
				body = `{"tag_name":"v0.5.0","assets":[{"name":"nightly-tag.txt","browser_download_url":"https://example.com/nightly-tag.txt"}]}`
			case "/nightly-tag.txt":
				body = "b11146\n"
			default:
				t.Fatalf("unexpected request: %s", r.URL)
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
		}}
		t.Cleanup(func() { http.DefaultTransport = old })
		got, err := DownloadVersion(t.Context(), cache, "v0.5.0")
		if err != nil {
			t.Fatal(err)
		}
		if got != exe || n != 2 {
			t.Fatalf("got %q after %d requests, want %q after 2", got, n, exe)
		}
	})
	t.Run("resolutionError", func(t *testing.T) {
		for _, tc := range []struct {
			name, release, marker string
		}{
			{"missingAsset", `{"tag_name":"v0.5.0","assets":[]}`, ""},
			{"invalidMarker", `{"tag_name":"v0.5.0","assets":[{"name":"nightly-tag.txt","browser_download_url":"https://example.com/nightly-tag.txt"}]}`, "v0.5.0"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				old := http.DefaultTransport
				http.DefaultTransport = releaseRoundTrip{fn: func(r *http.Request) (*http.Response, error) {
					body := tc.release
					if r.URL.Host == "example.com" {
						body = tc.marker
					}
					return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
				}}
				t.Cleanup(func() { http.DefaultTransport = old })
				if _, err := DownloadVersion(t.Context(), t.TempDir(), "v0.5.0"); err == nil {
					t.Fatal("accepted invalid stable release")
				}
			})
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, version := range []string{"", "0.5.0", "v0.5", "b0", "b-1", "../v0.5.0"} {
			if _, err := DownloadVersion(t.Context(), t.TempDir(), version); err == nil {
				t.Fatalf("accepted invalid release tag %q", version)
			}
		}
	})
}

type releaseRoundTrip struct {
	fn func(*http.Request) (*http.Response, error)
}

func (r releaseRoundTrip) RoundTrip(req *http.Request) (*http.Response, error) {
	return r.fn(req)
}

func testDownloadReleaseExtraction(t *testing.T) {
	for _, valid := range []bool{true, false} {
		t.Run(strconv.FormatBool(valid), func(t *testing.T) {
			t.Setenv("PATH", "") // Select the Windows CPU archive on CUDA hosts.
			cache := t.TempDir()
			exe := "llama-server"
			if runtime.GOOS == "windows" {
				exe += ".exe"
			}
			name := "unrelated"
			if !valid {
				if err := os.WriteFile(filepath.Join(cache, exe), []byte("old binary"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			if valid {
				name = exe
			}
			var buf bytes.Buffer
			if runtime.GOOS == "windows" {
				z := zip.NewWriter(&buf)
				w, err := z.Create(name)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := w.Write([]byte("binary")); err != nil {
					t.Fatal(err)
				}
				if err := z.Close(); err != nil {
					t.Fatal(err)
				}
			} else {
				gz := gzip.NewWriter(&buf)
				tr := tar.NewWriter(gz)
				if err := tr.WriteHeader(&tar.Header{Name: "llama-b1234/" + name, Size: 6, Mode: 0o755}); err != nil {
					t.Fatal(err)
				}
				if _, err := tr.Write([]byte("binary")); err != nil {
					t.Fatal(err)
				}
				if err := tr.WriteHeader(&tar.Header{Name: "llama-b1234/libllama.so", Size: 6, Mode: 0o755}); err != nil {
					t.Fatal(err)
				}
				if _, err := tr.Write([]byte("shared")); err != nil {
					t.Fatal(err)
				}
				if err := tr.WriteHeader(&tar.Header{Name: "llama-b1234/libllama.so.0", Typeflag: tar.TypeSymlink, Linkname: "libllama.so"}); err != nil {
					t.Fatal(err)
				}
				if err := tr.Close(); err != nil {
					t.Fatal(err)
				}
				if err := gz.Close(); err != nil {
					t.Fatal(err)
				}
			}
			rel := &ghrelease.Release{TagName: "b1234"}
			for _, suffix := range []string{"macos-arm64.tar.gz", "macos-x64.tar.gz", "ubuntu-arm64.tar.gz", "ubuntu-x64.tar.gz", "win-cpu-arm64.zip", "win-cpu-x64.zip"} {
				rel.Assets = append(rel.Assets, ghrelease.Asset{Name: "llama-b1234-bin-" + suffix, URL: "https://example.com/" + suffix})
			}
			b, err := json.Marshal(rel)
			if err != nil {
				t.Fatal(err)
			}
			old := http.DefaultTransport
			http.DefaultTransport = releaseRoundTrip{fn: func(r *http.Request) (*http.Response, error) {
				body := buf.Bytes()
				if r.URL.Host == "api.github.com" {
					body = b
				}
				return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(bytes.NewReader(body)), Header: make(http.Header)}, nil
			}}
			t.Cleanup(func() { http.DefaultTransport = old })
			got, err := DownloadRelease(t.Context(), cache, 1234)
			if !valid {
				if err == nil || got != "" {
					t.Fatalf("missing server: got %q, %v", got, err)
				}
				b, err := os.ReadFile(filepath.Join(cache, exe))
				if err != nil || string(b) != "old binary" {
					t.Fatalf("failed extraction changed cached executable: %q, %v", b, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got != filepath.Join(cache, exe) {
				t.Fatalf("unexpected executable path: %s", got)
			}
			if runtime.GOOS != "windows" {
				b, err := os.ReadFile(filepath.Join(cache, "libllama.so.0"))
				if err != nil || string(b) != "shared" {
					t.Fatalf("library symlink broken after installing: %q, %v", b, err)
				}
			}
		})
	}
}
