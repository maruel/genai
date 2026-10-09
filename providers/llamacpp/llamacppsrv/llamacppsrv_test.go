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
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"

	"github.com/maruel/genai/internal/ghrelease"
)

func TestDownloadRelease(t *testing.T) {
	t.Run("extraction", testDownloadReleaseExtraction)
}

func TestReadReleaseAlias(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, text := range []bool{false, true} {
			t.Run(strconv.FormatBool(text), func(t *testing.T) {
				path := filepath.Join(t.TempDir(), "v0.5.0")
				if text {
					if err := os.WriteFile(path, []byte("b1234\n"), 0o644); err != nil {
						t.Fatal(err)
					}
				} else if err := os.Symlink("b1234", path); err != nil {
					if symlinkUnsupported(err) {
						t.Skip("symlink support unavailable")
					}
					t.Fatal(err)
				}
				if tag, err := readReleaseAlias(path); err != nil || tag != "b1234" {
					t.Fatalf("alias = %q, %v; want b1234", tag, err)
				}
			})
		}
	})
	t.Run("error", func(t *testing.T) {
		for _, tag := range []string{"", "../b1234\n", "v0.5.0\n", "b0\n", "b1234", " b1234\n", "\tb1234\n", "b1234\r\n", "b1234\nb5678\n"} {
			t.Run(tag, func(t *testing.T) {
				path := filepath.Join(t.TempDir(), "v0.5.0")
				if err := os.WriteFile(path, []byte(tag), 0o644); err != nil {
					t.Fatal(err)
				}
				if _, err := readReleaseAlias(path); err == nil {
					t.Fatal("accepted invalid text alias")
				}
			})
		}
		if _, err := readReleaseAlias(t.TempDir()); err == nil {
			t.Fatal("accepted directory as alias")
		}
		if _, err := readReleaseAlias(filepath.Join(t.TempDir(), "missing")); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("missing alias error = %v", err)
		}
	})
}

func TestWriteReleaseAlias(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "v0.5.0")
		for range 2 {
			if err := writeReleaseAlias(path, "b1234"); err != nil {
				t.Fatal(err)
			}
		}
		b, err := os.ReadFile(path)
		if err != nil || string(b) != "b1234\n" {
			t.Fatalf("text alias = %q, %v", b, err)
		}
		entries, err := os.ReadDir(filepath.Dir(path))
		if err != nil || len(entries) != 1 {
			t.Fatalf("unexpected temporary files: %v, %v", entries, err)
		}
	})
	t.Run("error", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "v0.5.0")
		if err := os.WriteFile(path, []byte("b5678\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := writeReleaseAlias(path, "b1234"); err == nil {
			t.Fatal("overwrote conflicting alias")
		}
		if tag, err := readReleaseAlias(path); err != nil || tag != "b5678" {
			t.Fatalf("conflicting alias changed: %q, %v", tag, err)
		}
		if err := writeReleaseAlias(filepath.Join(t.TempDir(), "missing", "v0.5.0"), "b1234"); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("missing parent error = %v", err)
		}
	})
}

func TestSymlinkUnsupported(t *testing.T) {
	for _, tc := range []struct {
		name string
		err  error
		want bool
	}{
		{"windowsPrivilege", syscall.Errno(1314), runtime.GOOS == "windows"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			err := &os.LinkError{Op: "symlink", Old: "b1234", New: "v0.5.0", Err: tc.err}
			if got := symlinkUnsupported(err); got != tc.want {
				t.Fatalf("symlinkUnsupported(%v) = %v, want %v", err, got, tc.want)
			}
		})
	}
}

func TestParseBuildNumber(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		for _, tc := range []struct {
			name string
			out  string
			want int
		}{
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
	t.Run("cachedBuildRepair", func(t *testing.T) {
		cache := t.TempDir()
		if err := os.Mkdir(filepath.Join(cache, "b1234"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := writeReleaseAlias(filepath.Join(cache, "v0.5.0"), "b1234"); err != nil {
			t.Fatal(err)
		}
		old := http.DefaultTransport
		n := 0
		http.DefaultTransport = releaseRoundTrip{fn: func(r *http.Request) (*http.Response, error) {
			n++
			if r.URL.Path != "/repos/ggml-org/llama.cpp/releases/tags/b1234" {
				t.Fatalf("unexpected request: %s", r.URL)
			}
			return nil, errUnexpectedHTTPRequest
		}}
		t.Cleanup(func() { http.DefaultTransport = old })
		if _, err := DownloadVersion(t.Context(), cache, "v0.5.0"); !errors.Is(err, errUnexpectedHTTPRequest) {
			t.Fatalf("repair error = %v", err)
		}
		if n != 1 {
			t.Fatalf("repair requests = %d, want 1", n)
		}
	})
	t.Run("cachedResolutionError", func(t *testing.T) {
		for _, target := range []string{"../b1234", "v0.5.0", "b0"} {
			t.Run(target, func(t *testing.T) {
				cache := t.TempDir()
				if err := os.Symlink(target, filepath.Join(cache, "v0.5.0")); err != nil {
					if symlinkUnsupported(err) {
						t.Skip("symlink support unavailable")
					}
					t.Fatal(err)
				}
				old := http.DefaultTransport
				http.DefaultTransport = forbidRoundTrip{t: t}
				t.Cleanup(func() { http.DefaultTransport = old })
				if _, err := DownloadVersion(t.Context(), cache, "v0.5.0"); err == nil {
					t.Fatal("accepted invalid cached resolution")
				}
			})
		}
	})
	t.Run("installationError", func(t *testing.T) {
		cache := t.TempDir()
		old := http.DefaultTransport
		http.DefaultTransport = releaseRoundTrip{fn: func(r *http.Request) (*http.Response, error) {
			body := `{"tag_name":"v0.5.0","assets":[{"name":"nightly-tag.txt","browser_download_url":"https://example.com/nightly-tag.txt"}]}`
			if r.URL.Host == "example.com" {
				body = "b11146\n"
			} else if strings.HasSuffix(r.URL.Path, "/b11146") {
				return nil, errUnexpectedHTTPRequest
			}
			return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body)), Header: make(http.Header)}, nil
		}}
		t.Cleanup(func() { http.DefaultTransport = old })
		if _, err := DownloadVersion(t.Context(), cache, "v0.5.0"); err == nil {
			t.Fatal("accepted failed installation")
		}
		if _, err := os.Lstat(filepath.Join(cache, "v0.5.0")); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("published resolution after failure: %v", err)
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
			// A cached llama-server would be run to check its version, so the
			// sentinel is a library.
			if !valid {
				if err := os.WriteFile(filepath.Join(cache, "libllama.so"), []byte("old library"), 0o644); err != nil {
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
				b, err := os.ReadFile(filepath.Join(cache, "libllama.so"))
				if err != nil || string(b) != "old library" {
					t.Fatalf("failed extraction changed cached library: %q, %v", b, err)
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
