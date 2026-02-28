// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package ghrelease

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/klauspost/compress/zstd"
)

func TestExtractArchive(t *testing.T) {
	t.Run("Zip", func(t *testing.T) {
		dir := t.TempDir()
		archivePath := filepath.Join(dir, "test.zip")
		createArchive(t, archivePath, map[string]string{
			"subdir/hello.txt": "hello world",
			"subdir/foo.bin":   "foo content",
		})
		dstDir := filepath.Join(dir, "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := ExtractArchive(archivePath, dstDir, []string{"hello.txt", "foo.bin"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "hello.txt"), "hello world")
		assertFileContent(t, filepath.Join(dstDir, "foo.bin"), "foo content")
	})
	t.Run("TarGz", func(t *testing.T) {
		dir := t.TempDir()
		archivePath := filepath.Join(dir, "test.tar.gz")
		createArchive(t, archivePath, map[string]string{
			"subdir/hello.txt": "hello world",
			"subdir/foo.bin":   "foo content",
		})
		dstDir := filepath.Join(dir, "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := ExtractArchive(archivePath, dstDir, []string{"hello.txt", "foo.bin"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "hello.txt"), "hello world")
		assertFileContent(t, filepath.Join(dstDir, "foo.bin"), "foo content")
	})
	t.Run("Tgz", func(t *testing.T) {
		dir := t.TempDir()
		archivePath := filepath.Join(dir, "test.tgz")
		createArchive(t, archivePath, map[string]string{
			"a.txt": "data",
		})
		dstDir := filepath.Join(dir, "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := ExtractArchive(archivePath, dstDir, []string{"a.txt"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "a.txt"), "data")
	})
	t.Run("TarZst", func(t *testing.T) {
		dir := t.TempDir()
		archivePath := filepath.Join(dir, "test.tar.zst")
		createArchive(t, archivePath, map[string]string{
			"subdir/hello.txt": "hello world",
			"subdir/foo.bin":   "foo content",
		})
		dstDir := filepath.Join(dir, "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := ExtractArchive(archivePath, dstDir, []string{"hello.txt", "foo.bin"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "hello.txt"), "hello world")
		assertFileContent(t, filepath.Join(dstDir, "foo.bin"), "foo content")
	})
	t.Run("UnwantedSkipped", func(t *testing.T) {
		dir := t.TempDir()
		archivePath := filepath.Join(dir, "test.zip")
		createArchive(t, archivePath, map[string]string{
			"keep.txt":   "kept",
			"ignore.txt": "ignored",
		})
		dstDir := filepath.Join(dir, "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := ExtractArchive(archivePath, dstDir, []string{"keep.txt"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "keep.txt"), "kept")
		if _, err := os.Stat(filepath.Join(dstDir, "ignore.txt")); err == nil {
			t.Fatal("ignore.txt should not have been extracted")
		}
	})
	t.Run("GlobMatch", func(t *testing.T) {
		dir := t.TempDir()
		archivePath := filepath.Join(dir, "test.zip")
		createArchive(t, archivePath, map[string]string{
			"lib/foo.dll": "dll1",
			"lib/bar.dll": "dll2",
			"lib/baz.so":  "so1",
		})
		dstDir := filepath.Join(dir, "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := ExtractArchive(archivePath, dstDir, []string{"*.dll"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "foo.dll"), "dll1")
		assertFileContent(t, filepath.Join(dstDir, "bar.dll"), "dll2")
		if _, err := os.Stat(filepath.Join(dstDir, "baz.so")); err == nil {
			t.Fatal("baz.so should not have been extracted")
		}
	})
	t.Run("UnsupportedFormat", func(t *testing.T) {
		if err := ExtractArchive("file.rar", ".", nil); err == nil {
			t.Fatal("expected error for unsupported format")
		}
	})
}

func TestGetLatestRelease(t *testing.T) {
	t.Run("Valid", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path != "/repos/testowner/testrepo/releases/latest" {
				http.NotFound(w, r)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"tag_name":"v1.2.3","assets":[{"name":"foo.tar.gz","browser_download_url":"https://example.com/foo.tar.gz"},{"name":"bar.zip","browser_download_url":"https://example.com/bar.zip"}]}`))
		}))
		defer srv.Close()
		old := apiBaseURL
		apiBaseURL = srv.URL
		t.Cleanup(func() { apiBaseURL = old })

		rel, err := GetLatestRelease(context.Background(), "testowner", "testrepo")
		if err != nil {
			t.Fatal(err)
		}
		if rel.TagName != "v1.2.3" {
			t.Fatalf("tag: got %q, want %q", rel.TagName, "v1.2.3")
		}
		if len(rel.Assets) != 2 {
			t.Fatalf("assets: got %d, want 2", len(rel.Assets))
		}
		if rel.Assets[0].Name != "foo.tar.gz" {
			t.Fatalf("asset[0].Name: got %q, want %q", rel.Assets[0].Name, "foo.tar.gz")
		}
		if rel.Assets[0].URL != "https://example.com/foo.tar.gz" {
			t.Fatalf("asset[0].URL: got %q, want %q", rel.Assets[0].URL, "https://example.com/foo.tar.gz")
		}
	})
	t.Run("NotFound", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.NotFound(w, r)
		}))
		defer srv.Close()
		old := apiBaseURL
		apiBaseURL = srv.URL
		t.Cleanup(func() { apiBaseURL = old })

		if _, err := GetLatestRelease(context.Background(), "no", "repo"); err == nil {
			t.Fatal("expected error for 404")
		}
	})
}

func TestDownloadFile(t *testing.T) {
	t.Run("Valid", func(t *testing.T) {
		want := "file content here"
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(want))
		}))
		defer srv.Close()

		dst := filepath.Join(t.TempDir(), "downloaded.bin")
		if err := DownloadFile(context.Background(), srv.URL+"/file.bin", dst); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, dst, want)
	})
	t.Run("HTTPError", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "not found", http.StatusNotFound)
		}))
		defer srv.Close()

		dst := filepath.Join(t.TempDir(), "downloaded.bin")
		if err := DownloadFile(context.Background(), srv.URL+"/missing", dst); err == nil {
			t.Fatal("expected error for 404 response")
		}
	})
}

func TestDownloadAndExtract(t *testing.T) {
	t.Run("TarGzStream", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeTarGz(t, w, map[string]string{"dir/hello.txt": "streamed"})
		}))
		defer srv.Close()

		dstDir := filepath.Join(t.TempDir(), "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := DownloadAndExtract(context.Background(), srv.URL+"/archive.tar.gz", dstDir, []string{"hello.txt"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "hello.txt"), "streamed")
	})
	t.Run("TarZstStream", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeTarZst(t, w, map[string]string{"dir/hello.txt": "zst-streamed"})
		}))
		defer srv.Close()

		dstDir := filepath.Join(t.TempDir(), "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := DownloadAndExtract(context.Background(), srv.URL+"/archive.tar.zst", dstDir, []string{"hello.txt"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "hello.txt"), "zst-streamed")
	})
	t.Run("ZipFallback", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			writeZip(t, w, map[string]string{"dir/hello.txt": "zip-streamed"})
		}))
		defer srv.Close()

		dstDir := filepath.Join(t.TempDir(), "out")
		if err := os.Mkdir(dstDir, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := DownloadAndExtract(context.Background(), srv.URL+"/archive.zip", dstDir, []string{"hello.txt"}); err != nil {
			t.Fatal(err)
		}
		assertFileContent(t, filepath.Join(dstDir, "hello.txt"), "zip-streamed")
	})
	t.Run("HTTPError", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "gone", http.StatusGone)
		}))
		defer srv.Close()

		if err := DownloadAndExtract(context.Background(), srv.URL+"/archive.tar.gz", t.TempDir(), nil); err == nil {
			t.Fatal("expected error for non-200 response")
		}
	})
}

//

// createArchive creates an archive file at path. The format is determined by
// the file extension (.zip, .tar.gz, .tgz, .tar.zst).
func createArchive(t *testing.T, path string, files map[string]string) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	switch {
	case hasSuffix(path, ".zip"):
		writeZip(t, f, files)
	case hasSuffix(path, ".tar.gz", ".tgz"):
		writeTarGz(t, f, files)
	case hasSuffix(path, ".tar.zst"):
		writeTarZst(t, f, files)
	default:
		t.Fatalf("unsupported test archive format: %s", path)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func hasSuffix(s string, suffixes ...string) bool {
	for _, sfx := range suffixes {
		if filepath.Ext(s) == sfx || len(s) > len(sfx) && s[len(s)-len(sfx):] == sfx {
			return true
		}
	}
	return false
}

func writeZip(t *testing.T, w io.Writer, files map[string]string) {
	t.Helper()
	zw := zip.NewWriter(w)
	for name, content := range files {
		fw, err := zw.Create(name)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := fw.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeTarGz(t *testing.T, w io.Writer, files map[string]string) {
	t.Helper()
	gw := gzip.NewWriter(w)
	writeTar(t, gw, files)
	if err := gw.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeTarZst(t *testing.T, w io.Writer, files map[string]string) {
	t.Helper()
	zw, err := zstd.NewWriter(w)
	if err != nil {
		t.Fatal(err)
	}
	writeTar(t, zw, files)
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeTar(t *testing.T, w io.Writer, files map[string]string) {
	t.Helper()
	tw := tar.NewWriter(w)
	for name, content := range files {
		hdr := &tar.Header{
			Name: name,
			Mode: 0o644,
			Size: int64(len(content)),
		}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatal(err)
		}
		if _, err := tw.Write([]byte(content)); err != nil {
			t.Fatal(err)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
}

func assertFileContent(t *testing.T, path, want string) {
	t.Helper()
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	if string(got) != want {
		t.Fatalf("content mismatch for %s: got %q, want %q", path, got, want)
	}
}
