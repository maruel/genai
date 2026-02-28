// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package ghrelease provides shared helpers for downloading and extracting
// GitHub release assets.
package ghrelease

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/klauspost/compress/zstd"
)

// DownloadFile downloads a file from url and writes it to dst.
//
// Uses GITHUB_TOKEN from the environment for Bearer auth if set.
func DownloadFile(ctx context.Context, url, dst string) error {
	resp, err := doGet(ctx, url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	// Only then create the file.
	f, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o666)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, resp.Body)
	if err2 := f.Close(); err == nil {
		err = err2
	}
	return err
}

// doGet performs an HTTP GET with optional GITHUB_TOKEN auth and validates
// the response status.
func doGet(ctx context.Context, url string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, err
	}
	if token := os.Getenv("GITHUB_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		_ = resp.Body.Close()
		return nil, fmt.Errorf("unexpected http status code %d", resp.StatusCode)
	}
	return resp, nil
}

// apiBaseURL is the GitHub API base URL. Overridden in tests.
var apiBaseURL = "https://api.github.com"

// Asset is a file attached to a GitHub release.
type Asset struct {
	Name string `json:"name"`
	URL  string `json:"browser_download_url"`
}

// Release is the result of GetLatestRelease.
type Release struct {
	TagName string  `json:"tag_name"`
	Assets  []Asset `json:"assets"`
}

// GetLatestRelease returns the latest GitHub release for the given owner/repo,
// including the tag name and the list of downloadable assets.
//
// Uses GITHUB_TOKEN from the environment for Bearer auth if set.
func GetLatestRelease(ctx context.Context, owner, repo string) (*Release, error) {
	url := apiBaseURL + "/repos/" + owner + "/" + repo + "/releases/latest"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, http.NoBody)
	if err != nil {
		return nil, err
	}
	if token := os.Getenv("GITHUB_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	req.Header.Set("Accept", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	b, err := io.ReadAll(resp.Body)
	if err2 := resp.Body.Close(); err == nil {
		err = err2
	}
	if err != nil {
		return nil, fmt.Errorf("failed to get latest %s/%s release: %w", owner, repo, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get latest %s/%s release: http %d: %s", owner, repo, resp.StatusCode, string(b))
	}
	var release Release
	if err := json.Unmarshal(b, &release); err != nil {
		return nil, fmt.Errorf("failed to get latest %s/%s release: %w", owner, repo, err)
	}
	if release.TagName == "" {
		return nil, fmt.Errorf("failed to get latest %s/%s release: got %s", owner, repo, string(b))
	}
	return &release, nil
}

// ExtractArchive extracts files from archivePath into dstDir.
//
// Only files whose filepath.Base matches one of wantedFiles globs are
// extracted. Directory structure inside the archive is flattened. Files
// are written with mode 0o755.
//
// Supported formats: .zip, .tar.gz, .tgz, .tar.zst.
func ExtractArchive(archivePath, dstDir string, wantedFiles []string) error {
	switch {
	case strings.HasSuffix(archivePath, ".zip"):
		return extractZip(archivePath, dstDir, wantedFiles)
	case strings.HasSuffix(archivePath, ".tar.gz"), strings.HasSuffix(archivePath, ".tgz"):
		f, err := os.Open(archivePath)
		if err != nil {
			return err
		}
		defer func() { _ = f.Close() }()
		gr, err := gzip.NewReader(f)
		if err != nil {
			return err
		}
		defer func() { _ = gr.Close() }()
		return extractTar(gr, dstDir, wantedFiles)
	case strings.HasSuffix(archivePath, ".tar.zst"):
		f, err := os.Open(archivePath)
		if err != nil {
			return err
		}
		defer func() { _ = f.Close() }()
		zr, err := zstd.NewReader(f)
		if err != nil {
			return err
		}
		defer zr.Close()
		return extractTar(zr, dstDir, wantedFiles)
	default:
		return fmt.Errorf("unsupported archive format: %s", filepath.Base(archivePath))
	}
}

// DownloadAndExtract downloads a release archive from url and extracts
// matching files into dstDir.
//
// For tar-based formats (.tar.gz, .tgz, .tar.zst) the archive is streamed
// directly from the HTTP response without writing to disk. For .zip, a
// temporary file is used since the format requires random access.
//
// The archive format is determined from the url suffix.
func DownloadAndExtract(ctx context.Context, url, dstDir string, wantedFiles []string) error {
	resp, err := doGet(ctx, url)
	if err != nil {
		return err
	}
	defer func() { _ = resp.Body.Close() }()
	switch {
	case strings.HasSuffix(url, ".tar.gz"), strings.HasSuffix(url, ".tgz"):
		gr, err := gzip.NewReader(resp.Body)
		if err != nil {
			return err
		}
		defer func() { _ = gr.Close() }()
		return extractTar(gr, dstDir, wantedFiles)
	case strings.HasSuffix(url, ".tar.zst"):
		zr, err := zstd.NewReader(resp.Body)
		if err != nil {
			return err
		}
		defer zr.Close()
		return extractTar(zr, dstDir, wantedFiles)
	case strings.HasSuffix(url, ".zip"):
		// ZIP requires random access; spool to a temp file.
		f, err := os.CreateTemp("", "ghrelease-*.zip")
		if err != nil {
			return err
		}
		defer func() {
			_ = f.Close()
			_ = os.Remove(f.Name())
		}()
		if _, err := io.Copy(f, resp.Body); err != nil {
			return err
		}
		if err := f.Close(); err != nil {
			return err
		}
		return extractZip(f.Name(), dstDir, wantedFiles)
	default:
		return fmt.Errorf("unsupported archive format in url: %s", url)
	}
}

func extractZip(archivePath, dstDir string, wantedFiles []string) error {
	z, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer func() { _ = z.Close() }()
	for _, f := range z.File {
		// Ignore path; flatten directory structure.
		n := filepath.Base(f.Name)
		for _, desired := range wantedFiles {
			if ok, _ := filepath.Match(desired, n); !ok {
				continue
			}
			var src io.ReadCloser
			if src, err = f.Open(); err != nil {
				return err
			}
			var dst io.WriteCloser
			if dst, err = os.OpenFile(filepath.Join(dstDir, n), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o755); err == nil {
				_, err = io.CopyN(dst, src, int64(f.UncompressedSize64))
			}
			if err2 := src.Close(); err == nil {
				err = err2
			}
			if err2 := dst.Close(); err == nil {
				err = err2
			}
			if err != nil {
				return fmt.Errorf("failed to write %q: %w", n, err)
			}
		}
	}
	return nil
}

func extractTar(r io.Reader, dstDir string, wantedFiles []string) error {
	tarReader := tar.NewReader(r)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if header.Size == 0 || strings.HasSuffix(header.Name, "/") {
			continue
		}
		// Ignore path; flatten directory structure.
		n := filepath.Base(header.Name)
		if n == ".." || n == "." {
			continue
		}
		for _, desired := range wantedFiles {
			if ok, _ := filepath.Match(desired, n); !ok {
				continue
			}
			outFile, err := os.OpenFile(filepath.Join(dstDir, n), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o755) //nolint:gosec // G703: n is sanitized via filepath.Base above
			if err != nil {
				return err
			}
			_, err = io.Copy(outFile, tarReader)
			if err2 := outFile.Close(); err == nil {
				err = err2
			}
			if err != nil {
				return fmt.Errorf("failed to write %q: %w", n, err)
			}
		}
	}
	return nil
}
