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
	"errors"
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

// GetRelease returns a specific GitHub release by tag for the given owner/repo,
// including the tag name and the list of downloadable assets.
//
// Uses GITHUB_TOKEN from the environment for Bearer auth if set.
func GetRelease(ctx context.Context, owner, repo, tag string) (*Release, error) {
	u := apiBaseURL + "/repos/" + owner + "/" + repo + "/releases/tags/" + tag
	return getRelease(ctx, u, owner, repo, tag)
}

// GetLatestRelease returns the latest GitHub release for the given owner/repo,
// including the tag name and the list of downloadable assets.
//
// Uses GITHUB_TOKEN from the environment for Bearer auth if set.
func GetLatestRelease(ctx context.Context, owner, repo string) (*Release, error) {
	u := apiBaseURL + "/repos/" + owner + "/" + repo + "/releases/latest"
	return getRelease(ctx, u, owner, repo, "latest")
}

func getRelease(ctx context.Context, u, owner, repo, label string) (*Release, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, http.NoBody)
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
		return nil, fmt.Errorf("failed to get %s %s/%s release: %w", label, owner, repo, err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("failed to get %s %s/%s release: http %d: %s", label, owner, repo, resp.StatusCode, string(b))
	}
	var release Release
	if err := json.Unmarshal(b, &release); err != nil {
		return nil, fmt.Errorf("failed to get %s %s/%s release: %w", label, owner, repo, err)
	}
	if release.TagName == "" {
		return nil, fmt.Errorf("failed to get %s %s/%s release: got %s", label, owner, repo, string(b))
	}
	return &release, nil
}

// ExtractArchive extracts files from archivePath into dstDir.
//
// Only files matching one of wantedFiles globs are extracted. When
// preserveDir is false, the directory structure is flattened: only
// filepath.Base of each entry is matched and used as the output name.
// When preserveDir is true, the full archive path is matched against
// patterns and the directory structure is preserved under dstDir.
//
// Patterns are filepath.Match globs. Additionally, a pattern ending
// with "/..." matches any entry whose archive path starts with that
// prefix (recursive directory match).
//
// Files are written with mode 0o755.
//
// Supported formats: .zip, .tar.gz, .tgz, .tar.zst.
func ExtractArchive(archivePath, dstDir string, wantedFiles []string, preserveDir bool) error {
	switch {
	case strings.HasSuffix(archivePath, ".zip"):
		return extractZip(archivePath, dstDir, wantedFiles, preserveDir)
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
		return extractTar(gr, dstDir, wantedFiles, preserveDir)
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
		return extractTar(zr, dstDir, wantedFiles, preserveDir)
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
// The archive format is determined from the url suffix. See ExtractArchive
// for details on wantedFiles and preserveDir.
func DownloadAndExtract(ctx context.Context, url, dstDir string, wantedFiles []string, preserveDir bool) error {
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
		return extractTar(gr, dstDir, wantedFiles, preserveDir)
	case strings.HasSuffix(url, ".tar.zst"):
		zr, err := zstd.NewReader(resp.Body)
		if err != nil {
			return err
		}
		defer zr.Close()
		return extractTar(zr, dstDir, wantedFiles, preserveDir)
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
		return extractZip(f.Name(), dstDir, wantedFiles, preserveDir)
	default:
		return fmt.Errorf("unsupported archive format in url: %s", url)
	}
}

func extractZip(archivePath, dstDir string, wantedFiles []string, preserveDir bool) error {
	z, err := zip.OpenReader(archivePath)
	if err != nil {
		return err
	}
	defer func() { _ = z.Close() }()
	prefix := filepath.Clean(dstDir) + string(os.PathSeparator)
	for _, f := range z.File {
		cleaned := filepath.Clean(f.Name)
		if cleaned == "." {
			continue
		}
		var name string
		var matchName string
		if preserveDir {
			name = cleaned
			matchName = filepath.ToSlash(cleaned)
		} else {
			name = filepath.Base(cleaned)
			matchName = name
			if name == "." || name == ".." {
				continue
			}
		}
		if !matchAny(matchName, wantedFiles) {
			continue
		}
		dst := filepath.Join(dstDir, name)
		if !strings.HasPrefix(dst, prefix) {
			return fmt.Errorf("archive entry %q escapes destination directory", f.Name)
		}
		// ZIP directory entries have a trailing slash.
		if strings.HasSuffix(f.Name, "/") {
			if preserveDir {
				if err := os.MkdirAll(dst, 0o755); err != nil {
					return fmt.Errorf("failed to create directory %q: %w", name, err)
				}
			}
			continue
		}
		if preserveDir {
			if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
				return fmt.Errorf("failed to create parent directory for %q: %w", name, err)
			}
		}
		src, err := f.Open()
		if err != nil {
			return err
		}
		dstFile, err := os.OpenFile(dst, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o755)
		if err != nil {
			_ = src.Close()
			return err
		}
		_, err = io.CopyN(dstFile, src, int64(f.UncompressedSize64))
		if err2 := src.Close(); err == nil {
			err = err2
		}
		if err2 := dstFile.Close(); err == nil {
			err = err2
		}
		if err != nil {
			return fmt.Errorf("failed to write %q: %w", name, err)
		}
	}
	return nil
}

func extractTar(r io.Reader, dstDir string, wantedFiles []string, preserveDir bool) error {
	tarReader := tar.NewReader(r)
	prefix := filepath.Clean(dstDir) + string(os.PathSeparator)
	for {
		header, err := tarReader.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		cleaned := filepath.Clean(header.Name)
		var name string
		var matchName string
		if preserveDir {
			name = cleaned
			matchName = filepath.ToSlash(cleaned)
		} else {
			name = filepath.Base(cleaned)
			matchName = name
			if name == "." || name == ".." {
				continue
			}
		}
		if !matchAny(matchName, wantedFiles) {
			continue
		}
		dst := filepath.Join(dstDir, name)
		if !strings.HasPrefix(dst, prefix) {
			return fmt.Errorf("archive entry %q escapes destination directory", header.Name)
		}
		switch header.Typeflag {
		case tar.TypeDir:
			if preserveDir {
				if err := os.MkdirAll(dst, 0o755); err != nil {
					return fmt.Errorf("failed to create directory %q: %w", name, err)
				}
			}
		case tar.TypeSymlink:
			if preserveDir {
				if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
					return fmt.Errorf("failed to create parent directory for symlink %q: %w", name, err)
				}
			}
			// Validate the symlink target does not escape dstDir.
			if filepath.IsAbs(header.Linkname) {
				if preserveDir {
					return fmt.Errorf("symlink %q has absolute target %q", name, header.Linkname)
				}
				continue
			}
			resolved := filepath.Join(filepath.Dir(dst), header.Linkname) //nolint:gosec // Validated with prefix check below.
			resolved = filepath.Clean(resolved)
			if !strings.HasPrefix(resolved, prefix) {
				if preserveDir {
					return fmt.Errorf("symlink %q target %q escapes destination directory", name, header.Linkname)
				}
				continue
			}
			if err := os.Remove(dst); err != nil && !errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("failed to remove existing file %q: %w", name, err)
			}
			if err := os.Symlink(resolved, dst); err != nil {
				return fmt.Errorf("failed to create symlink %q: %w", name, err)
			}
		case tar.TypeReg:
			if header.Size == 0 {
				continue
			}
			if preserveDir {
				if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
					return fmt.Errorf("failed to create parent directory for %q: %w", name, err)
				}
			}
			outFile, err := os.OpenFile(dst, os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o755)
			if err != nil {
				return err
			}
			_, err = io.Copy(outFile, tarReader)
			if err2 := outFile.Close(); err == nil {
				err = err2
			}
			if err != nil {
				return fmt.Errorf("failed to write %q: %w", name, err)
			}
		}
	}
	return nil
}

// matchAny reports whether name matches any of the given patterns.
//
// A nil or empty patterns slice matches all names (extract everything).
// Patterns are filepath.Match globs. Additionally, a pattern ending with
// "/..." matches any name that has that prefix (recursive directory match).
//
// Both name and patterns use forward slashes as path separators.
// Callers must normalize with filepath.ToSlash before passing the name.
func matchAny(name string, patterns []string) bool {
	if len(patterns) == 0 {
		return true
	}
	for _, p := range patterns {
		if strings.HasSuffix(p, "/...") {
			prefix := strings.TrimSuffix(p, "/...")
			if strings.HasPrefix(name, prefix) {
				return true
			}
			continue
		}
		if ok, _ := filepath.Match(p, name); ok {
			return true
		}
	}
	return false
}
