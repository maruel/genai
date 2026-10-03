// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llamacppsrv downloads and starts llama-server from
// llama.cpp, directly from GitHub releases.
package llamacppsrv

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/ghrelease"
	"github.com/maruel/genai/providers/llamacpp"
)

// Version is the stable llama.cpp release last tested.
// Stable releases point to the nightly build containing their binaries.
const Version = "v0.5.0"

// Server is a llama-server instance.
type Server struct {
	url  string
	done <-chan error
	cmd  *exec.Cmd
}

// New creates a new instance of the llama-server and ensures the server is healthy.
//
// hostPort can be one of the forms "localhost", "localhost:8080", "localhost:0", ":8080", ":0" or "". "" is effectively
// "localhost:0", trying with port 8080 first then falling back to an ephemeral port.
//
// modelPath must be an absolute path to a local model file, or empty when using -hf/-hff in extraArgs
// to let llama-server download from HuggingFace directly.
//
// Doesn't pass "-ngl", "9999" by default so the user can override it.
//
// Output is redirected to logOutput if non-nil.
func New(ctx context.Context, exe, modelPath string, logOutput io.Writer, hostPort string, threads int, extraArgs []string) (*Server, error) {
	if !filepath.IsAbs(exe) {
		return nil, errors.New("exe must be an absolute path")
	}
	if modelPath != "" && !filepath.IsAbs(modelPath) {
		return nil, errors.New("modelPath must be an absolute path")
	}
	if hostPort == "" {
		hostPort = "localhost:0"
	}
	host, portStr, err := net.SplitHostPort(hostPort)
	if err != nil {
		return nil, err
	}
	port := 0
	if portStr != "" {
		if port, err = strconv.Atoi(portStr); err != nil {
			return nil, err
		}
	}
	if port == 0 {
		// First try the default port.
		var l net.Listener
		if l, err = net.Listen("tcp", host+":8080"); err != nil {
			if l, err = net.Listen("tcp", host+":0"); err != nil {
				return nil, err
			}
		}
		port = l.Addr().(*net.TCPAddr).Port
		if err := l.Close(); err != nil {
			return nil, err
		}
	}
	u := "http://" + host + ":" + strconv.Itoa(port)
	if threads == 0 {
		// Surprisingly llama-server seems to be hardcoded to 8 threads. Leave 2
		// cores (especially critical when HT) to allow us to get some CPU time.
		if threads = runtime.NumCPU() - 2; threads == 0 {
			threads = 1
		}
	}
	args := []string{exe, "--metrics", "--threads", strconv.Itoa(threads), "--port", strconv.Itoa(port)}
	if modelPath != "" {
		args = append([]string{exe, "--model", modelPath}, args[1:]...)
	}
	if host != "" {
		args = append(args, "--host", host)
	}
	args = append(args, extraArgs...)
	log.Printf("Args: %s", args)
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	// Make sure dynamic libraries will be found.
	cmd.Dir = filepath.Dir(exe)
	cmd.Env = serverEnv(cmd.Dir)
	if logOutput != nil {
		cmd.Stdout = logOutput
		cmd.Stderr = logOutput
	} else {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	cmd.Cancel = func() error {
		if runtime.GOOS == "windows" {
			return cmd.Process.Kill()
		}
		return cmd.Process.Signal(os.Interrupt)
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	done := make(chan error)
	go func() {
		err2 := cmd.Wait()
		if er, ok := errors.AsType[*exec.ExitError](err2); ok {
			s, ok := er.Sys().(syscall.WaitStatus)
			if ok && s.Signaled() {
				// It was simply killed.
				err2 = nil
			}
			// TODO: on Windows, figure out how to differentiate between normal quitting and an error.
		}
		done <- err2
		close(done)
	}()

	// Wait for the server to be ready.
	c, err := llamacpp.New(ctx, genai.ProviderOptionRemote(u))
	if err != nil {
		_ = cmd.Cancel()
		<-done
		return nil, fmt.Errorf("failed to create llamacpp client: %w", err)
	}
	// Loop until the server is healthy, process exits or the context is canceled.
	for ctx.Err() == nil {
		if status, _ := c.GetHealth(ctx); status == "ok" {
			break
		}
		select {
		case err := <-done:
			return nil, fmt.Errorf("starting llm server failed while querying for health: %w", err)
		case <-ctx.Done():
			_ = cmd.Cancel()
			<-done
			return nil, ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
	}

	return &Server{url: u, done: done, cmd: cmd}, nil
}

// Close stops the server and waits for it to exit.
func (s *Server) Close() error {
	_ = s.cmd.Cancel()
	err := <-s.done
	return err
}

// URL returns the URL to the server.
func (s *Server) URL() string {
	return s.url
}

// Done is a channel to listen to the server's termination. No need to call
// Close() if it is set.
func (s *Server) Done() <-chan error {
	return s.done
}

// DownloadVersion downloads a stable release (vX.Y.Z) or nightly build (bNNNN)
// into cache and returns the path to llama-server. Stable releases are resolved
// through their nightly-tag.txt asset; binaries report that nightly build number.
// Existing DownloadRelease callers can continue passing integer build numbers.
func DownloadVersion(ctx context.Context, cache, version string) (string, error) {
	if regexp.MustCompile(`^b[1-9]\d*$`).MatchString(version) {
		n, err := strconv.Atoi(version[1:])
		if err != nil {
			return "", fmt.Errorf("invalid build tag %q: %w", version, err)
		}
		return DownloadRelease(ctx, cache, n)
	}
	if !regexp.MustCompile(`^v\d+\.\d+\.\d+$`).MatchString(version) {
		return "", fmt.Errorf("invalid llama.cpp release tag %q", version)
	}
	rel, err := ghrelease.GetRelease(ctx, "ggml-org", "llama.cpp", version)
	if err != nil {
		return "", err
	}
	var asset *ghrelease.Asset
	for i := range rel.Assets {
		if rel.Assets[i].Name == "nightly-tag.txt" {
			asset = &rel.Assets[i]
			break
		}
	}
	if asset == nil {
		return "", fmt.Errorf("release %s has no nightly-tag.txt asset", version)
	}
	dir, err := os.MkdirTemp("", "llama-release-*")
	if err != nil {
		return "", err
	}
	defer func() { _ = os.RemoveAll(dir) }()
	p := filepath.Join(dir, "nightly-tag.txt")
	if err := ghrelease.DownloadFile(ctx, asset.URL, p); err != nil {
		return "", fmt.Errorf("resolving %s: %w", version, err)
	}
	b, err := os.ReadFile(p)
	if err != nil {
		return "", err
	}
	tag := strings.TrimSpace(string(b))
	if !regexp.MustCompile(`^b[1-9]\d*$`).MatchString(tag) {
		return "", fmt.Errorf("invalid nightly tag %q in release %s", tag, version)
	}
	return DownloadVersion(ctx, cache, tag)
}

// DownloadRelease downloads a specific release from GitHub into the specified
// directory and returns the file path to llama.cpp executable.
//
// Returns the file path to the executable.
func DownloadRelease(ctx context.Context, cache string, version int) (string, error) {
	if version <= 0 {
		return "", fmt.Errorf("invalid llama.cpp build number %d", version)
	}
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	llamaserver := filepath.Join(cache, "llama-server"+execSuffix)
	if cachedExecutable(llamaserver) {
		// Run it to confirm the version and that the file is not corrupted.
		// If this fails, starts from scratch.
		cmd := exec.CommandContext(ctx, llamaserver, "--version")
		cmd.Dir = filepath.Dir(llamaserver)
		cmd.Env = serverEnv(cmd.Dir)
		if out, err := cmd.CombinedOutput(); err == nil {
			if v, ok := parseBuildNumber(out); ok && v == version {
				return llamaserver, nil
			}
		}
	}

	build := "b" + strconv.Itoa(version)
	rel, err := ghrelease.GetRelease(ctx, "ggml-org", "llama.cpp", build)
	if err != nil {
		return "", fmt.Errorf("failed to get llama.cpp release %s: %w", build, err)
	}
	cuda := false
	if runtime.GOOS == "windows" && runtime.GOARCH == "amd64" {
		cuda = exec.CommandContext(ctx, "nvcc", "--version").Run() == nil
	}
	assets, err := releaseAssets(rel, runtime.GOOS, runtime.GOARCH, cuda)
	if err != nil {
		return "", err
	}
	if err := os.MkdirAll(cache, 0o755); err != nil {
		return "", err
	}
	// Extract into a clean directory so missing files and a partial download
	// cannot make an old executable appear to be a successful installation.
	dir, err := os.MkdirTemp(cache, ".llama-*")
	if err != nil {
		return "", err
	}
	defer func() { _ = os.RemoveAll(dir) }()
	for _, asset := range assets {
		if err := ghrelease.DownloadAndExtract(ctx, asset.URL, dir, []string{"*"}, false); err != nil {
			return "", fmt.Errorf("failed to download %s from github: %w", asset.Name, err)
		}
	}
	if !cachedExecutable(filepath.Join(dir, "llama-server"+execSuffix)) {
		return "", fmt.Errorf("release %s did not contain llama-server%s", build, execSuffix)
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", err
	}
	// Symlink targets created by ghrelease are absolute, so re-create them in
	// the final directory before removing the staging directory.
	for _, entry := range entries {
		src := filepath.Join(dir, entry.Name())
		dst := filepath.Join(cache, entry.Name())
		if err := os.Remove(dst); err != nil && !errors.Is(err, os.ErrNotExist) {
			return "", err
		}
		if entry.Type()&os.ModeSymlink != 0 {
			target, err := os.Readlink(src)
			if err != nil {
				return "", err
			}
			if err := os.Symlink(filepath.Base(target), dst); err != nil {
				return "", err
			}
		} else if err := os.Rename(src, dst); err != nil {
			return "", err
		}
	}
	return llamaserver, nil
}

func cachedExecutable(path string) bool {
	info, err := os.Stat(path)
	return err == nil && info.Mode().IsRegular()
}

func parseBuildNumber(out []byte) (int, bool) {
	re := regexp.MustCompile(`(?m)^\s*version:\s*(?:[^\n]+\(build\s+)?b?(\d+)(?:[ ,)\r\n]|$)`)
	m := re.FindSubmatch(out)
	if len(m) != 2 {
		return 0, false
	}
	v, err := strconv.Atoi(string(m[1]))
	return v, err == nil
}

func serverEnv(dir string) []string {
	env := os.Environ()
	switch runtime.GOOS {
	case "darwin":
		env = append(env, "DYLD_LIBRARY_PATH="+dir+":"+os.Getenv("DYLD_LIBRARY_PATH"))
	case "windows":
	default:
		env = append(env, "LD_LIBRARY_PATH="+dir+":"+os.Getenv("LD_LIBRARY_PATH"))
	}
	return env
}

// releaseAssets selects the binary archive and, for Windows CUDA, its matching
// runtime DLL archive. Runtime-only assets must never be selected as binaries.
func releaseAssets(rel *ghrelease.Release, goos, arch string, cuda bool) ([]ghrelease.Asset, error) {
	a := ""
	switch arch {
	case "amd64":
		a = "x64"
	case "arm64":
		a = "arm64"
	default:
		return nil, fmt.Errorf("unsupported llama.cpp platform %s/%s", goos, arch)
	}
	pattern := ""
	switch goos {
	case "darwin":
		pattern = "macos-" + a + ".*"
	case "linux":
		pattern = "ubuntu-" + a + ".*"
	case "windows":
		pattern = "win-cpu-" + a + ".zip"
		if cuda && arch == "amd64" {
			pattern = "win-cuda-*-" + a + ".zip"
		}
	default:
		return nil, fmt.Errorf("unsupported llama.cpp platform %s/%s", goos, arch)
	}
	prefix := "llama-" + rel.TagName + "-bin-"
	for i := range rel.Assets {
		asset := &rel.Assets[i]
		if !strings.HasPrefix(asset.Name, prefix) || (!strings.HasSuffix(asset.Name, ".tar.gz") && !strings.HasSuffix(asset.Name, ".zip")) {
			continue
		}
		if ok, _ := filepath.Match(pattern, strings.TrimPrefix(asset.Name, prefix)); !ok {
			continue
		}
		out := []ghrelease.Asset{*asset}
		if goos == "windows" && cuda && arch == "amd64" {
			name := "cudart-llama-bin-" + strings.TrimPrefix(asset.Name, prefix)
			for j := range rel.Assets {
				if rel.Assets[j].Name == name {
					return append(out, rel.Assets[j]), nil
				}
			}
			return nil, fmt.Errorf("release %s has no CUDA runtime asset %s", rel.TagName, name)
		}
		return out, nil
	}
	return nil, fmt.Errorf("no binary asset matching %q in release %s", pattern, rel.TagName)
}
