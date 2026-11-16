// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package ollamasrv downloads and starts ollama directly from GitHub releases.
package ollamasrv

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal/ghrelease"
	"github.com/maruel/genai/providers/ollama"
)

// Version is the version number that was last tried from
// https://github.com/ollama/ollama/releases
//
// You are free to use the version number that works best for you.
const Version = "v0.40.14"

// When updating the value above, regenerate the recordings with:
// RECORD=failure_only go test ./providers/ollama/...

var _ genai.Server = (*Server)(nil)

// Server is an "ollama serve" instance.
type Server struct {
	url  string
	cmd  *exec.Cmd
	done chan struct{}
	err  error // Written before done closes.
}

// New creates a new instance of the ollama serve server and ensures the server is healthy.
//
// hostPort can be one of the forms "localhost", "localhost:11434", "localhost:0", ":11434", ":0" or "". "" is effectively
// "localhost:0", trying with port 11434 first then falling back to an ephemeral port.
//
// Use env to specify the OLLAMA_xxx environment variables. They are documented with "ollama help serve".
//
// Output is redirected to logOutput if non-nil.
func New(ctx context.Context, exe string, logOutput io.Writer, hostPort string, env []string) (*Server, error) {
	if !filepath.IsAbs(exe) {
		return nil, errors.New("exe must be an absolute path")
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
		lc := net.ListenConfig{}
		if l, err = lc.Listen(ctx, "tcp", host+":11434"); err != nil {
			if l, err = lc.Listen(ctx, "tcp", host+":0"); err != nil {
				return nil, err
			}
		}
		port = l.Addr().(*net.TCPAddr).Port //nolint:forcetypeassert // A TCP listener returns a TCP address.
		if err := l.Close(); err != nil {
			return nil, err
		}
	}
	u := "http://" + host + ":" + strconv.Itoa(port)
	cmd := exec.CommandContext(ctx, exe, "serve")
	if logOutput != nil {
		cmd.Stdout = logOutput
		cmd.Stderr = logOutput
	} else {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	// Archive layout is platform-specific:
	//   Linux:   cache/{bin,lib}/ollama  (rpath $ORIGIN/../lib/ollama/)
	//   macOS:   cache/ollama, libs at cache/ (flat, rpath @executable_path)
	//   Windows: cache/{bin,lib}/ollama
	root := filepath.Dir(exe)
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		root = filepath.Dir(filepath.Dir(exe))
	}
	cmd.Dir = root
	cmd.Env = append(os.Environ(), "GIN_MODE=release", "OLLAMA_HOST="+u)
	switch runtime.GOOS {
	case "linux":
		cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+filepath.Join(root, "lib", "ollama")+":"+os.Getenv("LD_LIBRARY_PATH"))
	case "darwin":
		cmd.Env = append(cmd.Env, "DYLD_LIBRARY_PATH="+root+":"+os.Getenv("DYLD_LIBRARY_PATH"))
	}
	cmd.Env = append(cmd.Env, env...)
	// Close and context cancellation request the stop. "ollama serve" closes
	// its listener, unloads its runners and returns 0 on SIGINT or SIGTERM;
	// before its handler is installed, SIGINT terminates it.
	var requested atomic.Bool
	cmd.Cancel = func() error {
		requested.Store(true)
		if runtime.GOOS == "windows" {
			return cmd.Process.Kill()
		}
		return cmd.Process.Signal(os.Interrupt)
	}
	if err = cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start ollama: %w", err)
	}
	srv := &Server{url: u, cmd: cmd, done: make(chan struct{})}
	go func() {
		srv.err = base.ServerExit(cmd.Wait(), cmd.ProcessState, requested.Load(), os.Interrupt)
		close(srv.done)
	}()

	// Wait for the server to be ready.
	c, err := ollama.New(ctx, genai.ProviderOptionRemote(u))
	if err != nil {
		return nil, errors.Join(fmt.Errorf("failed to create ollama client: %w", err), srv.Close())
	}
	// Loop until the server is healthy, process exits or the context is canceled.
	for ctx.Err() == nil {
		if _, err := c.ListModels(ctx); err == nil {
			break
		}
		select {
		case <-srv.done:
			return nil, fmt.Errorf("starting ollama server failed while querying for health: %w", srv.err)
		case <-ctx.Done():
			return nil, errors.Join(ctx.Err(), srv.Close())
		case <-time.After(20 * time.Millisecond):
		}
	}

	return srv, nil
}

// Close implements genai.Server.
func (s *Server) Close() error {
	err := s.cmd.Cancel()
	if errors.Is(err, os.ErrProcessDone) {
		err = nil
	}
	<-s.done
	return errors.Join(err, s.err)
}

// URL implements genai.Server.
func (s *Server) URL() string {
	return s.url
}

// Done implements genai.Server.
func (s *Server) Done() <-chan struct{} {
	return s.done
}

// Err implements genai.Server.
func (s *Server) Err() error {
	select {
	case <-s.done:
		return s.err
	default:
		return nil
	}
}

// DownloadRelease downloads a specific release from GitHub into the specified
// directory and returns the file path to ollama executable.
//
// When unspecified, the latest release is downloaded.
//
// The archive directory structure is preserved under cache: the binary ends
// up at cache/bin/ollama, libraries at cache/lib/ollama/.
func DownloadRelease(ctx context.Context, cache, version string) (string, error) {
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	if version == "" {
		r, err := ghrelease.GetLatestRelease(ctx, "ollama", "ollama")
		if err != nil {
			return "", fmt.Errorf("failed figuring out latest ollama release version: %w", err)
		}
		version = r.TagName
	}
	if !strings.HasPrefix(version, "v") {
		return "", fmt.Errorf("version must be in the form v0.1.2, got %q", version)
	}
	// Binary path depends on archive layout:
	//   Linux:   bin/ollama (archive has bin/ + lib/ollama/ hierarchy)
	//   macOS:   ollama    (archive is flat, subdirs like mlx_metal_v3/)
	//   Windows: bin/ollama.exe
	ollamaexe := filepath.Join(cache, "ollama"+execSuffix)
	if runtime.GOOS == "linux" || runtime.GOOS == "windows" {
		ollamaexe = filepath.Join(cache, "bin", "ollama"+execSuffix)
	}
	if _, err := os.Stat(ollamaexe); err == nil {
		cmd := exec.CommandContext(ctx, ollamaexe, "--version")
		cmd.Env = append(os.Environ(), "OLLAMA_HOST=http://127.0.0.1:0")
		if out, err2 := cmd.CombinedOutput(); err2 == nil {
			re := regexp.MustCompile(`client version is (\d+\.\d+.\d+)`)
			if m := re.FindStringSubmatch(string(out)); len(m) == 2 {
				if "v"+m[1] == version {
					return ollamaexe, nil
				}
			}
		}
		// Wrong version or corrupted; clean and re-extract.
		if err := os.RemoveAll(cache); err != nil {
			return "", fmt.Errorf("failed to clean cache: %w", err)
		}
	} else if entries, _ := os.ReadDir(cache); len(entries) > 0 {
		// Cache has files but not at bin/ollama (e.g. old flat layout).
		if err := os.RemoveAll(cache); err != nil {
			return "", fmt.Errorf("failed to clean stale cache: %w", err)
		}
	}

	// Extract the entire archive with directory structure preserved.
	var archiveName string
	switch runtime.GOOS {
	case "darwin":
		archiveName = "ollama-darwin.tgz"
	case "linux":
		// TODO: rocm
		// The files are over 1.5GiB :(
		switch runtime.GOARCH {
		case "amd64":
			archiveName = "ollama-linux-amd64.tar.zst"
		case "arm64":
			archiveName = "ollama-linux-arm64.tar.zst"
		default:
			return "", fmt.Errorf("unsupported architecture %q", runtime.GOARCH)
		}
	case "windows":
		// TODO: rocm
		switch runtime.GOARCH {
		case "amd64":
			// The file is over 1.5GiB :(
			archiveName = "ollama-windows-amd64.zip"
		case "arm64":
			archiveName = "ollama-windows-arm64.zip"
		default:
			return "", fmt.Errorf("unsupported architecture %q", runtime.GOARCH)
		}
	default:
		return "", fmt.Errorf("unsupported OS %q", runtime.GOOS)
	}
	dlURL := "https://github.com/ollama/ollama/releases/download/" + url.PathEscape(version) + "/" + archiveName
	if err := ghrelease.DownloadAndExtract(ctx, dlURL, cache, nil, true); err != nil {
		return "", fmt.Errorf("failed to download and extract %s from github: %w", archiveName, err)
	}
	return ollamaexe, nil
}
