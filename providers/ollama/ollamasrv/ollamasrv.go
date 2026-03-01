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
	"syscall"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/ghrelease"
	"github.com/maruel/genai/providers/ollama"
)

// Version is the version number that was last tried from
// https://github.com/ollama/ollama/releases
//
// You are free to use the version number that works best for you.
const Version = "v0.17.4"

// When updating the value above, regenerate the recordings with:
// RECORD=all go test ./providers/ollama/...

// Server is an "ollama serve" instance.
type Server struct {
	url  string
	done <-chan error
	cmd  *exec.Cmd
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
		if l, err = net.Listen("tcp", host+":11434"); err != nil {
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
	cmd := exec.CommandContext(ctx, exe, "serve")
	if logOutput != nil {
		cmd.Stdout = logOutput
		cmd.Stderr = logOutput
	} else {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	// Make sure dynamic libraries will be found.
	cmd.Dir = filepath.Dir(exe)
	cmd.Env = append(os.Environ(), "GIN_MODE=release", "OLLAMA_HOST="+u)
	if runtime.GOOS != "windows" {
		cmd.Env = append(cmd.Env, "LD_LIBRARY_PATH="+cmd.Dir+":"+os.Getenv("LD_LIBRARY_PATH"))
	}
	cmd.Env = append(cmd.Env, env...)
	cmd.Cancel = func() error {
		if runtime.GOOS == "windows" {
			return cmd.Process.Kill()
		}
		return cmd.Process.Signal(os.Interrupt)
	}
	if err = cmd.Start(); err != nil {
		return nil, fmt.Errorf("failed to start ollama: %w", err)
	}
	done := make(chan error)
	go func() {
		err2 := cmd.Wait()
		var er *exec.ExitError
		if errors.As(err2, &er) {
			s, ok := er.Sys().(syscall.WaitStatus)
			if ok && s.Signaled() {
				// It was simply killed.
				err2 = nil
			}
			if runtime.GOOS == "windows" {
				// We need to figure out how to differentiate between normal quitting
				// and an error.
				err2 = nil
			}
		}
		done <- err2
		close(done)
	}()

	// Wait for the server to be ready.
	c, err := ollama.New(ctx, genai.ProviderOptionRemote(u))
	if err != nil {
		_ = cmd.Cancel()
		<-done
		return nil, fmt.Errorf("failed to create ollama client: %w", err)
	}
	// Loop until the server is healthy, process exits or the context is canceled.
	for ctx.Err() == nil {
		if _, err := c.ListModels(ctx); err == nil {
			break
		}
		select {
		case err := <-done:
			return nil, fmt.Errorf("starting ollama server failed while querying for health: %w", err)
		case <-ctx.Done():
			_ = cmd.Cancel()
			<-done
			return nil, ctx.Err()
		case <-time.After(20 * time.Millisecond):
		}
	}

	return &Server{url: u, done: done, cmd: cmd}, nil
}

// Close stops the Ollama server and waits for it to exit.
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

// DownloadRelease downloads a specific release from GitHub into the specified
// directory and returns the file path to ollama executable.
//
// When unspecified, the latest release is downloaded.
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
	ollamaexe := filepath.Join(cache, "ollama"+execSuffix)
	if _, err := os.Stat(ollamaexe); err == nil {
		// Run it to confirm the version and that the file is not corrupted.
		// If this fails, starts from scratch.
		cmd := exec.CommandContext(ctx, ollamaexe, "--version")
		// Disallow from connecting to the server.
		cmd.Env = append(os.Environ(), "OLLAMA_HOST=http://127.0.0.1:0")
		if out, err2 := cmd.CombinedOutput(); err2 == nil {
			re := regexp.MustCompile(`client version is (\d+\.\d+.\d+)`)
			if m := re.FindStringSubmatch(string(out)); len(m) == 2 {
				if "v"+m[1] == version {
					return ollamaexe, nil
				}
			}
		}
	}

	archiveName := ""
	wantedFiles := []string{filepath.Base(ollamaexe)}
	switch runtime.GOOS {
	case "darwin":
		archiveName = "ollama-darwin.tgz"
		wantedFiles = append(wantedFiles, "*.dylib", "*.so")
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
		wantedFiles = append(wantedFiles, "*.so", "*.so.*")
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
	if err := ghrelease.DownloadAndExtract(ctx, dlURL, cache, wantedFiles); err != nil {
		return "", fmt.Errorf("failed to download and extract %s from github: %w", archiveName, err)
	}
	return ollamaexe, nil
}
