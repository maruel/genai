// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llamacppsrv downloads and starts llama-server from
// llama.cpp, directly from GitHub releases.
package llamacppsrv

import (
	"archive/zip"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"syscall"
	"time"

	"github.com/maruel/genai/llamacpp"
)

// BuildNumber is the build number that was last tried from
// https://github.com/ggml-org/llama.cpp/releases
//
// You are free to use the build number that works best for you.
const BuildNumber = 5385

// Server is a llama-server instance.
type Server struct {
	port int
	done <-chan error
	cmd  *exec.Cmd
}

// NewServer creates a new instance of the llama-server and ensures the server
// is healthy.
//
// Doesn't pass "-ngl", "9999" by default so the user can override it.
//
// Output is redirected to logOutput if non-nil.
func NewServer(ctx context.Context, exe, modelPath string, logOutput io.Writer, hostPort string, threads int, extraArgs []string) (*Server, error) {
	if !filepath.IsAbs(exe) {
		return nil, errors.New("exe must be an absolute path")
	}
	if !filepath.IsAbs(modelPath) {
		return nil, errors.New("modelPath must be an absolute path")
	}
	host, portStr, err := net.SplitHostPort(hostPort)
	if err != nil {
		return nil, err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		return nil, err
	}
	if threads == 0 {
		// Surprisingly llama-server seems to be hardcoded to 8 threads. Leave 2
		// cores (especially critical when HT) to allow us to get some CPU time.
		if threads = runtime.NumCPU() - 2; threads == 0 {
			threads = 1
		}
	}
	args := []string{
		exe, "--model", modelPath, "--metrics", "--threads", strconv.Itoa(threads), "--port", portStr,
	}
	if host != "" {
		args = append(args, "--host", host)
	}
	args = append(args, extraArgs...)
	log.Printf("Args: %s", args)
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	// Make sure dynamic libraries will be found.
	cmd.Dir = filepath.Dir(exe)
	if runtime.GOOS != "windows" {
		cmd.Env = append(os.Environ(), "LD_LIBRARY_PATH="+cmd.Dir+":"+os.Getenv("LD_LIBRARY_PATH"))
	}
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
	if err = cmd.Start(); err != nil {
		return nil, err
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
	c, err := llamacpp.New("http://localhost:"+strconv.Itoa(port), nil, nil)
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

	return &Server{port: port, done: done, cmd: cmd}, nil
}

func (s *Server) Close() error {
	_ = s.cmd.Cancel()
	err := <-s.done
	return err
}

// URL returns the URL to the server.
func (s *Server) URL() string {
	return "http://localhost:" + strconv.Itoa(s.port)
}

// Done is a channel to listen to the server's termination. No need to call
// Close() if it is set.
func (s *Server) Done() <-chan error {
	return s.done
}

// DownloadRelease downloads a specific release from GitHub into the specified
// directory and returns the file path to llama.cpp executable.
//
// Returns the file path to the executable and true if it is llamafile, false
// if it is llama-server from llama.cpp.
func DownloadRelease(ctx context.Context, cache string, version int) (string, error) {
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	llamaserver := filepath.Join(cache, "llama-server"+execSuffix)
	if _, err := os.Stat(llamaserver); err == nil {
		// Run it to confirm the version and that the file is not corrupted.
		// If this fails, starts from scratch.
		if out, err := exec.CommandContext(ctx, llamaserver, "--version").CombinedOutput(); err == nil {
			if i := bytes.IndexByte(out, '\n'); i > 1 {
				re := regexp.MustCompile("^version: ([0-9]+) .+$")
				if m := re.FindStringSubmatch(string(out[0:i])); len(m) == 2 {
					if v, _ := strconv.Atoi(m[1]); v == version {
						return llamaserver, nil
					}
				}
			}
		}
	}

	build := "b" + strconv.Itoa(version)
	url := "https://github.com/ggml-org/llama.cpp/releases/download/" + build + "/"
	zipname := ""
	wantedFiles := []string{filepath.Base(llamaserver)}
	switch runtime.GOOS {
	case "darwin":
		// There's no point in supporting x64.
		zipname = "llama-" + build + "-bin-macos-arm64.zip"
		wantedFiles = append(wantedFiles, "*.dylib", "*.metal")
	case "linux":
		if runtime.GOARCH == "arm64" {
			zipname = "llama-" + build + "-bin-ubuntu-arm64.zip"
		} else if runtime.GOARCH == "amd64" {
			zipname = "llama-" + build + "-bin-ubuntu-x64.zip"
		} else {
			return "", fmt.Errorf("don't know how to select " + runtime.GOOS + "/" + runtime.GOARCH)
		}
		wantedFiles = append(wantedFiles, "*.so")
	case "windows":
		_, err := exec.Command("nvcc", "--version").CombinedOutput()
		if err == nil {
			// This is tricky because in the case of image generation, we may want to
			// run on the CPU instead.
			// TODO: We'll have to list the files on GH to determine the cuda version to get the exact filename. :(
			// TODO: Vulkan, HIP, OpenCL, sycl.
			zipname = "llama-" + build + "-bin-win-cuda12.4-x64.zip"
		} else if runtime.GOARCH == "arm64" {
			zipname = "llama-" + build + "-bin-win-cpu-arm64.zip"
		} else if runtime.GOARCH == "amd64" {
			zipname = "llama-" + build + "-bin-win-cpu-x64.zip"
		} else {
			return "", fmt.Errorf("don't know how to select " + runtime.GOOS + "/" + runtime.GOARCH)
		}
		wantedFiles = append(wantedFiles, "*.dll")
	default:
		return "", fmt.Errorf("don't know how to select " + runtime.GOOS + "/" + runtime.GOARCH)
	}
	zippath := filepath.Join(cache, zipname)
	if err := downloadFile(ctx, url+zipname, zippath); err != nil {
		return "", fmt.Errorf("failed to download %s from github: %w", zipname, err)
	}

	z, err := zip.OpenReader(zippath)
	if err != nil {
		return "", err
	}
	defer z.Close()
	for _, f := range z.File {
		// Files are under build/bin/; ignore path.
		n := filepath.Base(f.Name)
		for _, desired := range wantedFiles {
			if ok, _ := filepath.Match(desired, n); ok {
				var src io.ReadCloser
				if src, err = f.Open(); err != nil {
					return "", err
				}
				var dst io.WriteCloser
				if dst, err = os.OpenFile(filepath.Join(cache, n), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0o755); err == nil {
					_, err = io.CopyN(dst, src, int64(f.UncompressedSize64))
				}
				if err2 := src.Close(); err == nil {
					err = err2
				}
				if err2 := dst.Close(); err == nil {
					err = err2
				}
				if err != nil {
					return "", fmt.Errorf("failed to write %q: %w", n, err)
				}
			}
		}
	}
	return llamaserver, err
}

func downloadFile(ctx context.Context, url, dst string) error {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return err
	}
	if token := os.Getenv("GITHUB_TOKEN"); token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("unexpected http status code %d", resp.StatusCode)
	}
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
