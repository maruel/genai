// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llamacppsrv downloads llama.cpp executables and starts llama-server.
package llamacppsrv

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/ghrelease"
	"github.com/maruel/genai/providers/llamacpp"
)

// Version is the stable llama.cpp release last tested.
// Stable releases point to the nightly build containing their binaries.
const Version = "v0.9.0"

// Server is a llama-server instance.
type Server struct {
	url  string
	done <-chan error
	cmd  *exec.Cmd
}

// New starts llama-server and waits until it is healthy.
//
// exe must be an absolute path. Config supplies the model, projector, network,
// resource limits, extra arguments and output destination. Config.Args may
// contain -hf/-hff instead of a local model.
//
// New does not pass -ngl by default, so the caller can set it in Config.Args.
func New(ctx context.Context, exe string, cfg *Config) (*Server, error) {
	if !filepath.IsAbs(exe) {
		return nil, errors.New("exe must be an absolute path")
	}
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	hostPort := cfg.hostPort()
	host, portStr, err := net.SplitHostPort(hostPort)
	if err != nil {
		return nil, err
	}
	port, err := strconv.Atoi(portStr)
	if err != nil || port < 0 || port > 65535 {
		return nil, errors.New("port must be between 0 and 65535")
	}
	if port == 0 {
		// First try the default port.
		var l net.Listener
		lc := net.ListenConfig{}
		if l, err = lc.Listen(ctx, "tcp", net.JoinHostPort(host, "8080")); err != nil {
			if l, err = lc.Listen(ctx, "tcp", net.JoinHostPort(host, "0")); err != nil {
				return nil, err
			}
		}
		port = l.Addr().(*net.TCPAddr).Port //nolint:forcetypeassert // A TCP listener returns a TCP address.
		if err := l.Close(); err != nil {
			return nil, err
		}
	}
	u := "http://" + net.JoinHostPort(host, strconv.Itoa(port))
	args := cfg.args(exe, port)
	log.Printf("Args: %s", args)
	cmd := exec.CommandContext(ctx, args[0], args[1:]...) // #nosec G702 -- The caller explicitly selects an absolute executable; arguments use no shell.
	hideConsole(cmd)
	// Make sure dynamic libraries will be found.
	cmd.Dir = filepath.Dir(exe)
	cmd.Env = serverEnv(cmd.Dir)
	if cfg.LogOutput != nil {
		cmd.Stdout = cfg.LogOutput
		cmd.Stderr = cfg.LogOutput
	} else {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	// Context cancellation and Close share one shutdown signal. llama-server's
	// second SIGINT forces exit(1), interrupting its first graceful shutdown.
	var signalOnce sync.Once
	var signalErr error
	var shutdownRequested atomic.Bool
	cmd.Cancel = func() error {
		signalOnce.Do(func() {
			shutdownRequested.Store(true)
			if runtime.GOOS == "windows" {
				signalErr = cmd.Process.Kill()
			} else {
				signalErr = cmd.Process.Signal(os.Interrupt)
			}
		})
		return signalErr
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	done := make(chan error, 1)
	go func() {
		err2 := cmd.Wait()
		// CommandContext reports its context even when the one shutdown signal
		// produced a successful exit. That is expected termination, not cleanup failure.
		if ctx.Err() != nil && errors.Is(err2, ctx.Err()) && cmd.ProcessState != nil && cmd.ProcessState.Success() {
			err2 = nil
		}
		if er, ok := errors.AsType[*exec.ExitError](err2); ok {
			s, ok := er.Sys().(syscall.WaitStatus)
			if ok && shutdownRequested.Load() && s.Signaled() && s.Signal() == os.Interrupt {
				// The owner requested this interrupt. Other signals remain failures,
				// including crashes or external termination during shutdown.
				err2 = nil
			}
			// TODO: on Windows, figure out how to differentiate between normal quitting and an error.
		}
		done <- err2
		close(done)
	}()

	srv := &Server{url: u, done: done, cmd: cmd}
	// Wait for the server to be ready.
	c, err := llamacpp.New(ctx, genai.ProviderOptionRemote(u))
	if err != nil {
		return nil, errors.Join(fmt.Errorf("failed to create llamacpp client: %w", err), srv.Close())
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
			return nil, errors.Join(ctx.Err(), srv.Close())
		case <-time.After(20 * time.Millisecond):
		}
	}

	if err := ctx.Err(); err != nil {
		return nil, errors.Join(err, srv.Close())
	}
	return srv, nil
}

// Close stops the server and waits for it to exit.
func (s *Server) Close() error {
	err := s.cmd.Cancel()
	if errors.Is(err, os.ErrProcessDone) {
		err = nil
	}
	return errors.Join(err, <-s.done)
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

// Config configures a managed llama-server process.
type Config struct {
	// ModelPath is an absolute path to a local GGUF model. Empty is valid when
	// Args supplies -hf, -hff, or --model.
	ModelPath string
	// ProjectorPath is an absolute path to an mmproj GGUF. Empty disables --mmproj.
	// Multimodal models such as LFM3-Audio require it.
	ProjectorPath string
	// HostPort is host:port; empty defaults to localhost:0. A host without a port
	// defaults to port 0.
	HostPort string
	// Threads uses the normal server default when zero.
	Threads int
	// Slots sets --parallel. Zero leaves the server default.
	Slots int
	// ContextSize is tokens per slot. Zero leaves the server default.
	ContextSize int
	// Args are appended verbatim after the generated arguments.
	Args []string
	// LogOutput receives server stdout and stderr. Nil uses the process streams.
	LogOutput io.Writer
}

// Validate checks arguments that do not depend on the executable or the network.
func (c *Config) Validate() error {
	if c == nil {
		return errors.New("server configuration is required")
	}
	for _, p := range []string{c.ModelPath, c.ProjectorPath} {
		if p != "" && !filepath.IsAbs(p) {
			return errors.New("model and projector paths must be absolute")
		}
	}
	if c.Threads < 0 || c.Slots < 0 || c.ContextSize < 0 {
		return errors.New("threads, slots and context size must not be negative")
	}
	if c.Slots > 0 && c.ContextSize > math.MaxInt/c.Slots {
		return errors.New("total context size overflows")
	}
	hostPort := c.hostPort()
	host, port, err := net.SplitHostPort(hostPort)
	if err != nil {
		return fmt.Errorf("host:port: %w", err)
	}
	if ip := net.ParseIP(host); ip != nil && ip.To4() == nil {
		return errors.New("host must be an IPv4 address or hostname")
	}
	n, err := strconv.Atoi(port)
	if err != nil || n < 0 || n > 65535 {
		return errors.New("port must be between 0 and 65535")
	}
	return nil
}

func (c *Config) hostPort() string {
	if c.HostPort == "" {
		return "localhost:0"
	}
	if !strings.Contains(c.HostPort, ":") {
		return c.HostPort + ":0"
	}
	return c.HostPort
}

func (c *Config) args(exe string, port int) []string {
	host, _, _ := net.SplitHostPort(c.hostPort())
	threads := c.Threads
	if threads == 0 {
		// Surprisingly llama-server seems to be hardcoded to 8 threads. Leave 2
		// cores (especially critical when HT) to allow us to get some CPU time.
		if threads = runtime.NumCPU() - 2; threads == 0 {
			threads = 1
		}
	}
	args := []string{exe}
	if c.ModelPath != "" {
		args = append(args, "--model", c.ModelPath)
	}
	args = append(args, "--metrics", "--threads", strconv.Itoa(threads), "--port", strconv.Itoa(port))
	if host != "" {
		args = append(args, "--host", host)
	}
	if c.ProjectorPath != "" {
		args = append(args, "--mmproj", c.ProjectorPath)
	}
	if c.Slots > 0 {
		args = append(args, "--parallel", strconv.Itoa(c.Slots))
	}
	if c.ContextSize > 0 {
		slots := c.Slots
		if slots == 0 {
			slots = 1
		}
		args = append(args, "--ctx-size", strconv.Itoa(c.ContextSize*slots))
	}
	return append(args, c.Args...)
}

// DownloadVersion downloads a stable release (vX.Y.Z) or nightly build (bNNNN)
// into cache and returns the path to llama-server in a build-specific directory.
//
// Stable releases are resolved through their nightly-tag.txt asset. A relative
// symlink from the stable tag to its build directory persists that resolution
// after successful installation. When symlinks are unsupported or Windows
// lacks symlink privileges, a text file containing the build tag and a trailing
// newline serves as the alias.
// A cached resolution and matching executable need no GitHub requests.
// Missing or invalid executables are downloaded again for the resolved build.
func DownloadVersion(ctx context.Context, cache, version string) (string, error) {
	return downloadVersion(ctx, cache, version, toolServer)
}

// DownloadTTSVersion downloads a stable release (vX.Y.Z) or nightly build
// (bNNNN) into cache and returns the path to llama-tts in a build-specific
// directory. The release must contain llama-tts.
//
// Stable releases use the same cached resolution as DownloadVersion.
func DownloadTTSVersion(ctx context.Context, cache, version string) (string, error) {
	return downloadVersion(ctx, cache, version, toolTTS)
}

// DownloadRelease downloads a specific release from GitHub into the specified
// directory and returns the file path to llama.cpp executable.
//
// Returns the file path to the executable.
func DownloadRelease(ctx context.Context, cache string, version int) (string, error) {
	return downloadRelease(ctx, cache, version, toolServer)
}

type tool string

const (
	toolServer tool = "llama-server"
	toolTTS    tool = "llama-tts"
)

func downloadVersion(ctx context.Context, cache, version string, tool tool) (string, error) {
	if regexp.MustCompile(`^b[1-9]\d*$`).MatchString(version) {
		n, err := strconv.Atoi(version[1:])
		if err != nil {
			return "", fmt.Errorf("invalid build tag %q: %w", version, err)
		}
		return downloadRelease(ctx, filepath.Join(cache, version), n, tool)
	}
	if !regexp.MustCompile(`^v\d+\.\d+\.\d+$`).MatchString(version) {
		return "", fmt.Errorf("invalid llama.cpp release tag %q", version)
	}
	link := filepath.Join(cache, version)
	if tag, err := readReleaseAlias(link); err == nil {
		return downloadVersion(ctx, cache, tag, tool)
	} else if !errors.Is(err, os.ErrNotExist) {
		return "", fmt.Errorf("reading cached release %s: %w", version, err)
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
	exe, err := downloadVersion(ctx, cache, tag, tool)
	if err != nil {
		return "", err
	}
	// Symlink creation publishes the resolution atomically after installation.
	if err := os.Symlink(tag, link); err != nil {
		if errors.Is(err, os.ErrExist) {
			// Another caller may have published the same resolution.
			if target, er := readReleaseAlias(link); er == nil && target == tag {
				return exe, nil
			}
		}
		if !symlinkUnsupported(err) {
			return "", fmt.Errorf("caching release %s: %w", version, err)
		}
		if err := writeReleaseAlias(link, tag); err != nil {
			return "", fmt.Errorf("caching release %s as text: %w", version, err)
		}
	}
	return exe, nil
}

func downloadRelease(ctx context.Context, cache string, version int, tool tool) (string, error) {
	if version <= 0 {
		return "", fmt.Errorf("invalid llama.cpp build number %d", version)
	}
	execSuffix := ""
	if runtime.GOOS == "windows" {
		execSuffix = ".exe"
	}
	executable := filepath.Join(cache, string(tool)+execSuffix)
	if cachedExecutable(executable) {
		// Run it to confirm the version and that the file is not corrupted.
		// If this fails, starts from scratch.
		cmd := exec.CommandContext(ctx, executable, "--version") // #nosec G702 -- The caller-owned cache selects the runtime; this version probe uses no shell.
		hideConsole(cmd)
		cmd.Dir = filepath.Dir(executable)
		cmd.Env = serverEnv(cmd.Dir)
		if out, err := cmd.CombinedOutput(); err == nil {
			if v, ok := parseBuildNumber(out); ok && v == version {
				return executable, nil
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
		cmd := exec.CommandContext(ctx, "nvcc", "--version")
		hideConsole(cmd)
		cuda = cmd.Run() == nil
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
	if !cachedExecutable(filepath.Join(dir, string(tool)+execSuffix)) {
		return "", fmt.Errorf("release %s did not contain %s%s", build, tool, execSuffix)
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
	return filepath.Join(cache, string(tool)+execSuffix), nil
}

// readReleaseAlias reads only the two supported alias types, without following
// a symlink whose target could be outside the cache.
func readReleaseAlias(path string) (string, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return "", err
	}
	var tag string
	switch {
	case info.Mode()&os.ModeSymlink != 0:
		tag, err = os.Readlink(path)
	case info.Mode().IsRegular():
		var b []byte
		b, err = os.ReadFile(path)
		if err == nil {
			var ok bool
			if tag, ok = strings.CutSuffix(string(b), "\n"); !ok {
				return "", fmt.Errorf("release alias %s has no trailing newline", path)
			}
		}
	default:
		return "", fmt.Errorf("release alias %s is not a symlink or regular file", path)
	}
	if err != nil {
		return "", err
	}
	if !regexp.MustCompile(`^b[1-9]\d*$`).MatchString(tag) {
		return "", fmt.Errorf("invalid nightly tag %q in release alias %s", tag, path)
	}
	return tag, nil
}

func symlinkUnsupported(err error) bool {
	// Win32 ERROR_PRIVILEGE_NOT_HELD is not exported on other platforms.
	return errors.Is(err, errors.ErrUnsupported) || runtime.GOOS == "windows" && errors.Is(err, syscall.Errno(1314))
}

// writeReleaseAlias publishes complete text metadata, never a partially written tag.
func writeReleaseAlias(path, tag string) error {
	f, err := os.CreateTemp(filepath.Dir(path), ".llama-alias-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(f.Name()) }()
	_, err = f.WriteString(tag + "\n")
	if err == nil {
		err = f.Sync()
	}
	if er := f.Close(); err != nil || er != nil {
		return errors.Join(err, er)
	}
	if target, err := readReleaseAlias(path); err == nil {
		if target == tag {
			return nil
		}
		return fmt.Errorf("release alias %s already targets %s, not %s", path, target, tag)
	} else if !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(f.Name(), path)
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
	var a string
	switch arch {
	case "amd64":
		a = "x64"
	case "arm64":
		a = "arm64"
	default:
		return nil, fmt.Errorf("unsupported llama.cpp platform %s/%s", goos, arch)
	}
	var pattern string
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
