// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for managed server configuration.

package llamacppsrv

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"
)

func TestConfig(t *testing.T) {
	t.Run("args", func(t *testing.T) {
		cfg := Config{
			ModelPath:     filepath.Join(t.TempDir(), "model.gguf"),
			ProjectorPath: filepath.Join(t.TempDir(), "projector.gguf"),
			HostPort:      "127.0.0.1:19961",
			Threads:       4,
			Slots:         2,
			ContextSize:   16384,
		}
		got := cfg.args("/bin/llama-server", 19961)
		want := []string{
			"/bin/llama-server",
			"--model", cfg.ModelPath,
			"--metrics", "--threads", "4", "--port", "19961", "--host", "127.0.0.1",
			"--mmproj", cfg.ProjectorPath,
			"--parallel", "2", "--ctx-size", "32768",
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("args %v, want %v", got, want)
		}
	})
	t.Run("defaults", func(t *testing.T) {
		cfg := Config{}
		if err := cfg.Validate(); err != nil {
			t.Fatal(err)
		}
		got := cfg.args("/bin/llama-server", 8080)
		if len(got) != 8 || got[1] != "--metrics" || got[2] != "--threads" || got[6] != "--host" || got[7] != "localhost" {
			t.Fatalf("args %v", got)
		}
	})
	t.Run("Validate", func(t *testing.T) {
		t.Run("valid", func(t *testing.T) {
			for _, cfg := range []Config{
				{},
				{HostPort: "localhost"},
				{ModelPath: filepath.Join(t.TempDir(), "model.gguf"), ProjectorPath: filepath.Join(t.TempDir(), "projector.gguf")},
				{HostPort: "127.0.0.1:65535"},
			} {
				if err := cfg.Validate(); err != nil {
					t.Fatal(err)
				}
			}
		})
		t.Run("error", func(t *testing.T) {
			root := t.TempDir()
			for _, cfg := range []*Config{
				nil,
				{ModelPath: "model.gguf"},
				{ProjectorPath: "projector.gguf"},
				{ModelPath: filepath.Join(root, "model.gguf"), ProjectorPath: filepath.Join(root, "projector.gguf"), Threads: -1},
				{ModelPath: filepath.Join(root, "model.gguf"), ProjectorPath: filepath.Join(root, "projector.gguf"), Slots: -1},
				{ModelPath: filepath.Join(root, "model.gguf"), ProjectorPath: filepath.Join(root, "projector.gguf"), ContextSize: -1},
				{ModelPath: filepath.Join(root, "model.gguf"), ProjectorPath: filepath.Join(root, "projector.gguf"), Slots: 2, ContextSize: math.MaxInt},
				{ModelPath: filepath.Join(root, "model.gguf"), ProjectorPath: filepath.Join(root, "projector.gguf"), HostPort: "127.0.0.1:65536"},
				{ModelPath: filepath.Join(root, "model.gguf"), ProjectorPath: filepath.Join(root, "projector.gguf"), HostPort: "[::1]:0"},
			} {
				if err := cfg.Validate(); err == nil {
					t.Fatalf("accepted configuration %+v", cfg)
				}
			}
		})
	})
}

func TestNew(t *testing.T) {
	t.Run("error before process start", func(t *testing.T) {
		root := t.TempDir()
		for _, tc := range []struct {
			exe string
			cfg *Config
		}{
			{"relative/llama-server", &Config{}},
			{filepath.Join(root, "llama-server"), nil},
			{filepath.Join(root, "llama-server"), &Config{ModelPath: "model.gguf"}},
		} {
			if _, err := New(t.Context(), tc.exe, tc.cfg); err == nil {
				t.Fatalf("accepted executable %q config %+v", tc.exe, tc.cfg)
			}
		}
	})
}
