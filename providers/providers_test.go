// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the provider registry.

package providers

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/providers/anthropic"
)

func TestAvailable(t *testing.T) {
	orig := All
	t.Cleanup(func() { All = orig })
	for _, tc := range []struct {
		name       string
		ping       bool
		pingErr    error
		factoryErr error
		closeErr   error
		nilClient  bool
		want       bool
	}{
		{name: "no_ping", want: true},
		{name: "ping_success", ping: true, want: true},
		{name: "ping_error", ping: true, pingErr: errors.New("ping failed")},
		{name: "factory_error", factoryErr: errors.New("factory failed")},
		{name: "nil_client", factoryErr: errors.New("factory failed"), nilClient: true},
		{name: "close_error", closeErr: errors.New("close failed"), want: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var p *probeClient
			if !tc.nilClient {
				c, err := anthropic.New(t.Context(), genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("test"))
				if err != nil {
					t.Fatal(err)
				}
				p = &probeClient{Client: c, closeErr: tc.closeErr}
			}
			var log bytes.Buffer
			ctx := internal.WithLogger(t.Context(), slog.New(slog.NewTextHandler(&log, nil)))
			pinged := false
			All = map[string]Config{
				"probe": {Factory: func(context.Context, ...genai.ProviderOption) (genai.Provider, error) {
					if tc.nilClient {
						return nil, tc.factoryErr
					}
					if tc.ping {
						return &pingProbeClient{probeClient: p, ping: func(context.Context) error {
							pinged = true
							return tc.pingErr
						}}, tc.factoryErr
					}
					return p, tc.factoryErr
				}},
			}
			avail := Available(ctx)
			if _, ok := avail["probe"]; ok != tc.want {
				t.Errorf("available = %v, want %v", ok, tc.want)
			}
			if !tc.nilClient && !p.closed {
				t.Error("probe client remains open")
				if err := p.Client.Close(); err != nil {
					t.Fatal(err)
				}
			}
			if tc.closeErr != nil {
				for _, want := range []string{"level=WARN", "provider=probe", `err="close failed"`} {
					if !strings.Contains(log.String(), want) {
						t.Errorf("log %q does not contain %q", log.String(), want)
					}
				}
			}
			if want := tc.ping && tc.factoryErr == nil; pinged != want {
				t.Errorf("pinged = %v, want %v", pinged, want)
			}
		})
	}
	t.Run("concurrent", func(t *testing.T) {
		const count = 8
		ctx, cancel := context.WithTimeout(t.Context(), 5*time.Second)
		t.Cleanup(cancel)
		factories := make(chan struct{}, count)
		pings := make(chan struct{}, count)
		releaseFactories := make(chan struct{})
		releasePings := make(chan struct{})
		ps := make([]*probeClient, count)
		All = make(map[string]Config, count)
		for i := range count {
			c, err := anthropic.New(ctx, genai.ProviderOptionAPIKey("test"), genai.ProviderOptionModel("test"))
			if err != nil {
				t.Fatal(err)
			}
			p := &probeClient{Client: c}
			ps[i] = p
			All[strconv.Itoa(i)] = Config{Factory: func(ctx context.Context, _ ...genai.ProviderOption) (genai.Provider, error) {
				factories <- struct{}{}
				select {
				case <-releaseFactories:
				case <-ctx.Done():
					return p, ctx.Err()
				}
				return &pingProbeClient{probeClient: p, ping: func(ctx context.Context) error {
					pings <- struct{}{}
					select {
					case <-releasePings:
						return nil
					case <-ctx.Done():
						return ctx.Err()
					}
				}}, nil
			}}
		}
		result := make(chan map[string]Config, 1)
		go func() { result <- Available(ctx) }()
		for _, stage := range []struct {
			name    string
			started <-chan struct{}
			release chan struct{}
		}{
			{name: "factories", started: factories, release: releaseFactories},
			{name: "pings", started: pings, release: releasePings},
		} {
			for range count {
				select {
				case <-stage.started:
				case <-ctx.Done():
					<-result
					t.Fatalf("%s did not run concurrently: %v", stage.name, ctx.Err())
				}
			}
			close(stage.release)
		}
		avail := <-result
		if len(avail) != count {
			t.Errorf("available count = %d, want %d", len(avail), count)
		}
		for i, p := range ps {
			if _, ok := avail[strconv.Itoa(i)]; !ok {
				t.Errorf("provider %d missing", i)
			}
			if !p.closed {
				t.Errorf("provider %d remains open after Available returns", i)
				if err := p.Client.Close(); err != nil {
					t.Error(err)
				}
			}
		}
	})
}

type probeClient struct {
	*anthropic.Client
	closed   bool
	closeErr error
}

func (p *probeClient) Close() error {
	p.closed = true
	return errors.Join(p.Client.Close(), p.closeErr)
}

type pingProbeClient struct {
	*probeClient
	ping func(context.Context) error
}

func (p *pingProbeClient) Ping(ctx context.Context) error {
	return p.ping(ctx)
}
