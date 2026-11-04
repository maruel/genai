// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Scripted native audio peers for llama.cpp realtime tests.

package llamacpp_test

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/coder/websocket"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal/realtimetest"
	"github.com/maruel/genai/providers/llamacpp"
)

func realtimeClient(t *testing.T, remote string) *llamacpp.Client {
	c, err := llamacpp.New(t.Context(), genai.ProviderOptionRemote(remote))
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func realtimePeer(t *testing.T, script func(context.Context, *websocket.Conn)) string {
	return realtimetest.Peer(t, func(r *http.Request) {
		if r.URL.Path != "/v1/realtime" || r.URL.RawQuery != "" {
			t.Error("unexpected realtime path or query")
		}
	}, script)
}

const nativeSession = `{"type":"session.created","session":{"id":"local","type":"realtime","instructions":"","output_modalities":null,"tools":[],"max_output_tokens":"inf","audio":{"input":{"format":{"type":"audio/pcm","rate":16000},"turn_detection":null},"output":{"format":{"type":"audio/pcm","rate":24000}}}}}`

func realtimeSetup(t *testing.T, ctx context.Context, ws *websocket.Conn) {
	realtimetest.Write(t, ctx, ws, nativeSession)
	m := realtimetest.Read(t, ctx, ws)
	v, ok := m["session"].(map[string]any)
	if !ok || m["type"] != "session.update" {
		t.Error("missing setup update")
		return
	}
	v["id"] = "local"
	b, err := json.Marshal(map[string]any{"type": "session.updated", "session": v})
	if err != nil {
		t.Error(err)
		return
	}
	realtimetest.Write(t, ctx, ws, string(b))
}
