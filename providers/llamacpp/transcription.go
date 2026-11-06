// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Dedicated live transcription through llama.cpp's streaming audio encoder.

package llamacpp

import (
	"context"
	"encoding/json"
	"errors"

	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/realtime"
)

// TranscriptionConn owns native llamacpp WebSocket I/O, without transcript accumulation
// or utterance-commit policy. One receiver and one sender may run concurrently.
// Canceling an I/O context aborts the connection. The caller closes it once.
type TranscriptionConn struct {
	// Initial is the complete native setup acknowledgement, including negotiated
	// session fields. Treat it as immutable; it does not drive transport policy.
	Initial ServerMessage

	ws *realtime.Conn
}

// Send writes a native client message. Audio encoding and turn controls belong
// to the caller; no text accumulation or commit policy is applied.
func (c *TranscriptionConn) Send(ctx context.Context, in *ClientMessage) error {
	if in == nil {
		return errors.New("native client message is required")
	}
	data, err := json.Marshal(in)
	if err != nil {
		return err
	}
	return c.ws.Write(ctx, data)
}

// Receive returns one complete native server message. Invalid JSON terminates
// the connection; native provider error payloads remain available to the caller.
func (c *TranscriptionConn) Receive(ctx context.Context) (ServerMessage, error) {
	var msg ServerMessage
	data, err := c.ws.Read(ctx)
	if err != nil {
		return msg, err
	}
	if err := internal.UnmarshalJSON(data, &msg); err != nil {
		c.ws.Abort(err)
		return ServerMessage{}, err
	}
	return msg, nil
}

// Abort terminates pending reads/writes with the caller's cause. Close is still
// required to wait for transport cleanup.
func (c *TranscriptionConn) Abort(err error) { c.ws.Abort(err) }

// Close terminates I/O and waits for socket cleanup.
func (c *TranscriptionConn) Close() error { return c.ws.Close() }
