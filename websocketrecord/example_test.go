// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Example usage of the websocketrecord package.

package websocketrecord_test

import (
	"context"
	"fmt"
	"net/http/httptest"
	"strings"

	"golang.org/x/net/websocket"

	"github.com/maruel/genai/websocketrecord"
)

func Example() {
	// Start a real WebSocket echo server for demonstration.
	echoSrv := httptest.NewServer(websocket.Handler(func(ws *websocket.Conn) {
		for {
			var msg string
			if err := websocket.Message.Receive(ws, &msg); err != nil {
				return
			}
			if err := websocket.Message.Send(ws, strings.ToUpper(msg)); err != nil {
				return
			}
		}
	}))
	defer echoSrv.Close()

	// Record a WebSocket session.
	rec, err := websocketrecord.New("testdata/example")
	if err != nil {
		fmt.Println("New:", err)
		return
	}
	defer func() { _ = rec.Stop() }()

	cfg, err := websocket.NewConfig("ws"+echoSrv.URL[4:], echoSrv.URL)
	if err != nil {
		fmt.Println("NewConfig:", err)
		return
	}
	c, err := rec.DialContext(context.Background(), cfg)
	if err != nil {
		fmt.Println("DialContext:", err)
		return
	}
	defer func() { _ = c.Close() }()

	// Use c.Send and c.Receive instead of websocket.Message.
	if err := c.Send("hello"); err != nil {
		fmt.Println("Send:", err)
		return
	}
	var resp string
	if err := c.Receive(&resp); err != nil {
		fmt.Println("Receive:", err)
		return
	}
	fmt.Println(resp)
	// Output:
	// HELLO
}
