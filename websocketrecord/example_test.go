// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Example usage of the websocketrecord package.

package websocketrecord_test

import (
	"context"
	"fmt"

	"github.com/coder/websocket"

	"github.com/maruel/genai/websocketrecord"
)

func Example() {
	rec, err := websocketrecord.New("testdata/example")
	if err != nil {
		fmt.Println("New:", err)
		return
	}
	defer func() { _ = rec.Close() }()

	if err := rec.Start(""); err != nil {
		fmt.Println("Start:", err)
		return
	}
	c, _, err := websocket.Dial(context.Background(), rec.WSURL(), &websocket.DialOptions{})
	if err != nil {
		fmt.Println("Dial:", err)
		return
	}
	defer func() { _ = c.CloseNow() }()

	if err := c.Write(context.Background(), websocket.MessageText, []byte("hello")); err != nil {
		fmt.Println("Write:", err)
		return
	}
	_, data, err := c.Read(context.Background())
	if err != nil {
		fmt.Println("Read:", err)
		return
	}
	fmt.Println(string(data))
	// Output:
	// HELLO
}
