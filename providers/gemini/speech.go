// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Dedicated Gemini transcription and streamed text-to-speech through Interactions.

package gemini

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"sync"
)

const speechEndpoint = "https://generativelanguage.googleapis.com/v1beta/interactions"

// interactionWriter bounds the complete spooled JSON request, including metadata.
type interactionWriter struct {
	w    io.Writer
	size int
}

func (w *interactionWriter) Write(p []byte) (int, error) {
	if len(p) > 20_000_000-w.size {
		return 0, errors.New("inline speech request exceeds 20 MB")
	}
	n, err := w.w.Write(p)
	w.size += n
	return n, err
}

func spoolInteractionRequest(ctx context.Context, in *InteractionRequest) (_ *interactionUpload, err error) {
	f, err := os.CreateTemp("", "genai-gemini-audio-*.json")
	if err != nil {
		return nil, err
	}
	defer func() {
		if err != nil {
			err = errors.Join(err, f.Close(), os.Remove(f.Name()))
		}
	}()
	w := &interactionWriter{w: f}
	header := *in
	header.Input = nil
	data, err := json.Marshal(&header)
	if err != nil {
		return nil, err
	}
	if _, err = w.Write(data[:len(data)-1]); err != nil {
		return nil, err
	}
	if _, err = io.WriteString(w, `,"input":[`); err != nil {
		return nil, err
	}
	size := 0
	for i := range in.Input {
		if i != 0 {
			if _, err = io.WriteString(w, ","); err != nil {
				return nil, err
			}
		}
		content := in.Input[i]
		data, err = json.Marshal(&content)
		if err != nil {
			return nil, err
		}
		if content.Source == nil {
			if _, err = w.Write(data); err != nil {
				return nil, err
			}
			continue
		}
		if _, err = w.Write(data[:len(data)-1]); err != nil {
			return nil, err
		}
		if _, err = io.WriteString(w, `,"data":"`); err != nil {
			return nil, err
		}
		if _, err = content.Source.Seek(0, io.SeekStart); err != nil {
			return nil, err
		}
		encoder := base64.NewEncoder(base64.StdEncoding, w)
		buf := make([]byte, 32*1024)
		sourceSize := 0
		for {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			n, readErr := content.Source.Read(buf)
			size += n
			sourceSize += n
			if size > 14_000_000 {
				return nil, errors.New("inline speech audio exceeds 14 MB; upload the recording and use its URL")
			}
			if _, err = encoder.Write(buf[:n]); err != nil {
				return nil, err
			}
			if readErr != nil {
				if !errors.Is(readErr, io.EOF) {
					return nil, readErr
				}
				break
			}
		}
		if sourceSize == 0 {
			return nil, errors.New("speech audio is empty")
		}
		if err := encoder.Close(); err != nil {
			return nil, err
		}
		if _, err = io.WriteString(w, `"}`); err != nil {
			return nil, err
		}
	}
	if _, err = io.WriteString(w, `]}`); err != nil {
		return nil, err
	}
	u := &interactionUpload{ctx: ctx, file: f, size: int64(w.size)}
	u.stop = context.AfterFunc(ctx, u.cancel)
	return u, nil
}

// interactionUpload retains the private spool until the call and all HTTP bodies
// release it. Its lock serializes file reads, body closes, admission and cleanup.
// Cancellation revokes every reader. roundtrippers v0.5.1 Retry acquires GetBody
// before its backoff select and can return on cancellation without closing that
// unhanded body. Revocation closes that ownership gap without waiting forever
// for a transport that never received it. Normal early responses retain readers.
type interactionUpload struct {
	ctx  context.Context
	file *os.File
	size int64
	stop func() bool

	mu       sync.Mutex
	bodies   int
	released bool
	canceled bool
	closed   bool
	err      error
}

func (u *interactionUpload) open() (io.ReadCloser, error) {
	u.mu.Lock()
	defer u.mu.Unlock()
	if err := u.ctx.Err(); err != nil {
		u.cancelLocked()
		return nil, err
	}
	if u.closed {
		return nil, os.ErrClosed
	}
	u.bodies++
	return &interactionUploadBody{upload: u, reader: io.NewSectionReader(u.file, 0, u.size)}, nil
}
func (u *interactionUpload) release() error {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.released = true
	u.cleanupLocked()
	return u.err
}
func (u *interactionUpload) cancel() {
	u.mu.Lock()
	defer u.mu.Unlock()
	u.cancelLocked()
}
func (u *interactionUpload) cancelLocked() {
	u.canceled = true
	u.bodies = 0
	u.cleanupLocked()
}
func (u *interactionUpload) cleanupLocked() {
	if !u.closed && u.released && u.bodies == 0 {
		u.closed = true
		u.err = errors.Join(u.file.Close(), os.Remove(u.file.Name()))
		u.stop()
	}
}

// interactionUploadBody has independent replay offsets. Close is idempotent so
// cancellation and the transport can both relinquish the same body safely.
type interactionUploadBody struct {
	upload *interactionUpload
	reader *io.SectionReader
	// Protected by upload.mu.
	closed bool
}

func (b *interactionUploadBody) Read(p []byte) (int, error) {
	u := b.upload
	u.mu.Lock()
	defer u.mu.Unlock()
	if err := u.ctx.Err(); err != nil {
		u.cancelLocked()
		return 0, err
	}
	if b.closed || u.closed {
		return 0, os.ErrClosed
	}
	return b.reader.Read(p)
}
func (b *interactionUploadBody) Close() error {
	u := b.upload
	u.mu.Lock()
	defer u.mu.Unlock()
	if !b.closed {
		b.closed = true
		if !u.canceled {
			u.bodies--
		}
		u.cleanupLocked()
	}
	return u.err
}

// interactionStream owns one response body. finish is deliberately idempotent:
// iteration closes it on completion/early break, and the public finalizer also
// handles the case where the caller never starts iteration.
type interactionStream struct {
	body   io.ReadCloser
	scan   *bufio.Scanner
	err    error
	closed bool
}

func (s *interactionStream) finish() error {
	if !s.closed {
		s.closed = true
		s.err = errors.Join(s.err, s.body.Close())
	}
	return s.err
}
func (s *interactionStream) iterate(yield func(InteractionEvent) bool) {
	defer func() { _ = s.finish() }()
	if s.closed {
		return
	}
	for {
		event, err := s.next()
		if err != nil {
			s.err = err
			return
		}
		terminal := false
		switch event.EventType {
		case InteractionEventCompleted:
			complete := event.Interaction
			if complete.Status == "" {
				complete.Status = InteractionCompleted
			}
			s.err = complete.checkCompleted()
			terminal = true
		case InteractionEventError:
			if event.Error == nil {
				s.err = errors.New("speech stream reported an unspecified error")
			} else {
				s.err = event.Error
			}
			terminal = true
		case InteractionEventStatus:
			if event.Status != InteractionQueued && event.Status != InteractionInProgress && event.Status != InteractionCompleted {
				s.err = fmt.Errorf("speech stream status %q", event.Status)
				terminal = true
			}
		case InteractionEventDelta, InteractionEventCreated, InteractionEventStart, InteractionEventStop:
		default:
			s.err = fmt.Errorf("unexpected speech stream event %q", event.EventType)
			terminal = true
		}
		if !yield(event) || terminal {
			return
		}
	}
}
func (s *interactionStream) next() (InteractionEvent, error) {
	var data bytes.Buffer
	for s.scan.Scan() {
		line := s.scan.Bytes()
		if len(line) == 0 {
			if data.Len() != 0 {
				var out InteractionEvent
				err := json.Unmarshal(data.Bytes(), &out)
				return out, err
			}
			continue
		}
		if bytes.HasPrefix(line, []byte("data:")) {
			if data.Len()+len(line) > 1024*1024 {
				return InteractionEvent{}, errors.New("speech stream event exceeds 1 MiB")
			}
			data.Write(bytes.TrimPrefix(line[5:], []byte(" ")))
			data.WriteByte('\n')
		} else if line[0] != ':' && !bytes.HasPrefix(line, []byte("event:")) && !bytes.HasPrefix(line, []byte("id:")) {
			return InteractionEvent{}, errors.New("invalid speech event stream")
		}
	}
	if err := s.scan.Err(); err != nil {
		return InteractionEvent{}, err
	}
	return InteractionEvent{}, io.ErrUnexpectedEOF
}
