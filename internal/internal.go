// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package internal

import (
	"errors"
	"fmt"
	"io"
	"mime"
	"path"
	"path/filepath"

	"github.com/maruel/genai/genaiapi"
)

func ParseDocument(m *genaiapi.Message, maxSize int64) (string, []byte, error) {
	mimeType := mime.TypeByExtension(filepath.Ext(m.Filename))
	if m.URL != "" {
		// Not all provider require a mime-type so do not error out.
		if mimeType == "" {
			mimeType = mime.TypeByExtension(filepath.Ext(path.Base(m.URL)))
		}
		return mimeType, nil, nil
	}
	if mimeType == "" {
		return "", nil, errors.New("failed to determine mime-type, pass a filename with an extension")
	}
	size, err := m.Document.Seek(0, io.SeekEnd)
	if err != nil {
		return "", nil, fmt.Errorf("failed to seek data: %w", err)
	}
	if size > maxSize {
		return "", nil, fmt.Errorf("large files are not yet supported, max %dMiB", maxSize/1024/1024)
	}
	if _, err = m.Document.Seek(0, io.SeekStart); err != nil {
		return "", nil, fmt.Errorf("failed to seek data: %w", err)
	}
	var data []byte
	if data, err = io.ReadAll(m.Document); err != nil {
		return "", nil, fmt.Errorf("failed to read data: %w", err)
	}
	if len(data) == 0 {
		return "", nil, errors.New("empty data")
	}
	return mimeType, data, nil
}
