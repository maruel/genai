// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package internal is awesome sauce.
package internal

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"mime"
	"reflect"
	"strings"

	"github.com/invopop/jsonschema"
	"github.com/maruel/httpjson"
)

//go:generate go run regen_docs.go ..
//go:generate go run regen_readme.go ..
//go:generate go run regen_scoreboards.go ..

// BeLenient is used by all clients to enable or disable httpjson.Client.Lenient.
//
// It is true by default. Tests must manually set it to false.
var BeLenient = true

// Logger retrieves a slog.Logger from the context if any, otherwise returns slog.Default().
func Logger(ctx context.Context) *slog.Logger {
	v := ctx.Value(contextKey{})
	switch v := v.(type) {
	case *slog.Logger:
		return v
	default:
		return slog.Default()
	}
}

// WithLogger injects a slog.Logger into the context. It can be retrieved with Logger().
func WithLogger(ctx context.Context, logger *slog.Logger) context.Context {
	return context.WithValue(ctx, contextKey{}, logger)
}

// DecodeJSON is duplicate from httpjson.go in https://github.com/maruel/httpjson.
func DecodeJSON(d *json.Decoder, out any, r io.ReadSeeker) (bool, error) {
	d.UseNumber()
	if err := d.Decode(out); err != nil {
		// decode.object() in encoding/json.go does not return a structured error
		// when an unknown field is found or when the type is wrong. Process it manually.
		if r != nil {
			if s := err.Error(); strings.Contains(s, "json: unknown field ") || strings.Contains(s, "json: cannot unmarshal ") {
				// Decode again but this time capture all errors. Try first as a map (JSON object), then as a slice
				// (JSON list).
				for _, t := range []any{map[string]any{}, []any{}} {
					if _, err2 := r.Seek(0, 0); err2 != nil {
						// Unexpected.
						return false, err2
					}
					d = json.NewDecoder(r)
					d.UseNumber()
					if err2 := d.Decode(&t); err2 == nil {
						if err2 = errors.Join(httpjson.FindExtraKeys(reflect.TypeOf(out), t)...); err2 != nil {
							return true, err2
						}
					}
				}
			}
		}
		return false, err
	}
	return false, nil
}

//

// JSONSchemaFor returns the JSON schema for the given type.
//
// Many providers (including OpenAI) struggle with $ref that jsonschema package uses by default.
func JSONSchemaFor(t reflect.Type) *jsonschema.Schema {
	// No need to set an ID on the struct, it's unnecessary data that may confuse the tool.
	r := jsonschema.Reflector{Anonymous: true, DoNotReference: true}
	return r.ReflectFromType(t)
}

func MimeByExt(ext string) string {
	switch ext {
	case ".aac":
		return "audio/aac"
	case ".flac":
		return "audio/flac"
	case ".wav":
		return "audio/wav"
	default:
		return mime.TypeByExtension(ext)
	}
}

//

type contextKey struct{}
