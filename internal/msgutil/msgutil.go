// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package msgutil provides shared message utilities for subprocess-based
// providers.
package msgutil

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"

	"github.com/maruel/genai"
)

// ExtractOpaqueID extracts a string value from Reply.Opaque by walking messages
// in reverse order.
//
// It is used by subprocess-based providers to retrieve session or thread IDs
// stored in previous replies.
func ExtractOpaqueID(msgs genai.Messages, key string) string {
	for _, msg := range slices.Backward(msgs) {
		for j := range msg.Replies {
			if id, ok := msg.Replies[j].Opaque[key].(string); ok && id != "" {
				return id
			}
		}
	}
	return ""
}

// LastUserMsg returns the last user message in msgs.
//
// It is used by subprocess-based providers that send only the final user message to a subprocess.
// Earlier supplied user events are not serialized. Some callers resume CLI-owned sessions for context;
// others launch a fresh process. A successful call does not establish full-history or burst support.
func LastUserMsg(msgs genai.Messages) (genai.Message, error) {
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role() == "user" {
			if len(msgs[i].Requests) == 0 {
				return genai.Message{}, errors.New("last user message has no content")
			}
			return msgs[i], nil
		}
	}
	return genai.Message{}, errors.New("no user message found in msgs")
}

// WriteNDJSON marshals v as JSON and writes it followed by a newline.
//
// It is used by subprocess-based providers that communicate via NDJSON over
// stdin/stdout.
func WriteNDJSON(w io.Writer, v any) error {
	data, err := json.Marshal(v)
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = w.Write(data)
	return err
}

// CheckExtraArgs rejects command-line arguments that set a reserved flag.
//
// It is used by subprocess-based providers to validate GenOption.ExtraArgs
// against the flags the provider owns to drive its protocol. A flag matches
// with one or two leading dashes and with or without an "=value" suffix.
func CheckExtraArgs(args []string, reserved ...string) error {
	for _, a := range args {
		if a == "" {
			return errors.New("GenOption.ExtraArgs: empty argument")
		}
		if !strings.HasPrefix(a, "-") {
			continue
		}
		name, _, _ := strings.Cut(strings.TrimPrefix(strings.TrimPrefix(a, "-"), "-"), "=")
		if slices.Contains(reserved, name) {
			return fmt.Errorf("GenOption.ExtraArgs: %q is reserved by the provider", a)
		}
	}
	return nil
}

// SelectGPTTier returns the newest model ID of the form
// prefix+"gpt-<version>-<tier>" for marker.
//
// The tiers follow the Codex model list descriptions: luna is "fast and
// affordable" (genai.ModelCheap), terra is balanced (genai.ModelGood), and
// astra is "frontier intelligence" (genai.ModelSOTA). IDs with a further
// suffix, such as "-fast" service tiers, do not match.
func SelectGPTTier(models []genai.Model, prefix string, marker genai.ProviderOptionModel) (string, error) {
	var tier string
	switch marker {
	case genai.ModelCheap:
		tier = "luna"
	case genai.ModelGood:
		tier = "terra"
	case genai.ModelSOTA:
		tier = "astra"
	default:
		return "", fmt.Errorf("%q is not a model marker", marker)
	}
	best := ""
	var bestVer []int
	for _, m := range models {
		id := m.GetID()
		rest, ok := strings.CutPrefix(id, prefix+"gpt-")
		if !ok {
			continue
		}
		if rest, ok = strings.CutSuffix(rest, "-"+tier); !ok {
			continue
		}
		ver, ok := parseVersion(rest)
		if ok && (best == "" || slices.Compare(ver, bestVer) > 0) {
			best, bestVer = id, ver
		}
	}
	if best == "" {
		return "", fmt.Errorf("no %sgpt-*-%s model available for %s", prefix, tier, marker)
	}
	return best, nil
}

// parseVersion parses a dotted version such as "6" or "6.1".
func parseVersion(s string) ([]int, bool) {
	var ver []int
	for p := range strings.SplitSeq(s, ".") {
		n, err := strconv.Atoi(p)
		if err != nil || n < 0 {
			return nil, false
		}
		ver = append(ver, n)
	}
	return ver, true
}
