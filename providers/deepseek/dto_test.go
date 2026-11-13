// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the DeepSeek wire types.

package deepseek_test

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/maruel/genai/providers/deepseek"
)

const cacheUsageJSON = `{
	"choices": [{"finish_reason": "stop", "message": {}}],
	"usage": {
		"prompt_tokens": 100,
		"completion_tokens": 10,
		"prompt_cache_hit_tokens": 30,
		"prompt_cache_miss_tokens": 70
	}
}`

func TestUsageReportsCacheMisses(t *testing.T) {
	t.Run("sync", func(t *testing.T) {
		var resp deepseek.ChatResponse
		if err := json.Unmarshal([]byte(cacheUsageJSON), &resp); err != nil {
			t.Fatal(err)
		}
		res, err := resp.ToResult()
		if err != nil {
			t.Fatal(err)
		}
		if res.Usage.InputCachedTokens != 30 || res.Usage.InputCacheMissTokens != 70 {
			t.Fatalf("cache usage = hit %d, miss %d; want 30, 70", res.Usage.InputCachedTokens, res.Usage.InputCacheMissTokens)
		}
	})

	t.Run("stream", func(t *testing.T) {
		var chunk deepseek.ChatStreamChunkResponse
		if err := json.Unmarshal([]byte(cacheUsageJSON), &chunk); err != nil {
			t.Fatal(err)
		}
		fragments, finish := deepseek.ProcessStream(slices.Values([]deepseek.ChatStreamChunkResponse{chunk}))
		for range fragments {
		}
		usage, _, err := finish()
		if err != nil {
			t.Fatal(err)
		}
		if usage.InputCachedTokens != 30 || usage.InputCacheMissTokens != 70 {
			t.Fatalf("cache usage = hit %d, miss %d; want 30, 70", usage.InputCachedTokens, usage.InputCacheMissTokens)
		}
	})
}
