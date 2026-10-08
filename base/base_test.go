// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the base package.

package base

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
)

func TestCheckDuplicateGenOptions(t *testing.T) {
	t.Run("no_duplicates", func(t *testing.T) {
		opts := []genai.GenOption{
			genai.GenOptionSeed(1),
			&genai.GenOptionText{},
		}
		if err := CheckDuplicateGenOptions(opts); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("duplicate", func(t *testing.T) {
		opts := []genai.GenOption{
			genai.GenOptionSeed(1),
			genai.GenOptionSeed(2),
		}
		if err := CheckDuplicateGenOptions(opts); err == nil {
			t.Fatal("expected error for duplicate option")
		}
	})
	t.Run("empty", func(t *testing.T) {
		if err := CheckDuplicateGenOptions(nil); err != nil {
			t.Fatal(err)
		}
	})
}

func TestCheckDuplicateProviderOptions(t *testing.T) {
	t.Run("no_duplicates", func(t *testing.T) {
		opts := []genai.ProviderOption{
			genai.ProviderOptionAPIKey("key"),
			genai.ProviderOptionModel("model"),
		}
		if err := CheckDuplicateProviderOptions(opts); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("duplicate", func(t *testing.T) {
		opts := []genai.ProviderOption{
			genai.ProviderOptionModel("model1"),
			genai.ProviderOptionModel("model2"),
		}
		if err := CheckDuplicateProviderOptions(opts); err == nil {
			t.Fatal("expected error for duplicate option")
		}
	})
	t.Run("empty", func(t *testing.T) {
		if err := CheckDuplicateProviderOptions(nil); err != nil {
			t.Fatal(err)
		}
	})
}

func TestTimeSUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    TimeS
		wantErr bool
	}{
		{
			name:  "int64",
			input: `1234567890`,
			want:  TimeS(1234567890),
		},
		{
			name:  "float64 with fractional part",
			input: `1234567890.5`,
			want:  TimeS(1234567890.5),
		},
		{
			name:  "float64 with smaller fractional part",
			input: `1234567890.3`,
			want:  TimeS(1234567890.3),
		},
		{
			name:  "float64 with zero fractional part",
			input: `1234567890.0`,
			want:  TimeS(1234567890),
		},
		{
			name:    "invalid string",
			input:   `"not a number"`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ts TimeS
			err := json.Unmarshal([]byte(tt.input), &ts)
			if (err != nil) != tt.wantErr {
				t.Errorf("UnmarshalJSON() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !tt.wantErr && ts != tt.want {
				t.Errorf("UnmarshalJSON() = %v, want %v", ts, tt.want)
			}
		})
	}
}

func TestTimeSAsTime(t *testing.T) {
	tests := []struct {
		name string
		in   TimeS
		want time.Time
	}{
		{
			name: "integer seconds",
			in:   TimeS(1234567890),
			want: time.Unix(1234567890, 0).UTC(),
		},
		{
			name: "fractional seconds round to milliseconds",
			in:   TimeS(1234567890.1235),
			want: time.Unix(1234567890, 124*time.Millisecond.Nanoseconds()).UTC(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.in.AsTime(); got != tt.want {
				t.Errorf("AsTime() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTimeSIsZero(t *testing.T) {
	t.Run("zero", func(t *testing.T) {
		if !TimeS(0).IsZero() {
			t.Fatal("IsZero() = false, want true")
		}
	})
	t.Run("non_zero", func(t *testing.T) {
		if TimeS(1).IsZero() {
			t.Fatal("IsZero() = true, want false")
		}
	})
	t.Run("omitzero", func(t *testing.T) {
		type payload struct {
			CreatedAt TimeS `json:"createdAt,omitzero"`
		}
		got, err := json.Marshal(payload{})
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != `{}` {
			t.Fatalf("Marshal() = %s, want {}", got)
		}
	})
}

func TestTimeMSUnmarshalJSON(t *testing.T) {
	t.Run("valid", func(t *testing.T) {
		tests := []struct {
			name  string
			input string
			want  TimeMS
		}{
			{
				name:  "int64",
				input: `1234567890123`,
				want:  TimeMS(1234567890123),
			},
			{
				name:  "float64 with fractional part",
				input: `1234567890123.5`,
				want:  TimeMS(1234567890123.5),
			},
			{
				name:  "float64 with smaller fractional part",
				input: `1234567890123.3`,
				want:  TimeMS(1234567890123.3),
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				var ts TimeMS
				if err := json.Unmarshal([]byte(tt.input), &ts); err != nil {
					t.Fatal(err)
				}
				if ts != tt.want {
					t.Errorf("UnmarshalJSON() = %v, want %v", ts, tt.want)
				}
			})
		}
	})
	t.Run("error", func(t *testing.T) {
		var ts TimeMS
		if err := json.Unmarshal([]byte(`"not a number"`), &ts); err == nil {
			t.Fatal("expected error")
		}
	})
}

func TestTimeMSAsTime(t *testing.T) {
	tests := []struct {
		name string
		in   TimeMS
		want time.Time
	}{
		{
			name: "integer milliseconds",
			in:   TimeMS(1780832660165),
			want: time.Date(2026, 6, 7, 11, 44, 20, 165000000, time.UTC),
		},
		{
			name: "fractional milliseconds round to milliseconds",
			in:   TimeMS(1780832660165.5),
			want: time.Date(2026, 6, 7, 11, 44, 20, 166000000, time.UTC),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.in.AsTime(); got != tt.want {
				t.Errorf("AsTime() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestTimeMSIsZero(t *testing.T) {
	t.Run("zero", func(t *testing.T) {
		if !TimeMS(0).IsZero() {
			t.Fatal("IsZero() = false, want true")
		}
	})
	t.Run("non_zero", func(t *testing.T) {
		if TimeMS(1).IsZero() {
			t.Fatal("IsZero() = true, want false")
		}
	})
	t.Run("omitzero", func(t *testing.T) {
		type payload struct {
			StartedAt TimeMS `json:"startedAtMs,omitzero"`
		}
		got, err := json.Marshal(payload{})
		if err != nil {
			t.Fatal(err)
		}
		if string(got) != `{}` {
			t.Fatalf("Marshal() = %s, want {}", got)
		}
	})
}

func TestDurationMSAsDuration(t *testing.T) {
	tests := []struct {
		name string
		in   DurationMS
		want time.Duration
	}{
		{
			name: "integer milliseconds",
			in:   DurationMS(42),
			want: 42 * time.Millisecond,
		},
		{
			name: "fractional milliseconds",
			in:   DurationMS(599.3873493672435),
			want: 599*time.Millisecond + 387*time.Microsecond + 349*time.Nanosecond,
		},
		{
			name: "sub nanosecond truncates",
			in:   DurationMS(1.0000009),
			want: time.Millisecond,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.in.AsDuration(); got != tt.want {
				t.Errorf("AsDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDurationSAsDuration(t *testing.T) {
	tests := []struct {
		name string
		in   DurationS
		want time.Duration
	}{
		{
			name: "integer seconds",
			in:   DurationS(42),
			want: 42 * time.Second,
		},
		{
			name: "fractional seconds",
			in:   DurationS(2.9543220650000004),
			want: 2*time.Second + 954*time.Millisecond + 322*time.Microsecond + 65*time.Nanosecond,
		},
		{
			name: "sub nanosecond truncates",
			in:   DurationS(1.0000000009),
			want: time.Second,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.in.AsDuration(); got != tt.want {
				t.Errorf("AsDuration() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestUnknown(t *testing.T) {
	old := internal.BeLenient
	t.Cleanup(func() { internal.BeLenient = old })
	for _, lenient := range []bool{false, true} {
		t.Run(fmt.Sprintf("lenient=%t", lenient), func(t *testing.T) {
			internal.BeLenient = lenient
			for _, tc := range []struct {
				name  string
				value string
				empty bool
			}{
				{"null", "null", true},
				{"object", "{}", true},
				{"whitespace", " { \n\t } ", true},
				{"populated_object", `{"new":1}`, false},
				{"array", "[]", false},
				{"string", `""`, false},
				{"number", "0", false},
				{"boolean", "false", false},
			} {
				t.Run(tc.name, func(t *testing.T) {
					var u Unknown
					defer func() {
						v := recover()
						if want := !lenient && !tc.empty; (v != nil) != want {
							t.Fatalf("panic = %v, want panic %t", v, want)
						}
					}()
					b := []byte(tc.value)
					if err := json.Unmarshal(b, &u); err != nil {
						t.Fatal(err)
					}
					want := string(bytes.TrimSpace(b))
					if want == "null" {
						want = ""
					}
					b[0] = 'x'
					if string(u) != want {
						t.Fatalf("value = %s, want %s", u, want)
					}
					out, err := json.Marshal(u)
					if err != nil {
						t.Fatal(err)
					}
					var compact bytes.Buffer
					if err := json.Compact(&compact, []byte(tc.value)); err != nil {
						t.Fatal(err)
					}
					if string(out) != compact.String() {
						t.Fatalf("round trip = %s, want %s", out, compact.String())
					}
				})
			}
		})
	}
	t.Run("null_resets", func(t *testing.T) {
		for _, lenient := range []bool{false, true} {
			internal.BeLenient = lenient
			u := Unknown(`{"previous":true}`)
			if err := json.Unmarshal([]byte("null"), &u); err != nil {
				t.Fatal(err)
			}
			if u != nil {
				t.Fatalf("null = %s, want nil", u)
			}
			b, err := json.Marshal(struct {
				Value Unknown `json:"value,omitzero"`
			}{u})
			if err != nil || string(b) != "{}" {
				t.Fatalf("omitted null = %s, %v", b, err)
			}
		}
	})
	t.Run("marshal", func(t *testing.T) {
		b, err := json.Marshal(Unknown(nil))
		if err != nil || string(b) != "null" {
			t.Fatalf("nil = %s, %v", b, err)
		}
		if _, err := json.Marshal(Unknown("invalid")); err == nil {
			t.Fatal("expected invalid JSON error")
		}
		b, err = json.Marshal(struct {
			Value Unknown `json:"value,omitzero"`
		}{})
		if err != nil || string(b) != "{}" {
			t.Fatalf("omitted = %s, %v", b, err)
		}
	})
}

func TestNotImplemented(t *testing.T) {
	t.Run("Embed", func(t *testing.T) {
		p := NotImplemented{}
		if p.Capabilities().Embed {
			t.Fatal("fallback advertises embedding implementation")
		}
		for _, in := range []*genai.EmbeddingRequest{nil, {Inputs: []genai.Request{{Text: "hello"}}}} {
			out, err := p.Embed(t.Context(), in)
			if _, ok := errors.AsType[*ErrNotSupported](err); !ok || out != nil {
				t.Fatalf("got %+v, %v", out, err)
			}
		}
	})
}

// TestDecodeResponse preserves successful HTTP error envelopes and custom decode failures.
func TestDecodeResponse(t *testing.T) {
	for _, lenient := range []bool{false, true} {
		t.Run(strconv.FormatBool(lenient), func(t *testing.T) {
			c := ProviderBase[*decodeAPIError]{Lenient: lenient}
			c.lateInit()
			for _, tc := range []decodeResponseCase{{"custom vector error", `{"vector":"invalid"}`, false}, {"HTTP 200 API error", `{"error":"API failed"}`, true}} {
				t.Run(tc.name, func(t *testing.T) {
					var out decodeResponseOutput
					resp := &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(tc.body))}
					err := c.DecodeResponse(resp, "http://test", &out)
					if tc.api {
						er, ok := errors.AsType[*decodeAPIError](err)
						if !ok || er.Error() != "API failed" {
							t.Fatalf("API error: %v", err)
						}
					} else if err == nil || !strings.Contains(err.Error(), "invalid vector") {
						t.Fatalf("decode error: %v", err)
					}
				})
			}
		})
	}
}

type decodeResponseCase struct {
	name, body string
	api        bool
}
type decodeResponseOutput struct {
	Vector decodeFailVector `json:"vector"`
}
type decodeFailVector []float32

func (*decodeFailVector) UnmarshalJSON([]byte) error { return errors.New("invalid vector") }

type decodeAPIError struct {
	ErrorVal string `json:"error"`
}

func (e *decodeAPIError) Error() string  { return e.ErrorVal }
func (*decodeAPIError) IsAPIError() bool { return true }

func TestEmbeddingVector(t *testing.T) {
	t.Run("UnmarshalJSON", func(t *testing.T) {
		t.Run("error", func(t *testing.T) {
			for _, tc := range []embeddingVectorDecodeCase{{"invalid base64", `"!"`}, {"empty", `""`}, {"partial float", `"AA=="`}, {"null", `null`}, {"object", `{}`}, {"boolean", `true`}, {"invalid array element", `["bad"]`}} {
				t.Run(tc.name, func(t *testing.T) {
					v := EmbeddingVector{7}
					if err := json.Unmarshal([]byte(tc.body), &v); err == nil || v != nil {
						t.Fatalf("vector %+v, error %v", v, err)
					}
				})
			}
		})
		t.Run("valid", func(t *testing.T) {
			var v EmbeddingVector
			if err := json.Unmarshal([]byte(`"AACAPwAAQMA="`), &v); err != nil {
				t.Fatal(err)
			}
			runtime.GC()
			if !slices.Equal(v, EmbeddingVector{1, -3}) {
				t.Fatalf("values %+v", v)
			}
		})
	})
}

type embeddingVectorDecodeCase struct{ name, body string }
