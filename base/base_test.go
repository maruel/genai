// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for the base package.

package base

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/maruel/genai"
)

func TestCheckDuplicateOptions(t *testing.T) {
	t.Run("no_duplicates", func(t *testing.T) {
		opts := []genai.ProviderOption{
			genai.ProviderOptionAPIKey("key"),
			genai.ProviderOptionModel("model"),
		}
		if err := CheckDuplicateOptions(opts); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("duplicate", func(t *testing.T) {
		opts := []genai.ProviderOption{
			genai.ProviderOptionModel("model1"),
			genai.ProviderOptionModel("model2"),
		}
		if err := CheckDuplicateOptions(opts); err == nil {
			t.Fatal("expected error for duplicate option")
		}
	})
	t.Run("empty", func(t *testing.T) {
		if err := CheckDuplicateOptions(nil); err != nil {
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
