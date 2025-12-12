// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package base

import (
	"encoding/json"
	"testing"
	"time"
)

func TestTimeUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    Time
		wantErr bool
	}{
		{
			name:  "int64",
			input: `1234567890`,
			want:  Time(1234567890),
		},
		{
			name:  "float64 with fractional part >= 0.5",
			input: `1234567890.5`,
			want:  Time(1234567891),
		},
		{
			name:  "float64 with fractional part < 0.5",
			input: `1234567890.3`,
			want:  Time(1234567890),
		},
		{
			name:  "float64 with zero fractional part",
			input: `1234567890.0`,
			want:  Time(1234567890),
		},
		{
			name:    "invalid string",
			input:   `"not a number"`,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var ts Time
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

func TestTimeAsTime(t *testing.T) {
	ts := Time(1234567890)
	got := ts.AsTime()
	want := time.Unix(1234567890, 0).UTC()
	if got != want {
		t.Errorf("AsTime() = %v, want %v", got, want)
	}
}
