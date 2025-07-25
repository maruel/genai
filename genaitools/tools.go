// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package genaitools contains ToolDef for popular tools.
package genaitools

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/maruel/genai"
)

// Arithmetic executes the arithmetic operation over two numbers.
//
// It first tries to do the calculation using int64, then using float64.
//
// The supported operations are "addition", "subtraction", "multiplication" and "division".
var Arithmetic = genai.ToolDef{
	Name:        "arithmetic",
	Description: "Calculates a mathematical arithmetic operation.",
	Callback: func(ctx context.Context, args *calculateArgs) (string, error) {
		if i1, err := args.FirstNumber.Int64(); err == nil {
			if i2, err := args.SecondNumber.Int64(); err == nil {
				switch args.Operation {
				case "addition":
					return strconv.FormatInt(i1+i2, 10), nil
				case "subtraction":
					return strconv.FormatInt(i1-i2, 10), nil
				case "multiplication":
					return strconv.FormatInt(i1*i2, 10), nil
				case "division":
					if i1%i2 == 0 {
						return strconv.FormatInt(i1/i2, 10), nil
					}
					// Otherwise fall back as float.
				default:
					return "", fmt.Errorf("unknown operation %q", args.Operation)
				}
			}
		}
		n1, err := args.FirstNumber.Float64()
		if err != nil {
			return "", fmt.Errorf("couldn't understand the first number: %w", err)
		}
		n2, err := args.SecondNumber.Float64()
		if err != nil {
			return "", fmt.Errorf("couldn't understand the second number: %w", err)
		}
		r := 0.
		switch args.Operation {
		case "addition":
			r = n1 + n2
		case "subtraction":
			r = n1 - n2
		case "multiplication":
			r = n1 * n2
		case "division":
			r = n1 / n2
		default:
			return "", fmt.Errorf("unknown operation %q", args.Operation)
		}
		// Do not use %g all the time because it tends to use exponents too quickly
		// and the LLM is super confused about that.
		// Do not use naive %f all the time because the LLM gets confused with
		// decimals.
		if r == math.Trunc(r) {
			return fmt.Sprintf("%.0f", r), nil
		}
		return fmt.Sprintf("%f", r), nil
	},
}

type calculateArgs struct {
	Operation    string      `json:"operation" jsonschema:"enum=addition,enum=subtraction,enum=multiplication,enum=division"`
	FirstNumber  json.Number `json:"first_number" jsonschema:"type=number"`
	SecondNumber json.Number `json:"second_number" jsonschema:"type=number"`
}

// GetTodayClockTime returns the current time and day in a format that the LLM
// can understand. It includes the weekend.
var GetTodayClockTime = genai.ToolDef{
	Name:        "get_today_date_current_clock_time",
	Description: "Get the current clock time and today's date.",
	Callback: func(ctx context.Context, e *empty) (string, error) {
		return time.Now().Format("Monday 2006-01-02 15:04:05"), nil
	},
}

type empty struct{}
