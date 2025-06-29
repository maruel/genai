// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Command list-models fetches and prints out the list of models from the selected providers.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"reflect"
	"slices"
	"sort"
	"strings"
	"syscall"

	"github.com/maruel/genai"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/providers"
	"github.com/maruel/genai/providers/huggingface"
)

func printStructDense(v any, indent string) string {
	val := reflect.ValueOf(v)
	if val.Kind() == reflect.Ptr {
		if val.IsNil() {
			return indent + "nil"
		}
		val = val.Elem()
	}
	if val.Kind() != reflect.Struct {
		return indent + fmt.Sprintf("%v", v)
	}
	typ := val.Type()
	var fields []string
	// Iterate through struct fields
	for i := range val.NumField() {
		f := val.Field(i)
		fn := typ.Field(i).Name
		switch f.Kind() {
		case reflect.Struct:
			// Recursively print nested structs
			v := printStructDense(f.Interface(), indent+"  ")
			fields = append(fields, fmt.Sprintf("%s%s: {\n%s\n}", indent, fn, v))
		case reflect.Ptr:
			if f.IsNil() {
				fields = append(fields, fmt.Sprintf("%s%s: nil", indent, fn))
			} else {
				// Recursively handle pointers
				v := printStructDense(f.Interface(), indent+"  ")
				fields = append(fields, fmt.Sprintf("%s%s: &{\n%s\n}", indent, fn, v))
			}
		case reflect.Slice, reflect.Array:
			if f.Len() == 0 {
				fields = append(fields, fmt.Sprintf("%s%s: []", indent, fn))
			} else {
				var elements []string
				for j := range f.Len() {
					elem := f.Index(j)
					if elem.Kind() == reflect.Struct || elem.Kind() == reflect.Ptr {
						elements = append(elements, printStructDense(elem.Interface(), indent+"  "))
					} else {
						elements = append(elements, fmt.Sprintf("%v", elem.Interface()))
					}
				}
				fields = append(fields, fmt.Sprintf("%s%s: [%s]", indent, fn, strings.Join(elements, ",")))
			}
		default:
			fields = append(fields, fmt.Sprintf("%s%s: %v", indent, fn, f.Interface()))
		}
	}
	return strings.Join(fields, "\n")
}

func getProvidersModel() []string {
	var names []string
	for name, f := range providers.All {
		if c, _ := f("", nil); c != nil {
			if _, ok := c.(genai.ProviderModel); ok {
				names = append(names, name)
			}
		}
	}
	sort.Strings(names)
	return names
}

func mainImpl() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()

	names := getProvidersModel()
	provider := flag.String("provider", "", "backend to use: "+strings.Join(names, ", "))
	all := flag.Bool("all", false, "include all details")
	strict := flag.Bool("strict", false, "assert no unknown fields in the APIs are found")
	flag.Parse()
	if flag.NArg() != 0 {
		return errors.New("unexpected arguments")
	}
	if *strict {
		internal.BeLenient = false
	}
	if *provider == "" {
		return errors.New("-provider is required")
	}
	if !slices.Contains(names, *provider) {
		return fmt.Errorf("unknown backend %q", *provider)
	}
	c, err := providers.All[*provider]("", nil)
	if err != nil {
		return err
	}
	models, err := c.(genai.ProviderModel).ListModels(ctx)
	if err != nil {
		return err
	}
	m := make(map[string]genai.Model, len(models))
	s := make([]string, 0, len(models))
	for _, model := range models {
		if t, ok := model.(*huggingface.Model); ok {
			if t.TrendingScore < 1 {
				continue
			}
		}
		name := model.String()
		s = append(s, name)
		m[name] = model
	}
	sort.Slice(s, func(i, j int) bool {
		return strings.ToLower(s[i]) < strings.ToLower(s[j])
	})
	for _, name := range s {
		fmt.Printf("%s\n", name)
		if *all {
			_, _ = os.Stdout.WriteString(printStructDense(m[name], "  ") + "\n")
		}
	}
	return nil
}

func main() {
	if err := mainImpl(); err != nil {
		if err != context.Canceled {
			fmt.Fprintf(os.Stderr, "list-models: %s\n", err)
		}
		os.Exit(1)
	}
}
