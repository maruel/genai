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
	"sort"
	"strings"
	"syscall"

	"github.com/maruel/genai"
	"github.com/maruel/genai/anthropic"
	"github.com/maruel/genai/cerebras"
	"github.com/maruel/genai/cloudflare"
	"github.com/maruel/genai/cohere"
	"github.com/maruel/genai/deepseek"
	"github.com/maruel/genai/gemini"
	"github.com/maruel/genai/groq"
	"github.com/maruel/genai/huggingface"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/mistral"
	"github.com/maruel/genai/openai"
	"github.com/maruel/genai/pollinations"
	"github.com/maruel/genai/togetherai"
)

var providers = map[string]func() (genai.ProviderModel, error){
	"anthropic": func() (genai.ProviderModel, error) {
		return anthropic.New("", "", nil)
	},
	"cerebras": func() (genai.ProviderModel, error) {
		return cerebras.New("", "", nil)
	},
	"cloudflare": func() (genai.ProviderModel, error) {
		return cloudflare.New("", "", "", nil)
	},
	"cohere": func() (genai.ProviderModel, error) {
		return cohere.New("", "", nil)
	},
	"deepseek": func() (genai.ProviderModel, error) {
		return deepseek.New("", "", nil)
	},
	"gemini": func() (genai.ProviderModel, error) {
		return gemini.New("", "", nil)
	},
	"groq": func() (genai.ProviderModel, error) {
		return groq.New("", "", nil)
	},
	"huggingface": func() (genai.ProviderModel, error) {
		return huggingface.New("", "", nil)
	},
	"mistral": func() (genai.ProviderModel, error) {
		return mistral.New("", "", nil)
	},
	"openai": func() (genai.ProviderModel, error) {
		return openai.New("", "", nil)
	},
	"pollinations": func() (genai.ProviderModel, error) {
		return pollinations.New("", "", nil)
	},
	"togetherai": func() (genai.ProviderModel, error) {
		return togetherai.New("", "", nil)
	},
}

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

func mainImpl() error {
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM, os.Interrupt)
	defer stop()

	names := make([]string, 0, len(providers))
	for name := range providers {
		names = append(names, name)
	}
	sort.Strings(names)
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
	fn := providers[*provider]
	if fn == nil {
		return fmt.Errorf("unknown backend %q", *provider)
	}
	b, err := fn()
	if err != nil {
		return err
	}
	models, err := b.ListModels(ctx)
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
