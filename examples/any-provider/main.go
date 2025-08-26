// Let the user chose the provider by name.
//
// The relevant environment variable (e.g. `ANTHROPIC_API_KEY`,
// `OPENAI_API_KEY`, etc) is used automatically for authentication.
//
// Automatically selects a models on behalf of the user.
//
// Supports Ollama (https://ollama.com/) and llama-server (https://github.com/ggml-org/llama.cpp)
// even if they run on a remote host or non-default port.

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"maps"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/adapters"
	"github.com/maruel/genai/providers"
)

func main() {
	ctx := context.Background()
	s := strings.Join(slices.Sorted(maps.Keys(providers.Available(ctx))), ", ")
	if s == "" {
		s = "set environment variables, e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc"
	}
	provider := flag.String("provider", "", "provider to use, "+s)
	model := flag.String("model", "", "model to use; "+genai.ModelCheap+", "+genai.ModelGood+" (default) or "+genai.ModelSOTA+" for automatic model selection")
	remote := flag.String("remote", "", "url to use, e.g. when using ollama or llama-server on another host")
	flag.Parse()

	query := strings.Join(flag.Args(), " ")
	if query == "" {
		log.Fatal("provide a query")
	}
	p, err := LoadProvider(ctx, *provider, &genai.ProviderOptions{Model: *model, Remote: *remote})
	if err != nil {
		log.Fatal(err)
	}
	resp, err := p.GenSync(ctx, genai.Messages{genai.NewTextMessage(query)})
	if err != nil {
		log.Fatalf("failed to use provider %q: %s", *provider, err)
	}
	fmt.Printf("%s\n", resp.String())
}

// LoadProvider loads a provider.
func LoadProvider(ctx context.Context, provider string, opts *genai.ProviderOptions) (genai.Provider, error) {
	if provider == "" {
		return nil, fmt.Errorf("no provider specified")
	}
	f := providers.All[provider]
	if f == nil {
		return nil, fmt.Errorf("unknown provider %q", provider)
	}
	c, err := f(ctx, opts, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to provider %q: %w", provider, err)
	}
	// Wrap the provider with an adapter to process "<think>" tokens automatically ONLY if needed.
	return adapters.WrapThinking(c), nil
}
