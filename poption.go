// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

package genai

import (
	"errors"
	"net/http"
)

// ProviderOption is an option for provider constructors.
type ProviderOption interface {
	Validate() error
}

// ProviderOptionAPIKey provides an API key to authenticate to the server.
//
// Most providers require an API key, and the client will look at an environment variable
// "<PROVIDER>_API_KEY" to use as a default value if unspecified.
type ProviderOptionAPIKey string

// Validate implements Validatable.
func (p ProviderOptionAPIKey) Validate() error {
	if p == "" {
		return errors.New("ProviderOptionAPIKey cannot be empty")
	}
	return nil
}

// ProviderOptionRemote is the remote address to access the service.
//
// It is mostly used by locally hosted services (llamacpp, ollama) or for generic client (openaicompatible).
type ProviderOptionRemote string

// Validate implements Validatable.
func (p ProviderOptionRemote) Validate() error {
	if p == "" {
		return errors.New("ProviderOptionRemote cannot be empty")
	}
	return nil
}

// ProviderOptionModel specifies which model to use.
//
// For automatic model selection, use the predefined constants ModelCheap, ModelGood, or ModelSOTA
// directly as provider options. The provider internally calls ListModels() to discover models
// and select based on its heuristics. Providers that do not support ListModels (e.g. bfl or
// perplexity) use a hardcoded list.
//
// When unspecified, no model is selected. This is useful when only calling ListModels().
// Generation calls will fail.
//
// Keep in mind that as providers cycle through new models, it's possible a specific model ID is not
// available anymore or that the default model changes.
type ProviderOptionModel string

// Validate implements Validatable.
func (p ProviderOptionModel) Validate() error {
	if p == "" {
		return errors.New("ProviderOptionModel cannot be empty")
	}
	return nil
}

// ProviderOptionModalities is the list of output modalities you request the model to support.
//
// Most provider support text output only. Most models support output of only one modality, either text,
// image, audio or video. But a few models do support both text and images.
//
// When unspecified, the provider defaults to all modalities supported by the provider and the selected model.
//
// Even when Model is set to a specific model ID, a ListModels call may be made to discover its supported
// output modalities for providers that support multiple output modalities.
//
// ProviderOptionModalities can be set without a Model to test if a provider supports a modality without
// causing a ListModels call.
type ProviderOptionModalities Modalities

// Validate implements Validatable.
func (p ProviderOptionModalities) Validate() error {
	if len(p) == 0 {
		return errors.New("ProviderOptionModalities cannot be empty")
	}
	return Modalities(p).Validate()
}

// ProviderOptionPreloadedModels is a list of models that are preloaded into the provider, to replace the call to
// ListModels, for example with automatic model selection and modality detection.
//
// This is mostly used for unit tests or repeated client creation to save on HTTP requests.
type ProviderOptionPreloadedModels []Model

// Validate implements Validatable.
func (p ProviderOptionPreloadedModels) Validate() error {
	if len(p) == 0 {
		return errors.New("ProviderOptionPreloadedModels cannot be empty")
	}
	return nil
}

// ProviderOptionTransportWrapper wraps the HTTP transport used by the provider.
//
// This is useful for adding middleware like logging, tracing, or HTTP recording for tests.
type ProviderOptionTransportWrapper func(http.RoundTripper) http.RoundTripper

// Validate implements Validatable.
func (p ProviderOptionTransportWrapper) Validate() error {
	if p == nil {
		return errors.New("ProviderOptionTransportWrapper cannot be nil")
	}
	return nil
}

// Model markers for automatic model selection. These are ProviderOptionModel values
// that can be passed directly to provider constructors.
const (
	// ModelCheap requests the provider to automatically select the cheapest model it can find.
	ModelCheap ProviderOptionModel = "CHEAP"
	// ModelGood requests the provider to automatically select a good every day model that has a good
	// performance/cost trade-off.
	ModelGood ProviderOptionModel = "GOOD"
	// ModelSOTA requests the provider to automatically select the best state-of-the-art model
	// it can find.
	ModelSOTA ProviderOptionModel = "SOTA"
)
