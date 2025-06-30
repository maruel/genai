// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package scoreboard creates a scoreboard based on a provider instance.
package scoreboard

import (
	"errors"

	"github.com/maruel/genai"
)

// CreateScenario calculates the supported Scenario for the given provider and its current model.
func CreateScenario(c genai.Provider) (genai.Scenario, error) {
	m := c.ModelID()
	if m == "" {
		return genai.Scenario{}, errors.New("provider must have a model")
	}
	out := genai.Scenario{
		Models: []string{m},
		In:     map[genai.Modality]genai.ModalCapability{},
		Out:    map[genai.Modality]genai.ModalCapability{},
	}
	if p, ok := c.(genai.ProviderGen); ok {
		if err := exerciseGen(p, &out); err != nil {
			return out, err
		}
	}
	if p, ok := c.(genai.ProviderGenDoc); ok {
		if err := exerciseGenDoc(p, &out); err != nil {
			return out, err
		}
	}
	return out, nil
}

func exerciseGen(c genai.ProviderGen, out *genai.Scenario) error {
	return errors.New("implement me")
}

func exerciseGenDoc(c genai.ProviderGenDoc, out *genai.Scenario) error {
	return errors.New("implement me")
}
