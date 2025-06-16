// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openai implements a client for the OpenAI API.
//
// It is described at https://platform.openai.com/docs/api-reference/
package openai

import "github.com/maruel/genai/providers/openai/openaichat"

var New = openaichat.New

type Client = openaichat.Client
