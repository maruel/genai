// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package openaibase contains shared types and client operations used by both
// the OpenAI Chat Completion and Responses API providers.
//
// These types map directly to the JSON objects exchanged with the OpenAI
// platform API endpoints that are shared between the two APIs (models, files,
// images, batches, errors).
//
// Source: https://platform.openai.com/docs/api-reference/
package openaibase

import (
	"errors"
	"fmt"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// ServiceTier is the quality of service to determine the request's priority.
type ServiceTier string

const (
	// ServiceTierAuto will utilize scale tier credits until they are exhausted if the Project is Scale tier
	// enabled, else the request will be processed using the default service tier with a lower uptime SLA and no
	// latency guarantee.
	//
	// https://openai.com/api-scale-tier/
	ServiceTierAuto ServiceTier = "auto"
	// ServiceTierDefault has the request be processed using the default service tier with a lower uptime SLA
	// and no latency guarantee.
	ServiceTierDefault ServiceTier = "default"
	// ServiceTierFlex has the request be processed with the Flex Processing service tier.
	//
	// Flex processing is in beta, and currently only available for GPT-5, o3 and o4-mini models.
	//
	// https://platform.openai.com/docs/guides/flex-processing
	ServiceTierFlex ServiceTier = "flex"
)

// Validate implements genai.Validatable.
func (s ServiceTier) Validate() error {
	switch s {
	case "", ServiceTierAuto, ServiceTierDefault, ServiceTierFlex:
		return nil
	default:
		return fmt.Errorf("invalid service tier %q", s)
	}
}

// ReasoningEffort is the effort the model should put into reasoning. Default is Medium.
//
// https://platform.openai.com/docs/api-reference/assistants/createAssistant#assistants-createassistant-reasoning_effort
// https://platform.openai.com/docs/guides/reasoning
type ReasoningEffort string

// Reasoning effort values.
const (
	ReasoningEffortNone    ReasoningEffort = "none"
	ReasoningEffortMinimal ReasoningEffort = "minimal"
	ReasoningEffortLow     ReasoningEffort = "low"
	ReasoningEffortMedium  ReasoningEffort = "medium"
	ReasoningEffortHigh    ReasoningEffort = "high"
	ReasoningEffortXHigh   ReasoningEffort = "xhigh"
)

// Validate implements genai.Validatable.
func (r ReasoningEffort) Validate() error {
	switch r {
	case "", ReasoningEffortNone, ReasoningEffortMinimal, ReasoningEffortLow, ReasoningEffortMedium, ReasoningEffortHigh, ReasoningEffortXHigh:
		return nil
	default:
		return fmt.Errorf("invalid reasoning effort %q", r)
	}
}

// Background is only supported on gpt-image-1.
type Background string

// Background mode values.
const (
	BackgroundAuto        Background = "auto"
	BackgroundTransparent Background = "transparent"
	BackgroundOpaque      Background = "opaque"
)

// Validate implements genai.Validatable.
func (b Background) Validate() error {
	switch b {
	case "", BackgroundAuto, BackgroundTransparent, BackgroundOpaque:
		return nil
	default:
		return fmt.Errorf("invalid background %q", b)
	}
}

// ImageRequest is documented at https://platform.openai.com/docs/api-reference/images
type ImageRequest struct {
	Prompt            string     `json:"prompt"`
	Model             string     `json:"model,omitzero"`              // Default to dall-e-2, unless a gpt-image-1 specific parameter is used.
	Background        Background `json:"background,omitzero"`         // Default "auto"
	Moderation        string     `json:"moderation,omitzero"`         // gpt-image-1: "low" or "auto"
	N                 int64      `json:"n,omitzero"`                  // Number of images to return
	OutputCompression float64    `json:"output_compression,omitzero"` // Defaults to 100. Only supported on gpt-image-1 with webp or jpeg
	OutputFormat      string     `json:"output_format,omitzero"`      // "png", "jpeg" or "webp". Defaults to png. Only supported on gpt-image-1.
	Quality           string     `json:"quality,omitzero"`            // "auto", gpt-image-1: "high", "medium", "low". dall-e-3: "hd", "standard". dall-e-2: "standard".
	ResponseFormat    string     `json:"response_format,omitzero"`    // "url" or "b64_json"; url is valid for 60 minutes; gpt-image-1 only returns b64_json
	Size              string     `json:"size,omitzero"`               // "auto", gpt-image-1: "1024x1024", "1536x1024", "1024x1536". dall-e-3: "1024x1024", "1792x1024", "1024x1792". dall-e-2: "256x256", "512x512", "1024x1024".
	Style             string     `json:"style,omitzero"`              // dall-e-3: "vivid", "natural"
	User              string     `json:"user,omitzero"`               // End-user to help monitor and detect abuse
}

// Init initializes the request from the given parameters.
func (i *ImageRequest) Init(msg *genai.Message, model string, opts ...genai.GenOption) error {
	if err := msg.Validate(); err != nil {
		return err
	}
	for i := range msg.Requests {
		if msg.Requests[i].Text == "" {
			return errors.New("only text can be passed as input")
		}
	}
	i.Prompt = msg.String()
	i.Model = model

	// This is unfortunate.
	switch model {
	case "gpt-image-1":
		i.Moderation = "low"
		// Other supported options: Background, OutputFormat, OutputCompression, Quality, Size.
	case "dall-e-3":
		// Other supported options: Size (e.g. 1792x1024).
		i.ResponseFormat = "b64_json"
	case "dall-e-2":
		// We assume dall-e-2 is only used for smoke testing, so use the smallest image.
		i.Size = "256x256"
		// Maximum prompt length is 1000 characters.
		// Since we assume this is only for testing, silently cut it off.
		if len(i.Prompt) > 1000 {
			i.Prompt = i.Prompt[:1000]
		}
		i.ResponseFormat = "b64_json"
	default:
		// Silently pass.
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *GenOptionImage:
			i.Background = v.Background
		case *genai.GenOptionImage:
			if v.Height != 0 && v.Width != 0 {
				i.Size = fmt.Sprintf("%dx%d", v.Width, v.Height)
			}
		default:
			return &base.ErrNotSupported{Options: []string{internal.TypeName(opt)}}
		}
	}
	return nil
}

// ImageResponse is the provider-specific image generation response.
type ImageResponse struct {
	Created base.TimeS        `json:"created"`
	Data    []ImageChoiceData `json:"data"`
	Usage   struct {
		InputTokens        int64 `json:"input_tokens"`
		OutputTokens       int64 `json:"output_tokens"`
		TotalTokens        int64 `json:"total_tokens"`
		InputTokensDetails struct {
			TextTokens  int64 `json:"text_tokens"`
			ImageTokens int64 `json:"image_tokens"`
		} `json:"input_tokens_details"`
		OutputTokensDetails struct {
			TextTokens  int64 `json:"text_tokens"`
			ImageTokens int64 `json:"image_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
	Background   string `json:"background"`    // "opaque"
	Size         string `json:"size"`          // e.g. "1024x1024"
	Quality      string `json:"quality"`       // e.g. "medium"
	OutputFormat string `json:"output_format"` // e.g. "png"
}

// ImageChoiceData is the data for one image generation choice.
type ImageChoiceData struct {
	B64JSON       []byte `json:"b64_json"`
	RevisedPrompt string `json:"revised_prompt"` // dall-e-3 only
	URL           string `json:"url"`            // Unsupported for gpt-image-1
}

// GenOptionImage defines OpenAI specific options.
type GenOptionImage struct {
	// Background is only supported on gpt-image-1.
	Background Background
}

// Validate implements genai.Validatable.
func (o *GenOptionImage) Validate() error {
	return o.Background.Validate()
}

// Model is documented at https://platform.openai.com/docs/api-reference/models/object
//
// Sadly the modalities aren't reported. The only way I can think of to find it at run time is to fetch
// https://platform.openai.com/docs/models/gpt-4o-mini-realtime-preview, find the div containing
// "Modalities:", then extract the modalities from the text.
type Model struct {
	ID      string     `json:"id"`
	Object  string     `json:"object"`
	Created base.TimeS `json:"created"`
	OwnedBy string     `json:"owned_by"`
}

// GetID implements genai.Model.
func (m *Model) GetID() string {
	return m.ID
}

func (m *Model) String() string {
	return fmt.Sprintf("%s (%s)", m.ID, m.Created.AsTime().Format("2006-01-02"))
}

// Context implements genai.Model.
func (m *Model) Context() int64 {
	return 0
}

// ModelsResponse represents the response structure for OpenAI models listing.
type ModelsResponse struct {
	Object string  `json:"object"` // list
	Data   []Model `json:"data"`
}

// ToModels converts OpenAI models to genai.Model interfaces.
func (r *ModelsResponse) ToModels() []genai.Model {
	models := make([]genai.Model, len(r.Data))
	for i := range r.Data {
		models[i] = &r.Data[i]
	}
	return models
}

// File is documented at https://platform.openai.com/docs/api-reference/files/object
type File struct {
	Bytes         int64      `json:"bytes"` // File size
	CreatedAt     base.TimeS `json:"created_at"`
	ExpiresAt     base.TimeS `json:"expires_at"`
	Filename      string     `json:"filename"`
	ID            string     `json:"id"`
	Object        string     `json:"object"`         // "file"
	Purpose       string     `json:"purpose"`        // One of: assistants, assistants_output, batch, batch_output, fine-tune, fine-tune-results and vision
	Status        string     `json:"status"`         // Deprecated
	StatusDetails string     `json:"status_details"` // Deprecated
}

// GetID implements genai.Model.
func (f *File) GetID() string {
	return f.ID
}

// GetDisplayName implements genai.CacheItem.
func (f *File) GetDisplayName() string {
	return f.Filename
}

// GetExpiry implements genai.CacheItem.
func (f *File) GetExpiry() time.Time {
	return f.ExpiresAt.AsTime()
}

// FileDeleteResponse is documented at https://platform.openai.com/docs/api-reference/files/delete
type FileDeleteResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"` // "file"
	Deleted bool   `json:"deleted"`
}

// FileListResponse is documented at https://platform.openai.com/docs/api-reference/files/list
type FileListResponse struct {
	Data   []File `json:"data"`
	Object string `json:"object"` // "list"
}

// BatchRequest is documented at https://platform.openai.com/docs/api-reference/batch/create
type BatchRequest struct {
	CompletionWindow string            `json:"completion_window"` // Must be "24h"
	Endpoint         string            `json:"endpoint"`          // One of /v1/responses, /v1/chat/completions, /v1/embeddings, /v1/completions
	InputFileID      string            `json:"input_file_id"`     // File must be JSONL
	Metadata         map[string]string `json:"metadata,omitzero"` // Maximum 16 keys of 64 chars, values max 512 chars
}

// Batch is documented at https://platform.openai.com/docs/api-reference/batch/object
type Batch struct {
	CancelledAt      base.TimeS `json:"cancelled_at"`
	CancellingAt     base.TimeS `json:"cancelling_at"`
	CompletedAt      base.TimeS `json:"completed_at"`
	CompletionWindow string     `json:"completion_window"` // "24h"
	CreatedAt        base.TimeS `json:"created_at"`
	Endpoint         string     `json:"endpoint"`      // Same as BatchRequest.Endpoint
	ErrorFileID      string     `json:"error_file_id"` // File ID containing the outputs of requests with errors.
	Errors           struct {
		Data []struct {
			Code    string `json:"code"`
			Line    int64  `json:"line"`
			Message string `json:"message"`
			Param   string `json:"param"`
		} `json:"data"`
	} `json:"errors"`
	ExpiredAt     base.TimeS        `json:"expired_at"`
	ExpiresAt     base.TimeS        `json:"expires_at"`
	FailedAt      base.TimeS        `json:"failed_at"`
	FinalizingAt  base.TimeS        `json:"finalizing_at"`
	ID            string            `json:"id"`
	InProgressAt  base.TimeS        `json:"in_progress_at"`
	InputFileID   string            `json:"input_file_id"` // Input data
	Metadata      map[string]string `json:"metadata"`
	Model         string            `json:"model,omitzero"`
	Object        string            `json:"object"`         // "batch"
	OutputFileID  string            `json:"output_file_id"` // Output data
	RequestCounts struct {
		Completed int64 `json:"completed"`
		Failed    int64 `json:"failed"`
		Total     int64 `json:"total"`
	} `json:"request_counts"`
	Status string     `json:"status"`         // "completed", "in_progress", "validating", "finalizing"
	Usage  BatchUsage `json:"usage,omitzero"` // Token usage for the batch
}

// BatchUsage represents token usage information for a batch.
type BatchUsage struct {
	InputTokens        int64 `json:"input_tokens"`
	OutputTokens       int64 `json:"output_tokens"`
	TotalTokens        int64 `json:"total_tokens"`
	InputTokensDetails struct {
		CachedTokens int64 `json:"cached_tokens"`
	} `json:"input_tokens_details"`
	OutputTokensDetails struct {
		ReasoningTokens int64 `json:"reasoning_tokens"`
	} `json:"output_tokens_details"`
}

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
	ErrorVal ErrorResponseError `json:"error"`
}

func (er *ErrorResponse) Error() string {
	out := ""
	if er.ErrorVal.Type != "" {
		out += er.ErrorVal.Type
	}
	if er.ErrorVal.Code != "" {
		if out != "" {
			out += "/"
		}
		out += er.ErrorVal.Code
	}
	if er.ErrorVal.Status != "" {
		out += fmt.Sprintf("(%s)", er.ErrorVal.Status)
	}
	if er.ErrorVal.Param != "" {
		if out != "" {
			out += " "
		}
		out += fmt.Sprintf("for %q", er.ErrorVal.Param)
	}
	if er.ErrorVal.Message != "" {
		if out != "" {
			out += ": "
		}
		out += er.ErrorVal.Message
	}
	return out
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// ErrorResponseError is the nested error in an error response.
type ErrorResponseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Status  string `json:"status"`
	Type    string `json:"type"`
	Param   string `json:"param"`
}
