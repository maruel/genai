// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Wire types for the Black Forest Labs REST API.
//
// Documentation: https://docs.bfl.ai/api-reference

package bfl

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"strings"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
)

// ImageRequest is described in https://docs.bfl.ml/api-reference/tasks/generate-an-image-with-flux-11-[pro]
// and the likes.
type ImageRequest struct {
	Prompt           string `json:"prompt"`
	ImagePrompt      []byte `json:"image_prompt,omitzero"`
	Width            int64  `json:"width,omitzero"`             // Default 1024, [256, 1440], multiple of 32.
	Height           int64  `json:"height,omitzero"`            // Default 768, [256, 1440], multiple of 32.
	PromptUpsampling bool   `json:"prompt_upsampling,omitzero"` //
	Seed             int64  `json:"seed,omitzero"`              //
	SafetyTolerance  int64  `json:"safety_tolerance"`           // [0, 6], 0 being strict.
	OutputFormat     string `json:"output_format,omitzero"`     // Default: "jpeg"; "png"
	WebhookURL       string `json:"webhook_url,omitzero"`       // Receive webhook notifications; max 2083 characters
	WebhookSecret    string `json:"webhook_secret,omitzero"`    // Optional secret for signature verification

	// FLUX.1 [pro], [dev]
	Steps    int64   `json:"steps,omitzero"`    // Default 28 for dev, 40 for pro, [1, 50]
	Guidance float64 `json:"guidance,omitzero"` // [1.5, 5]

	// FLUX.1 [pro]
	Interval float64 `json:"interval,omitzero"` // [1, 4]

	// FLUX.1.1 [pro] ultra
	AspectRatio string `json:"aspect_ratio,omitzero"` // Default: "16:9", between "21:9" and "9:21"
	Raw         bool   `json:"raw,omitzero"`          // Less processed, more natural-looking images

	// FLUX.1 [pro] fill
	Mask []byte `json:"mask,omitzero"` // Areas to modify the image. Black no modification.

	// FLUX.1 [pro] expand
	Top    int64 `json:"top,omitzero"`    // [0, 2048]
	Bottom int64 `json:"bottom,omitzero"` // [0, 2048]
	Left   int64 `json:"left,omitzero"`   // [0, 2048]
	Right  int64 `json:"right,omitzero"`  // [0, 2048]

	// FLUX.1 [pro] canny (control image)
	CannyLowThreshold  int64 `json:"canny_low_threshold,omitzero"`  // [0, 500] for edge detection; default 50; TODO: 0
	CannyHighThreshold int64 `json:"canny_high_threshold,omitzero"` // [0, 500] for edge detection; default 200; TODO: 0

	// FLUX.1 [pro] canny, depth
	ControlImage      []byte `json:"control_image,omitzero"`      // One of the two
	PreprocessedImage []byte `json:"preprocessed_image,omitzero"` //

	// FLUX Kontext Dev, Pro, Max
	InputImage  string `json:"input_image,omitzero"`
	InputImage2 string `json:"input_image2,omitzero"`
	InputImage3 string `json:"input_image3,omitzero"`
	InputImage4 string `json:"input_image4,omitzero"`
}

// Init initializes the request from the given parameters.
func (i *ImageRequest) Init(msgs genai.Messages, model string, opts ...genai.GenOption) error {
	if err := msgs.Validate(); err != nil {
		return err
	}
	if len(msgs) != 1 {
		return errors.New("must pass exactly one Message")
	}
	msg := msgs[0]
	msg.Requests = slices.Clone(msg.Requests)
	for _, r := range msg.Requests {
		if r.Text != "" {
			continue
		}
		// Check if this is a text/plain document that we can handle
		if !r.Doc.IsZero() {
			mimeType, data, err := r.Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return fmt.Errorf("failed to read document: %w", err)
			}
			switch {
			// text/plain, text/markdown
			case strings.HasPrefix(mimeType, "text/"):
				if r.Doc.URL != "" {
					return fmt.Errorf("%s documents must be provided inline, not as a URL", mimeType)
				}
				// Convert document content to text content.
				r.Text = string(data)
				r.Doc = genai.Doc{}
			case strings.HasPrefix(mimeType, "image/"):
				// Used in Kontext
				s := r.Doc.URL
				if s == "" {
					s = fmt.Sprintf("data:%s;base64,%s", mimeType, base64.StdEncoding.EncodeToString(data))
				}
				switch {
				case i.InputImage == "":
					i.InputImage = s
				case i.InputImage2 == "":
					i.InputImage2 = s
				case i.InputImage3 == "":
					i.InputImage3 = s
				case i.InputImage4 == "":
					i.InputImage4 = s
				default:
					return errors.New("too many images")
				}
				r.Doc = genai.Doc{}
			default:
				return errors.New("only text and text documents can be passed as input")
			}
			continue
		}
		return errors.New("unknown Request type")
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return err
		}
		switch v := opt.(type) {
		case *genai.GenOptionImage:
			i.Height = int64(v.Height)
			i.Width = int64(v.Width)
		case genai.GenOptionSeed:
			i.Seed = int64(v)
		default:
			return &base.ErrNotSupported{Options: []string{internal.TypeName(opt)}}
		}
	}

	i.Prompt = msg.String()
	return nil
}

// ImageRequestResponse is the same for all requests since everything is asynchronous.
//
// It's in https://docs.bfl.ml/api-reference/tasks/generate-an-image-with-flux-11-[pro] and the likes.
type ImageRequestResponse struct {
	ID string `json:"id"`
	// PollingURL is the URL to poll for the result. Generated images expire after 10 minutes and become
	// inaccessible.
	PollingURL string `json:"polling_url"`
	// Cost, InputMP, and OutputMP are optional fields returned by the API.
	Cost     json.RawMessage `json:"cost,omitempty"`
	InputMP  json.RawMessage `json:"input_mp,omitempty"`
	OutputMP json.RawMessage `json:"output_mp,omitempty"`
}

// ImageResult is the provider-specific image generation result.
type ImageResult struct {
	ID     string `json:"id"`
	Status string `json:"status"` // "ready", "Pending"
	Result struct {
		// Sample is the result. The signed URL are only valid for 10 minutes. This means the user has to call get
		// result again to get the result after 10 minutes.
		Sample    string  `json:"sample"`
		Prompt    string  `json:"prompt"`
		Seed      int64   `json:"seed"`
		StartTime float64 `json:"start_time"`
		EndTime   float64 `json:"end_time"`
		Duration  float64 `json:"duration"`
	} `json:"result"`
	Progress float64  `json:"progress"` // [0, 1]
	Details  struct{} `json:"details"`
	Preview  struct{} `json:"preview"`
}

// ImageWebhookResponse is the provider-specific image webhook response.
type ImageWebhookResponse struct {
	ID         string `json:"id"`
	Status     string `json:"status"`
	WebhookURL string `json:"webhook_url,omitzero"`
}

// ErrorResponse is the provider-specific error response.
type ErrorResponse struct {
	Detail string `json:"detail"`
}

func (er *ErrorResponse) Error() string {
	return er.Detail
}

// IsAPIError implements base.ErrorResponseI.
func (er *ErrorResponse) IsAPIError() bool {
	return true
}
