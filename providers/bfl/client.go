// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package bfl implements a client for Black Forest Labs API.
//
// It is described at https://docs.bfl.ml/quick_start/generating_images
package bfl

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
)

// Scoreboard for Black Forest Labs.
//
// # Warnings
//
//   - The API is asynchronous and supports webhook but genai doesn't expose these yet, so the client polls
//     for now.
//   - The API and acceptable values are highly model-dependent, so it is easy to make an invalid request.
//
// See https://docs.bfl.ml/quick_start/generating_images
var Scoreboard = genai.Scoreboard{
	Scenarios: []genai.Scenario{
		{
			In:  []genai.Modality{genai.ModalityText},
			Out: []genai.Modality{genai.ModalityImage},
			Models: []string{
				"flux-dev",
				"flux-kontext-pro",
				"flux-kontext-max",
				"flux-pro-1.1-ultra",
				"flux-pro-1.1",
				"flux-pro",
			},
			GenDoc: &genai.FunctionalityDoc{
				OutputURL:          true,
				BrokenTokenUsage:   true,
				BrokenFinishReason: true,
			},
		},
	},
}

// ImageRequest is highly dependent on the model used.
//
// It's in https://docs.bfl.ml/api-reference/tasks/generate-an-image-with-flux-11-[pro] and the likes.
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
	Steps    int64   `json:"steps,omitzero"`    // Default 40, [1, 50]
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

	// FLUX Kontext Pro, Max
	InputImage []byte `json:"input_image,omitzero"`
}

// ImageRequestResponse is the same for all requests since everything is asynchronous.
//
// It's in https://docs.bfl.ml/api-reference/tasks/generate-an-image-with-flux-11-[pro] and the likes.
type ImageRequestResponse struct {
	ID         string `json:"id"`
	PollingURL string `json:"polling_url"`
}

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
}

type ImageWebhookResponse struct {
	ID         string `json:"id"`
	Status     string `json:"status"`
	WebhookURL string `json:"webhook_url,omitzero"`
}

//

type ErrorResponse struct {
	Detail string `json:"detail"`
}

func (er *ErrorResponse) String() string {
	return "error " + er.Detail
}

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	base.Provider[*ErrorResponse]
	Model string
}

// New creates a new client to talk to the Black Forest Labs platform API.
//
// If apiKey is not provided, it tries to load it from the BFL_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://dashboard.bfl.ai/keys
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.bfl.ml/quick_start/generating_images
//
// Pass model base.PreferredCheap to use a good cheap model, base.PreferredGood for a good model or
// base.PreferredSOTA to use its SOTA model. Keep in mind that as providers cycle through new models, it's
// possible the model is not available anymore.
//
// wrapper can be used to throttle outgoing requests, record calls, etc. It defaults to base.DefaultTransport.
func New(apiKey, model string, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	const apiKeyURL = "https://dashboard.bfl.ai/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("BFL_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "BFL_API_KEY", URL: apiKeyURL}
		}
	}
	// If Black Forest Labs ever implement model listing, add this.
	switch model {
	case base.PreferredCheap:
		model = "flux-dev"
	case base.PreferredGood:
		model = "flux-pro-1.1"
	case base.PreferredSOTA:
		model = "flux-pro-1.1-ultra"
	default:
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	return &Client{
		Model: model,
		Provider: base.Provider[*ErrorResponse]{
			ProviderName: "bfl",
			APIKeyURL:    apiKeyURL,
			ClientJSON: httpjson.Client{
				Lenient: internal.BeLenient,
				Client: &http.Client{
					Transport: &roundtrippers.Header{
						Header:    http.Header{"x-key": {apiKey}},
						Transport: &roundtrippers.RequestID{Transport: t},
					},
				},
			},
		},
	}, err
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ModelID() string {
	return c.Model
}

func (c *Client) GenDoc(ctx context.Context, msg genai.Message, opts genai.Options) (genai.Result, error) {
	id, err := c.GenAsync(ctx, genai.Messages{msg}, opts)
	if err != nil {
		return genai.Result{}, err
	}
	// They recommend in their documentation to poll every 0.5s.
	// TODO: Expose a webhook with a custom OptionsImage.
	for {
		select {
		case <-ctx.Done():
			return genai.Result{}, ctx.Err()
		case <-time.After(waitForPoll):
		}
		res, err := c.PokeResult(ctx, id)
		if res.FinishReason == genai.Pending {
			continue
		}
		return res, err
	}
}

func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Job, error) {
	if len(msgs) != 1 {
		return "", errors.New("must pass exactly one Message")
	}
	msg := msgs[0]
	for i := range msg.Contents {
		if msg.Contents[i].Text == "" {
			// TODO: kontext
			return "", errors.New("only text can be passed as input")
		}
	}
	req := ImageRequest{
		Prompt: msg.AsText(),
	}
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsImage:
			req.Height = int64(v.Height)
			req.Width = int64(v.Width)
			req.Seed = v.Seed
		case *genai.OptionsText:
			req.Seed = v.Seed
		}
	}
	reqresp := ImageRequestResponse{}
	if err := c.DoRequest(ctx, "POST", "https://api.bfl.ai/v1/"+c.Model, &req, &reqresp); err != nil {
		return "", err
	}
	return genai.Job(reqresp.ID), nil
}

// PokeResult retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	imgres, err := c.GetResultRaw(ctx, id)
	if err != nil {
		return res, err
	}
	if imgres.Status == "Pending" {
		res.FinishReason = genai.Pending
		return res, nil
	}
	if imgres.Status != "Ready" {
		return res, fmt.Errorf("unexpected status: %#v", imgres)
	}
	res.Role = genai.Assistant
	res.Contents = []genai.Content{{Filename: "content.jpg", URL: imgres.Result.Sample}}
	return res, nil
}

// GetResult retrieves the result for a job ID.
func (c *Client) GetResultRaw(ctx context.Context, id genai.Job) (ImageResult, error) {
	res := ImageResult{}
	u := "https://api.us1.bfl.ai/v1/get_result?id=" + url.QueryEscape(string(id))
	err := c.DoRequest(ctx, "GET", u, nil, &res)
	return res, err
}

var waitForPoll = 500 * time.Millisecond

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGenAsync   = &Client{}
	_ genai.ProviderGenDoc     = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
