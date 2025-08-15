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
	"slices"
	"strconv"
	"strings"
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
	Country:      "DE",
	DashboardURL: "https://dashboard.bfl.ai/",
	Scenarios: []genai.Scenario{
		{
			Models: []string{
				"flux-dev", // https://docs.bfl.ai/api-reference/tasks/generate-an-image-with-flux1-[dev]
			},
			In: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			Out: map[genai.Modality]genai.ModalCapability{
				genai.ModalityImage: {
					URL:              true,
					SupportedFormats: []string{"image/jpeg"},
				},
			},
			GenDoc: &genai.FunctionalityDoc{
				ReportRateLimits:   true,
				BrokenTokenUsage:   genai.True,
				BrokenFinishReason: true,
				Seed:               true,
			},
		},
		// It cost a fair amount of money to run the test.
		/*
			{
				Models: []string{
					"flux-kontext-dev",    // Described at https://docs.bfl.ai/api-reference/models/report-model-usage
					"flux-tools",          // Described at https://docs.bfl.ai/api-reference/models/report-model-usage
					"flux-pro-1.0-depth",  // https://docs.bfl.ai/api-reference/tasks/generate-an-image-with-flux1-depth-[pro]-using-a-control-image
					"flux-pro-1.0-canny",  // https://docs.bfl.ai/api-reference/tasks/generate-an-image-with-flux1-canny-[pro]-using-a-control-image
					"flux-pro-1.0-expand", // https://docs.bfl.ai/api-reference/tasks/expand-an-image-by-adding-pixels-on-any-side
					"flux-pro-1.0-fill",   // https://docs.bfl.ai/api-reference/tasks/generate-an-image-with-flux1-fill-[pro]-using-an-input-image-and-mask
					"flux-kontext-pro",    // https://docs.bfl.ai/api-reference/tasks/edit-or-create-an-image-with-flux-kontext-pro
					"flux-kontext-max",    // https://docs.bfl.ai/api-reference/tasks/edit-or-create-an-image-with-flux-kontext-max
					"flux-pro-1.1-ultra",  // https://docs.bfl.ai/api-reference/tasks/generate-an-image-with-flux-11-[pro]-with-ultra-mode-and-optional-raw-mode
					"flux-pro-1.1",        // https://docs.bfl.ai/api-reference/tasks/generate-an-image-with-flux-11-[pro]
					"flux-pro",            // https://docs.bfl.ai/api-reference/tasks/generate-an-image-with-flux1-[pro]
				},
			},
		*/
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
	ID string `json:"id"`
	// PollingURL is the URL to poll for the result. Generated images expire after 10 minutes and become
	// inaccessible.
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
	Preview  struct{} `json:"preview"`
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

func (er *ErrorResponse) Error() string {
	return er.Detail
}

func (er *ErrorResponse) IsAPIError() bool {
	return true
}

// Client implements genai.ProviderGen and genai.ProviderModel.
type Client struct {
	base.Provider[*ErrorResponse]
	Model  string
	remote string
}

// New creates a new client to talk to the Black Forest Labs platform API.
//
// If opts.APIKey is not provided, it tries to load it from the BFL_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://dashboard.bfl.ai/keys
//
// opts.Remote defaults to "https://api.bfl.ai" and can be specified to use a region specific backend.
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.bfl.ml/quick_start/generating_images
//
// wrapper optionally wraps the HTTP transport. Useful for HTTP recording and playback, or to tweak HTTP
// retries, or to throttle outgoing requests.
func New(opts *genai.OptionsProvider, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if opts.AccountID != "" {
		return nil, errors.New("unexpected option AccountID")
	}
	apiKey := opts.APIKey
	const apiKeyURL = "https://dashboard.bfl.ai/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("BFL_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "BFL_API_KEY", URL: apiKeyURL}
		}
	}
	// If Black Forest Labs ever implement model listing, start using it!
	// When no model is specified, use the dev model since costs increase quite fast.
	model := opts.Model
	switch model {
	case base.NoModel:
		model = ""
	case base.PreferredCheap, "":
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
	remote := opts.Remote
	if remote == "" {
		remote = "https://api.bfl.ai"
	}
	return &Client{
		Model:  model,
		remote: remote,
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

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit

	limit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit"), 10, 64)
	remaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining"), 10, 64)
	reset, _ := strconv.ParseInt(h.Get("X-Ratelimit-Reset"), 10, 64)

	if limit != 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     limit,
			Remaining: remaining,
			Reset:     time.Unix(reset, 0).Round(10 * time.Millisecond),
		})
	}
	return limits
}

// GenDoc synchronously generates a document.
//
// Generation can be rather slow, several seconds or minutes.
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
			if res, err := c.PokeResult(ctx, id); res.FinishReason != genai.Pending {
				return res, err
			}
		}
	}
}

func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Job, error) {
	if err := msgs.Validate(); err != nil {
		return "", err
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return "", err
		}
		if supported := opts.Modalities(); !slices.Contains(supported, genai.ModalityImage) {
			return "", fmt.Errorf("modality image not supported, supported: %s", supported)
		}
	}
	if len(msgs) != 1 {
		return "", errors.New("must pass exactly one Message")
	}
	msg := msgs[0]
	for i := range msg.Requests {
		if msg.Requests[i].Text != "" {
			continue
		}
		// Check if this is a text/plain document that we can handle
		if !msg.Requests[i].Doc.IsZero() {
			mimeType, data, err := msg.Requests[i].Doc.Read(10 * 1024 * 1024)
			if err != nil {
				return "", fmt.Errorf("failed to read document: %w", err)
			}
			// TODO: kontext support images as imnput
			if !strings.HasPrefix(mimeType, "text/plain") {
				return "", errors.New("only text and text/plain documents can be passed as input")
			}
			if msg.Requests[i].Doc.URL != "" {
				return "", errors.New("text/plain documents must be provided inline, not as a URL")
			}
			// Convert document content to text content
			msg.Requests[i].Text = string(data)
			msg.Requests[i].Doc = genai.Doc{}
			continue
		}
		return "", errors.New("unknown Request type")
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
		default:
			return "", fmt.Errorf("unsupported options type %T", opts)
		}
	}
	reqresp, err := c.GenAsyncRaw(ctx, req)
	return genai.Job(reqresp.ID), err
}

func (c *Client) GenAsyncRaw(ctx context.Context, req ImageRequest) (ImageRequestResponse, error) {
	// https://docs.bfl.ml/quick_start/generating_images
	// https://docs.bfl.ai/integration_guidelines#polling-url-usage
	// TODO: Switch to use PollingURL
	reqresp := ImageRequestResponse{}
	err := c.DoRequest(ctx, "POST", c.remote+"/v1/"+c.Model, &req, &reqresp)
	return reqresp, err
}

// PokeResult retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	imgres, err := c.PokeResultRaw(ctx, id)
	if err != nil {
		return res, err
	}
	res.Limits = processHeaders(c.LastResponseHeaders())
	if imgres.Status == "Pending" {
		res.FinishReason = genai.Pending
		return res, nil
	}
	if imgres.Status != "Ready" {
		return res, fmt.Errorf("unexpected status: %#v", imgres)
	}
	res.Replies = []genai.Reply{{Doc: genai.Doc{Filename: "content.jpg", URL: imgres.Result.Sample}}}
	return res, res.Validate()
}

// PokeResultRaw retrieves the result for a job ID if already available.
func (c *Client) PokeResultRaw(ctx context.Context, id genai.Job) (ImageResult, error) {
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
