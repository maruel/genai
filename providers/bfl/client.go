// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package bfl implements a client for Black Forest Labs API.
//
// It is described at https://docs.bfl.ml/quick_start/generating_images
package bfl

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/roundtrippers"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/scoreboard"
)

//go:embed scoreboard.json
var scoreboardJSON []byte

// Scoreboard for Black Forest Labs.
func Scoreboard() scoreboard.Score {
	var s scoreboard.Score
	d := json.NewDecoder(bytes.NewReader(scoreboardJSON))
	d.DisallowUnknownFields()
	if err := d.Decode(&s); err != nil {
		panic(fmt.Errorf("failed to unmarshal scoreboard.json: %w", err))
	}
	return s
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

// Client implements genai.Provider.
type Client struct {
	base.NotImplemented
	impl   base.ProviderBase[*ErrorResponse]
	remote string
}

// New creates a new client to talk to the Black Forest Labs platform API.
//
// If ProviderOptionAPIKey is not provided, it tries to load it from the BFL_API_KEY environment variable.
// If none is found, it will still return a client coupled with an base.ErrAPIKeyRequired error.
// Get your API key at https://dashboard.bfl.ai/keys
//
// ProviderOptionRemote defaults to "https://api.bfl.ai" and can be specified to use a region specific backend.
//
// To use multiple models, create multiple clients.
// Use one of the model from https://docs.bfl.ml/quick_start/generating_images
func New(ctx context.Context, opts ...genai.ProviderOption) (*Client, error) {
	var apiKey, model, remote string
	var modalities genai.Modalities
	var wrapper func(http.RoundTripper) http.RoundTripper
	if err := base.CheckDuplicateOptions(opts); err != nil {
		return nil, err
	}
	for _, opt := range opts {
		if err := opt.Validate(); err != nil {
			return nil, err
		}
		switch v := opt.(type) {
		case genai.ProviderOptionAPIKey:
			apiKey = string(v)
		case genai.ProviderOptionModel:
			model = string(v)
		case genai.ProviderOptionModalities:
			modalities = genai.Modalities(v)
		case genai.ProviderOptionRemote:
			remote = string(v)
		case genai.ProviderOptionTransportWrapper:
			wrapper = v
		default:
			return nil, fmt.Errorf("unsupported option type %T", opt)
		}
	}
	const apiKeyURL = "https://dashboard.bfl.ai/keys"
	var err error
	if apiKey == "" {
		if apiKey = os.Getenv("BFL_API_KEY"); apiKey == "" {
			err = &base.ErrAPIKeyRequired{EnvVar: "BFL_API_KEY", URL: apiKeyURL}
		}
	}
	mod := genai.Modalities{genai.ModalityImage}
	if len(modalities) != 0 && !slices.Equal(modalities, mod) {
		return nil, fmt.Errorf("unexpected option Modalities %s, only image is supported", mod)
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	if remote == "" {
		remote = "https://api.bfl.ai"
	}
	c := &Client{
		remote: remote,
		impl: base.ProviderBase[*ErrorResponse]{
			APIKeyURL: apiKeyURL,
			Lenient:   internal.BeLenient,
			Client: http.Client{
				Transport: &roundtrippers.Header{
					Header:    http.Header{"x-key": {apiKey}},
					Transport: &roundtrippers.RequestID{Transport: t},
				},
			},
		},
	}
	if err == nil {
		switch model {
		case "":
		case string(genai.ModelCheap), string(genai.ModelGood), string(genai.ModelSOTA):
			c.impl.Model = c.selectBestImageModel(model)
			c.impl.OutputModalities = mod
		default:
			c.impl.Model = model
			c.impl.OutputModalities = mod
		}
	}
	return c, err
}

// selectBestImageModel selects the model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestImageModel(preference string) string {
	// If Black Forest Labs ever implement model listing, start using it!
	switch preference {
	case string(genai.ModelCheap):
		// flex-2-dev consistently returns 403 / Forbidden; I don't believe they serve it yet.
		// https://docs.bfl.ml/api-reference/models/generate-an-image-with-flux1-[dev]
		return "flux-dev"
	case string(genai.ModelGood), "":
		return "flux-2-pro"
	case string(genai.ModelSOTA):
		return "flux-2-max"
	default:
		return ""
	}
}

// Name implements genai.Provider.
//
// It returns the name of the provider.
func (c *Client) Name() string {
	return "bfl"
}

// ModelID implements genai.Provider.
//
// It returns the selected model ID.
func (c *Client) ModelID() string {
	return c.impl.Model
}

// OutputModalities implements genai.Provider.
//
// It returns the output modalities, i.e. what kind of output the model will generate (text, audio, image,
// video, etc).
func (c *Client) OutputModalities() genai.Modalities {
	return c.impl.OutputModalities
}

// Scoreboard implements genai.Provider.
func (c *Client) Scoreboard() scoreboard.Score {
	return Scoreboard()
}

// HTTPClient returns the HTTP client to fetch results (e.g. videos) generated by the provider.
func (c *Client) HTTPClient() *http.Client {
	return &c.impl.Client
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

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	// They recommend in their documentation to poll every 0.5s.
	waitForPoll := 500 * time.Millisecond
	filtered := make([]genai.GenOption, 0, len(opts))
	for _, opt := range opts {
		if v, ok := opt.(genai.GenOptionPollInterval); ok {
			waitForPoll = time.Duration(v)
		} else {
			filtered = append(filtered, opt)
		}
	}
	id, err := c.GenAsync(ctx, msgs, filtered...)
	if err != nil {
		return genai.Result{}, err
	}
	// TODO: Expose a webhook with a custom OptionsImage.
	// Loop until the result is available.
	for {
		select {
		case <-ctx.Done():
			return genai.Result{}, ctx.Err()
		case <-time.After(waitForPoll):
			if res, err := c.PokeResult(ctx, id); res.Usage.FinishReason != genai.Pending {
				return res, err
			}
		}
	}
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	return base.SimulateStream(ctx, c, msgs, opts...)
}

// GenAsync implements genai.ProviderGenAsync.
//
// It requests the providers' asynchronous API and returns the job ID.
func (c *Client) GenAsync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Job, error) {
	req := ImageRequest{}
	if err := req.Init(msgs, c.impl.Model, opts...); err != nil {
		return "", err
	}
	reqresp, err := c.GenAsyncRaw(ctx, &req)
	// We return the polling URL instead of the ID, e.g. genai.Job(reqresp.ID). BFL is very clear that we should
	// not construct the URL ourselves.
	return genai.Job(reqresp.PollingURL), err
}

// GenAsyncRaw runs an asynchronous generation request.
func (c *Client) GenAsyncRaw(ctx context.Context, req *ImageRequest) (ImageRequestResponse, error) {
	// https://docs.bfl.ml/quick_start/generating_images
	// https://docs.bfl.ai/integration_guidelines#polling-url-usage
	// TODO: Switch to use PollingURL
	reqresp := ImageRequestResponse{}
	err := c.impl.DoRequest(ctx, "POST", c.remote+"/v1/"+c.impl.Model, &req, &reqresp)
	return reqresp, err
}

// PokeResult implements genai.ProviderGenAsync.
//
// It retrieves the result for a job ID.
func (c *Client) PokeResult(ctx context.Context, id genai.Job) (genai.Result, error) {
	res := genai.Result{}
	imgres, err := c.PokeResultRaw(ctx, id)
	if err != nil {
		return res, err
	}
	res.Usage.Limits = processHeaders(c.impl.LastResponseHeaders())
	if imgres.Status == "Pending" {
		res.Usage.FinishReason = genai.Pending
		return res, nil
	}
	if imgres.Status != "Ready" {
		return res, fmt.Errorf("unexpected status: %#v", imgres)
	}
	res.Replies = []genai.Reply{{Doc: genai.Doc{Filename: "content.jpg", URL: imgres.Result.Sample}}}
	if err := res.Validate(); err != nil {
		return res, err
	}
	return res, nil
}

// PokeResultRaw retrieves the result for a job ID if already available.
func (c *Client) PokeResultRaw(ctx context.Context, id genai.Job) (ImageResult, error) {
	res := ImageResult{}
	// The job ID is an URL.
	p, err := url.Parse(string(id))
	if err != nil {
		return res, fmt.Errorf("job ID should be the polling URL: %w", err)
	}
	if !strings.HasSuffix(p.Hostname(), ".bfl.ai") {
		return res, errors.New("job ID should be the polling URL hosted on bfl.ai")
	}
	err = c.impl.DoRequest(ctx, "GET", string(id), nil, &res)
	return res, err
}

// Capabilities implements genai.Provider.
func (c *Client) Capabilities() genai.ProviderCapabilities {
	return genai.ProviderCapabilities{
		GenAsync: true,
	}
}

var _ genai.Provider = &Client{}
