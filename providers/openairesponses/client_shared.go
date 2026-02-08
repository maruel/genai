// Copyright 2025 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// This file contains all the structures and code that is exactly the same between the OpenAI chat and responses APIs.

package openairesponses

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/bb"
	"github.com/maruel/genai/scoreboard"
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

//

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

// Background is only supported on gpt-image-1.
type Background string

// Background mode values.
const (
	BackgroundAuto        Background = "auto"
	BackgroundTransparent Background = "transparent"
	BackgroundOpaque      Background = "opaque"
)

// ImageResponse is the provider-specific image generation response.
type ImageResponse struct {
	Created base.Time         `json:"created"`
	Data    []ImageChoiceData `json:"data"`
	Usage   struct {
		InputTokens        int64 `json:"input_tokens"`
		OutputTokens       int64 `json:"output_tokens"`
		TotalTokens        int64 `json:"total_tokens"`
		InputTokensDetails struct {
			TextTokens  int64 `json:"text_tokens"`
			ImageTokens int64 `json:"image_tokens"`
		} `json:"input_tokens_details"`
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

//

// Model is documented at https://platform.openai.com/docs/api-reference/models/object
//
// Sadly the modalities aren't reported. The only way I can think of to find it at run time is to fetch
// https://platform.openai.com/docs/models/gpt-4o-mini-realtime-preview, find the div containing
// "Modalities:", then extract the modalities from the text.
type Model struct {
	ID      string    `json:"id"`
	Object  string    `json:"object"`
	Created base.Time `json:"created"`
	OwnedBy string    `json:"owned_by"`
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

//

// File is documented at https://platform.openai.com/docs/api-reference/files/object
type File struct {
	Bytes         int64     `json:"bytes"` // File size
	CreatedAt     base.Time `json:"created_at"`
	ExpiresAt     base.Time `json:"expires_at"`
	Filename      string    `json:"filename"`
	ID            string    `json:"id"`
	Object        string    `json:"object"`         // "file"
	Purpose       string    `json:"purpose"`        // One of: assistants, assistants_output, batch, batch_output, fine-tune, fine-tune-results and vision
	Status        string    `json:"status"`         // Deprecated
	StatusDetails string    `json:"status_details"` // Deprecated
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

//

// detectModelModalities tries its best to figure out the modality of a model
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) detectModelModalities(ctx context.Context, model string) (genai.Modalities, error) { //nolint:unparam // ctx for future use.
	// Damn you OpenAI! This is super fragile.
	// TODO: Fill out the scoreboard from https://platform.openai.com/docs/models then use that. This is sad
	// because ListModels is useless. See
	// https://discord.com/channels/974519864045756446/1070006915414900886/threads/1408629226864775265 for the
	// author's feature request.
	if strings.HasPrefix(model, "dall") || strings.Contains(model, "image") {
		return genai.Modalities{genai.ModalityImage}, nil
		// TODO } else if strings.Contains(model, "audio") {
		//	return genai.Modalities{genai.ModalityAudio, genai.ModalityText}, nil
	}
	return genai.Modalities{genai.ModalityText}, nil
}

// selectBestTextModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestTextModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	var created base.Time
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Do not select the specialized models automatically.
		if strings.Contains(m.ID, "audio") || strings.Contains(m.ID, "codex") || strings.Contains(m.ID, "deep") || strings.Contains(m.ID, "image") || strings.Contains(m.ID, "realtime") || strings.Contains(m.ID, "search") {
			continue
		}
		// The o family of models is not usable with completion API.
		if strings.HasPrefix(m.ID, "o") {
			continue
		}
		switch {
		case cheap:
			if strings.HasPrefix(m.ID, "gpt") && strings.HasSuffix(m.ID, "-nano") && (created == 0 || m.Created > created) {
				created = m.Created
				selectedModel = m.ID
			}
		case good:
			if strings.HasPrefix(m.ID, "gpt") && strings.HasSuffix(m.ID, "-mini") && (created == 0 || m.Created > created) {
				created = m.Created
				selectedModel = m.ID
			}
		default:
			if strings.HasPrefix(m.ID, "gpt") && !strings.Contains(m.ID, "-nano") && !strings.Contains(m.ID, "-mini") && (created == 0 || m.Created > created) {
				created = m.Created
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestImageModel selects the most appropriate model based on the preference (cheap, good, or SOTA).
//
// We may want to make this function overridable in the future by the client since this is going to break one
// day or another.
func (c *Client) selectBestImageModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood)
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// OpenAI doesn't report much for each model. :(
		isCheap := strings.HasPrefix(m.ID, "dall")
		isGood := strings.HasSuffix(m.ID, "-mini")
		isSOTA := strings.Contains(m.ID, "image")
		if !isCheap && !isSOTA {
			continue
		}
		switch {
		case cheap:
			if isCheap {
				if selectedModel == "" || m.ID > selectedModel {
					selectedModel = m.ID
				}
			}
		case good:
			if isGood {
				if selectedModel == "" || m.ID > selectedModel {
					selectedModel = m.ID
				}
			}
		default:
			if isSOTA && !isGood {
				selectedModel = m.ID
			}
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a model automatically")
	}
	return selectedModel, nil
}

// selectBestVideoModel selects the most appropriate video model based on the preference (cheap, good, or SOTA).
//
// Video models are identified by the "sora-" prefix.
func (c *Client) selectBestVideoModel(ctx context.Context, preference string) (string, error) {
	mdls, err := c.ListModels(ctx)
	if err != nil {
		return "", fmt.Errorf("failed to automatically select the model: %w", err)
	}
	cheap := preference == string(genai.ModelCheap)
	good := preference == string(genai.ModelGood) || preference == ""
	selectedModel := ""
	for _, mdl := range mdls {
		m := mdl.(*Model)
		// Only consider models starting with "sora-"
		if !strings.HasPrefix(m.ID, "sora-") {
			continue
		}
		// Determine if model is pro based on name
		isPro := strings.Contains(m.ID, "pro")
		matches := false
		switch {
		case cheap:
			matches = !isPro
		case good:
			matches = !isPro
		default:
			matches = isPro
		}
		if !matches {
			continue
		}
		// Select the best available model, preferring newer versions lexicographically
		if selectedModel == "" || m.ID > selectedModel {
			selectedModel = m.ID
		}
	}
	if selectedModel == "" {
		return "", errors.New("failed to find a video model automatically")
	}
	return selectedModel, nil
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

// GenSync implements genai.Provider.
func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (genai.Result, error) {
	if c.isAudio() {
		return genai.Result{}, errors.New("OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio")
	}
	if c.isImage() || c.isVideo() {
		if len(msgs) != 1 {
			return genai.Result{}, errors.New("must pass exactly one Message")
		}
		return c.genDoc(ctx, &msgs[0], opts...)
	}
	return c.impl.GenSync(ctx, msgs, opts...)
}

// GenStream implements genai.Provider.
func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, opts ...genai.GenOption) (iter.Seq[genai.Reply], func() (genai.Result, error)) {
	if c.isAudio() {
		return func(yield func(genai.Reply) bool) {}, func() (genai.Result, error) {
			return genai.Result{}, errors.New("OpenAI Responses API does not support audio output as of December 2025; see https://platform.openai.com/docs/guides/audio")
		}
	}
	if c.isImage() || c.isVideo() {
		return base.SimulateStream(ctx, c, msgs, opts...)
	}
	return c.impl.GenStream(ctx, msgs, opts...)
}

// genDoc is a simplified version of GenSync.
func (c *Client) genDoc(ctx context.Context, msg *genai.Message, opts ...genai.GenOption) (genai.Result, error) {
	// https://platform.openai.com/docs/api-reference/images/create
	res := genai.Result{}
	if err := c.impl.Validate(); err != nil {
		return res, err
	}
	req := ImageRequest{}
	if err := req.Init(msg, c.impl.Model, opts...); err != nil {
		return res, err
	}
	url := "https://api.openai.com/v1/images/generations"

	// It is very different because it requires a multi-part upload.
	// https://platform.openai.com/docs/api-reference/images/createEdit
	// url = "https://api.openai.com/v1/images/edits"

	resp := ImageResponse{}
	if err := c.impl.DoRequest(ctx, "POST", url, &req, &resp); err != nil {
		return res, err
	}
	res.Replies = make([]genai.Reply, len(resp.Data))
	for i := range resp.Data {
		n := "content.jpg"
		if len(resp.Data) > 1 {
			n = fmt.Sprintf("content%d.jpg", i+1)
		}
		if url := resp.Data[i].URL; url != "" {
			res.Replies[i].Doc = genai.Doc{Filename: n, URL: url}
		} else if d := resp.Data[i].B64JSON; len(d) != 0 {
			res.Replies[i].Doc = genai.Doc{Filename: n, Src: &bb.BytesBuffer{D: resp.Data[i].B64JSON}}
		} else {
			return res, errors.New("internal error")
		}
	}
	if err := res.Validate(); err != nil {
		return res, err
	}
	return res, nil
}

// ListModels implements genai.Provider.
func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	if c.impl.PreloadedModels != nil {
		return c.impl.PreloadedModels, nil
	}
	// https://platform.openai.com/docs/api-reference/models/list
	var resp ModelsResponse
	if err := c.impl.DoRequest(ctx, "GET", "https://api.openai.com/v1/models", nil, &resp); err != nil {
		return nil, err
	}
	return resp.ToModels(), nil
}

func (c *Client) isAudio() bool {
	return slices.Contains(c.impl.OutputModalities, genai.ModalityAudio)
}

func (c *Client) isImage() bool {
	return slices.Contains(c.impl.OutputModalities, genai.ModalityImage)
}

func (c *Client) isVideo() bool {
	return slices.Contains(c.impl.OutputModalities, genai.ModalityVideo)
}

func processHeaders(h http.Header) []genai.RateLimit {
	var limits []genai.RateLimit
	requestsLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Requests"), 10, 64)
	requestsRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Requests"), 10, 64)
	requestsReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Requests"))

	tokensLimit, _ := strconv.ParseInt(h.Get("X-Ratelimit-Limit-Tokens"), 10, 64)
	tokensRemaining, _ := strconv.ParseInt(h.Get("X-Ratelimit-Remaining-Tokens"), 10, 64)
	tokensReset, _ := time.ParseDuration(h.Get("X-Ratelimit-Reset-Tokens"))

	if requestsLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Requests,
			Period:    genai.PerOther,
			Limit:     requestsLimit,
			Remaining: requestsRemaining,
			Reset:     time.Now().Add(requestsReset).Round(10 * time.Millisecond),
		})
	}
	if tokensLimit > 0 {
		limits = append(limits, genai.RateLimit{
			Type:      genai.Tokens,
			Period:    genai.PerOther,
			Limit:     tokensLimit,
			Remaining: tokensRemaining,
			Reset:     time.Now().Add(tokensReset).Round(10 * time.Millisecond),
		})
	}
	return limits
}
