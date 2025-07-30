// Copyright 2024 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Package llamacpp implements a client for the llama-server native API, not
// the OpenAI compatible one.
//
// It is described at
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#api-endpoints
//
// The implementation is at https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp
package llamacpp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/invopop/jsonschema"
	"github.com/maruel/genai"
	"github.com/maruel/genai/base"
	"github.com/maruel/genai/internal"
	"github.com/maruel/genai/internal/sse"
	"github.com/maruel/httpjson"
	"github.com/maruel/roundtrippers"
	"golang.org/x/sync/errgroup"
)

// Scoreboard for llama.cpp.
//
// # Warnings
//
//   - llama.cpp supports now more than what the client here has implemented, like vision and tool calling.
var Scoreboard = genai.Scoreboard{
	Country:      "Local",
	DashboardURL: "https://github.com/ggml-org/llama.cpp",
	Scenarios: []genai.Scenario{
		{
			Models: []string{"unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q5_K_M.gguf"},
			// TODO: It supports genai.ModalityImage
			In: map[genai.Modality]genai.ModalCapability{
				genai.ModalityText: {Inline: true},
			},
			Out: map[genai.Modality]genai.ModalCapability{genai.ModalityText: {Inline: true}},
			// TODO: Implement all the feature set.
			GenSync: &genai.FunctionalityText{
				Seed: true,
			},
			GenStream: &genai.FunctionalityText{
				Seed: true,
			},
		},
	},
}

// HealthResponse is documented at
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#get-health-returns-heath-check-result
type HealthResponse struct {
	Status          string
	SlotsIdle       int64 `json:"slots_idle"`
	SlotsProcessing int64 `json:"slots_processing"`
}

// CompletionRequest is documented at
// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
type CompletionRequest struct {
	// TODO: Prompt can be a string, a list of tokens or a mix.
	Prompt              string             `json:"prompt"`
	Temperature         float64            `json:"temperature,omitzero"`
	DynaTempRange       float64            `json:"dynatemp_range,omitzero"`
	DynaTempExponent    float64            `json:"dynatemp_exponent,omitzero"`
	TopK                int64              `json:"top_k,omitzero"`
	TopP                float64            `json:"top_p,omitzero"`
	MinP                float64            `json:"min_p,omitzero"`
	NPredict            int64              `json:"n_predict,omitzero"` // Maximum number of tokens to predict
	NIndent             int64              `json:"n_indent,omitzero"`
	NKeep               int64              `json:"n_keep,omitzero"`
	Stream              bool               `json:"stream"`
	Stop                []string           `json:"stop,omitzero"`
	TypicalP            float64            `json:"typical_p,omitzero"`
	RepeatPenalty       float64            `json:"repeat_penalty,omitzero"`
	RepeatLastN         int64              `json:"repeat_last_n,omitzero"`
	PresencePenalty     float64            `json:"presence_penalty,omitzero"`
	FrequencyPenalty    float64            `json:"frequency_penalty,omitzero"`
	DryMultiplier       float64            `json:"dry_multiplier,omitzero"`
	DryBase             float64            `json:"dry_base,omitzero"`
	DryAllowedLength    int64              `json:"dry_allowed_length,omitzero"`
	DryPenaltyLastN     int64              `json:"dry_penalty_last_n,omitzero"`
	DrySequenceBreakers []string           `json:"dry_sequence_breakers,omitzero"`
	XTCProbability      float64            `json:"xtc_probability,omitzero"`
	XTCThreshold        float64            `json:"xtc_threshold,omitzero"`
	Mirostat            int32              `json:"mirostat,omitzero"`
	MirostatTau         float64            `json:"mirostat_tau,omitzero"`
	MirostatEta         float64            `json:"mirostat_eta,omitzero"`
	Grammar             string             `json:"grammar,omitzero"`
	JSONSchema          *jsonschema.Schema `json:"json_schema,omitzero"`
	Seed                int64              `json:"seed,omitzero"`
	IgnoreEos           bool               `json:"ignore_eos,omitzero"`
	LogitBias           []any              `json:"logit_bias,omitzero"`
	Nprobs              int64              `json:"n_probs,omitzero"`
	MinKeep             int64              `json:"min_keep,omitzero"`
	TMaxPredictMS       int64              `json:"t_max_predict_ms,omitzero"`
	ImageData           []any              `json:"image_data,omitzero"`
	IDSlot              int64              `json:"id_slot,omitzero"`
	CachePrompt         bool               `json:"cache_prompt,omitzero"`
	ReturnTokens        bool               `json:"return_tokens,omitzero"`
	Samplers            []string           `json:"samplers,omitzero"`
	TimingsPerToken     bool               `json:"timings_per_token,omitzero"`
	PostSamplingProbs   bool               `json:"post_sampling_probs,omitzero"`
	ResponseFields      []string           `json:"response_fields,omitzero"`
	Lora                []Lora             `json:"lora,omitzero"`
}

type Lora struct {
	ID    int64   `json:"id,omitzero"`
	Scale float64 `json:"scale,omitzero"`
}

// Init initializes the provider specific completion request with the generic completion request.
func (c *CompletionRequest) Init(msgs genai.Messages, opts genai.Options, model string) error {
	var errs []error
	var unsupported []string
	c.CachePrompt = true
	if opts != nil {
		switch v := opts.(type) {
		case *genai.OptionsText:
			c.NPredict = v.MaxTokens
			c.Seed = v.Seed
			c.Temperature = v.Temperature
			c.TopP = v.TopP
			c.TopK = v.TopK
			c.Stop = v.Stop
			if v.ReplyAsJSON {
				errs = append(errs, errors.New("unsupported option ReplyAsJSON"))
			}
			if v.DecodeAs != nil {
				errs = append(errs, errors.New("unsupported option DecodeAs"))
			}
			if len(v.Tools) != 0 {
				// TODO: June 2025, support was added recently for streaming.
				errs = append(errs, errors.New("unsupported option Tools"))
			}
		default:
			errs = append(errs, fmt.Errorf("unsupported options type %T", opts))
		}
	}
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

type CompletionResponse struct {
	Index              int64   `json:"index"`
	Content            string  `json:"content"`
	Tokens             []int64 `json:"tokens"`
	IDSlot             int64   `json:"id_slot"`
	Stop               bool    `json:"stop"`
	Model              string  `json:"model"`
	TokensPredicted    int64   `json:"tokens_predicted"`
	TokensEvaluated    int64   `json:"tokens_evaluated"`
	GenerationSettings struct {
		NPredict            int64    `json:"n_predict"`
		Seed                int64    `json:"seed"`
		Temperature         float64  `json:"temperature"`
		DynaTempRange       float64  `json:"dynatemp_range"`
		DynaTempExponent    float64  `json:"dynatemp_exponent"`
		TopK                int64    `json:"top_k"`
		TopP                float64  `json:"top_p"`
		MinP                float64  `json:"min_p"`
		XTCProbability      float64  `json:"xtc_probability"`
		XTCThreshold        float64  `json:"xtc_threshold"`
		TypicalP            float64  `json:"typical_p"`
		RepeatLastN         int64    `json:"repeat_last_n"`
		RepeatPenalty       float64  `json:"repeat_penalty"`
		PresencePenalty     float64  `json:"presence_penalty"`
		FrequencyPenalty    float64  `json:"frequency_penalty"`
		DryMultiplier       float64  `json:"dry_multiplier"`
		DryBase             float64  `json:"dry_base"`
		DryAllowedLength    int64    `json:"dry_allowed_length"`
		DryPenaltyLastN     int64    `json:"dry_penalty_last_n"`
		DrySequenceBreakers []string `json:"dry_sequence_breakers"`
		Mirostat            int32    `json:"mirostat"`
		MirostatTau         float64  `json:"mirostat_tau"`
		MirostatEta         float64  `json:"mirostat_eta"`
		Stop                []string `json:"stop"`
		MaxTokens           int64    `json:"max_tokens"`
		NKeep               int64    `json:"n_keep"`
		NDiscard            int64    `json:"n_discard"`
		IgnoreEos           bool     `json:"ignore_eos"`
		Stream              bool     `json:"stream"`
		LogitBias           []any    `json:"logit_bias"`
		NProbs              int64    `json:"n_probs"`
		MinKeep             int64    `json:"min_keep"`
		Grammar             string   `json:"grammar"`
		GrammarLazy         bool     `json:"grammar_lazy"`
		GrammarTriggers     []string `json:"grammar_triggers"`
		PreservedTokens     []string `json:"preserved_tokens"`
		ChatFormat          string   `json:"chat_format"`
		ReasoningFormat     string   `json:"reasoning_format"`
		ReasoningInContent  bool     `json:"reasoning_in_content"`
		ThinkingForcedOpen  bool     `json:"thinking_forced_open"`
		Samplers            []string `json:"samplers"`
		SpeculativeNMax     int64    `json:"speculative.n_max"`
		SpeculativeNMin     int64    `json:"speculative.n_min"`
		SpeculativePMin     float64  `json:"speculative.p_min"`
		TimingsPerToken     bool     `json:"timings_per_token"`
		PostSamplingProbs   bool     `json:"post_sampling_probs"`
		Lora                []Lora   `json:"lora"`
		TopNSigma           float64  `json:"top_n_sigma"`
	} `json:"generation_settings"`
	Prompt       string   `json:"prompt"`
	HasNewLine   bool     `json:"has_new_line"`
	Truncated    bool     `json:"truncated"`
	StopType     StopType `json:"stop_type"`
	StoppingWord string   `json:"stopping_word"`
	TokensCached int64    `json:"tokens_cached"`
	Timings      Timings  `json:"timings"`
}

func (c *CompletionResponse) ToResult() (genai.Result, error) {
	out := genai.Result{
		Message: genai.Message{
			Role: genai.Assistant,
			// Mistral Nemo really likes "▁".
			Contents: []genai.Content{{Text: strings.ReplaceAll(c.Content, "\u2581", " ")}},
		},
		Usage: genai.Usage{
			InputTokens:       c.TokensPredicted,
			InputCachedTokens: c.TokensCached,
			OutputTokens:      c.TokensEvaluated,
			FinishReason:      c.StopType.ToFinishReason(),
		},
	}
	return out, nil
}

type StopType string

const (
	StopEOS   StopType = "eos"
	StopLimit StopType = "limit"
	StopWord  StopType = "word"
)

func (s StopType) ToFinishReason() genai.FinishReason {
	switch s {
	case StopEOS:
		return genai.FinishedStop
	case StopLimit:
		return genai.FinishedLength
	case StopWord:
		return genai.FinishedStopSequence
	default:
		if !internal.BeLenient {
			panic(s)
		}
		return genai.FinishReason(s)
	}
}

type Timings struct {
	PromptN             int64   `json:"prompt_n"`
	PromptMS            float64 `json:"prompt_ms"`
	PromptPerTokenMS    float64 `json:"prompt_per_token_ms"`
	PromptPerSecond     float64 `json:"prompt_per_second"`
	PredictedN          int64   `json:"predicted_n"`
	PredictedMS         float64 `json:"predicted_ms"`
	PredictedPerTokenMS float64 `json:"predicted_per_token_ms"`
	PredictedPerSecond  float64 `json:"predicted_per_second"`
}

type CompletionStreamChunkResponse struct {
	// Always
	Index           int64   `json:"index"`
	Content         string  `json:"content"`
	Tokens          []int64 `json:"tokens"`
	Stop            bool    `json:"stop"`
	IDSlot          int64   `json:"id_slot"`
	TokensPredicted int64   `json:"tokens_predicted"`
	TokensEvaluated int64   `json:"tokens_evaluated"`

	// Last message
	Model              string   `json:"model"`
	GenerationSettings any      `json:"generation_settings"`
	Prompt             string   `json:"prompt"`
	HasNewLine         bool     `json:"has_new_line"`
	Truncated          bool     `json:"truncated"`
	StopType           StopType `json:"stop_type"`
	StoppingWord       string   `json:"stopping_word"`
	TokensCached       int64    `json:"tokens_cached"`
	Timings            Timings  `json:"timings"`
}

type applyTemplateRequest struct {
	Messages []Message `json:"messages"`
}

func (a *applyTemplateRequest) Init(opts genai.Options, msgs genai.Messages) error {
	sp := ""
	if v, ok := opts.(*genai.OptionsText); ok {
		sp = v.SystemPrompt
	}
	var errs []error
	unsupported := []string{}
	offset := 0
	if sp != "" {
		offset = 1
	}
	a.Messages = make([]Message, len(msgs)+offset)
	if sp != "" {
		a.Messages[0].Role = "system"
		a.Messages[0].Content = sp
	}
	for i := range msgs {
		if err := a.Messages[i+offset].From(&msgs[i]); err != nil {
			errs = append(errs, fmt.Errorf("message %d: %w", i, err))
		}
	}
	if len(unsupported) > 0 {
		// If we have unsupported features but no other errors, return a continuable error
		if len(errs) == 0 {
			return &genai.UnsupportedContinuableError{Unsupported: unsupported}
		}
		// Otherwise, add the unsupported features to the error list
		errs = append(errs, &genai.UnsupportedContinuableError{Unsupported: unsupported})
	}
	return errors.Join(errs...)
}

func (msg *Message) From(m *genai.Message) error {
	// We don't filter the role here.
	msg.Role = string(m.Role)
	if len(m.Contents) != 1 {
		return fmt.Errorf("expected exactly one block, got %d", len(m.Contents))
	}
	if m.Contents[0].Text != "" {
		msg.Content = m.Contents[0].Text
	} else if m.Contents[0].Document != nil {
		// Check if this is a text/plain document
		mimeType, data, err := m.Contents[0].ReadDocument(10 * 1024 * 1024)
		if err != nil {
			return fmt.Errorf("failed to read document: %w", err)
		}
		if strings.HasPrefix(mimeType, "text/plain") {
			if m.Contents[0].URL != "" {
				return errors.New("text/plain documents must be provided inline, not as a URL")
			}
			msg.Content = string(data)
		} else {
			return fmt.Errorf("llamacpp only supports text/plain documents, got %s", mimeType)
		}
	} else {
		return fmt.Errorf("unsupported content type %v", m.Contents[0])
	}
	if len(m.ToolCalls) != 0 {
		return errors.New("implement tool call results")
	}
	if len(m.ToolCallResults) != 0 {
		return errors.New("implement tool call results")
	}
	return nil
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

//

type ModelHF struct {
	Name         string   `json:"name"`  // Path to the file
	Model        string   `json:"model"` // Path to the file
	ModifiedAt   string   `json:"modified_at"`
	Size         string   `json:"size"`
	Digest       string   `json:"digest"`
	Type         string   `json:"type"` // "model"
	Description  string   `json:"description"`
	Tags         []string `json:"tags"`
	Capabilities []string `json:"capabilities"` // "completion"
	Parameters   string   `json:"parameters"`
	Details      struct {
		ParentModel       string   `json:"parent_model"`
		Format            string   `json:"format"` // "gguf"
		Family            string   `json:"family"`
		Families          []string `json:"families"`
		ParameterSize     string   `json:"parameter_size"`
		QuantizationLevel string   `json:"quantization_level"`
	} `json:"details"`
}

type ModelOpenAI struct {
	ID      string    `json:"id"`     // Path to the file
	Object  string    `json:"object"` // "model"
	Created base.Time `json:"created"`
	OwnedBy string    `json:"owned_by"` // "llamacpp"
	Meta    struct {
		VocabType int64 `json:"vocab_type"` // 1
		NVocab    int64 `json:"n_vocab"`
		NCtxTrain int64 `json:"n_ctx_train"`
		NEmbd     int64 `json:"n_embd"`
		NParams   int64 `json:"n_params"`
		Size      int64 `json:"size"`
	} `json:"meta"`
}

// Model is a synthetic struct combining the information from both ModelHF and ModelOpenAI
type Model struct {
	HF     ModelHF
	OpenAI ModelOpenAI
}

// GetID implements genai.Model.
//
// It returns the base path name otherwise it's overwhelming and breaks our test cases.
func (m *Model) GetID() string {
	return filepath.Base(m.OpenAI.ID)
}

func (m *Model) String() string {
	return m.OpenAI.ID
}

func (m *Model) Context() int64 {
	return m.OpenAI.Meta.NCtxTrain
}

type ModelsResponse struct {
	Models []ModelHF     `json:"models"`
	Object string        `json:"object"` // "list"
	Data   []ModelOpenAI `json:"data"`
}

func (m *ModelsResponse) ToModels() []genai.Model {
	var out []genai.Model
	if len(m.Models) != len(m.Data) {
		panic(fmt.Errorf("unexpected response; got different list sizes for models: %d vs %d", len(m.Models), len(m.Data)).Error())
	}
	for i := range m.Models {
		out = append(out, &Model{HF: m.Models[i], OpenAI: m.Data[i]})
	}
	return out
}

//

// PromptEncoding describes how to encode the prompt.
type PromptEncoding struct {
	// Prompt encoding.
	BeginOfText              string `yaml:"begin_of_text"`
	SystemTokenStart         string `yaml:"system_token_start"`
	SystemTokenEnd           string `yaml:"system_token_end"`
	UserTokenStart           string `yaml:"user_token_start"`
	UserTokenEnd             string `yaml:"user_token_end"`
	AssistantTokenStart      string `yaml:"assistant_token_start"`
	AssistantTokenEnd        string `yaml:"assistant_token_end"`
	ToolsAvailableTokenStart string `yaml:"tools_available_token_start"`
	ToolsAvailableTokenEnd   string `yaml:"tools_available_token_end"`
	ToolCallTokenStart       string `yaml:"tool_call_token_start"`
	ToolCallTokenEnd         string `yaml:"tool_call_token_end"`
	ToolCallResultTokenStart string `yaml:"tool_call_result_token_start"`
	ToolCallResultTokenEnd   string `yaml:"tool_call_result_token_end"`

	_ struct{}
}

// Validate checks for obvious errors in the fields.
func (p *PromptEncoding) Validate() error {
	// TODO: I lied. Please send a PR.
	return nil
}

// TokenPerformance is the performance for the metrics
type TokenPerformance struct {
	Count    int
	Duration time.Duration
}

// Rate is the number of token per second.
func (t *TokenPerformance) Rate() float64 {
	if t.Duration == 0 {
		return 0
	}
	return float64(t.Count) / (float64(t.Duration) / float64(time.Second))
}

// Metrics represents the metrics for the LLM server.
type Metrics struct {
	Prompt             TokenPerformance
	Generated          TokenPerformance
	KVCacheUsage       float64
	KVCacheTokens      int
	RequestsProcessing int
	RequestedPending   int
}

//

type applyTemplateResponse struct {
	Prompt string `json:"prompt"`
}

type ErrorResponse struct {
	Error struct {
		Code    int64
		Message string
		Type    string
	} `json:"error"`
}

func (er *ErrorResponse) String() string {
	return fmt.Sprintf("error %d (%s): %s", er.Error.Code, er.Error.Type, er.Error.Message)
}

//

// Client implements genai.ProviderGen.
type Client struct {
	base.Provider[*ErrorResponse]

	baseURL        string
	completionsURL string
	modelsURL      string
	encoding       *PromptEncoding
}

// New creates a new client to talk to a llama-server instance.
//
// encoding is optional.
//
// baseURL defaults to "http://localhost:8080". It is not a model, so automatic model values
// base.PreferredCheap, base.PreferredGood and base.PreferredSOTA also default to "http://localhost:8080".
// llama-server doesn't have any mean of authentication so there's no API key.
//
// wrapper can be used to throttle outgoing requests, record calls, etc. It defaults to base.DefaultTransport.
func New(baseURL string, encoding *PromptEncoding, wrapper func(http.RoundTripper) http.RoundTripper) (*Client, error) {
	if baseURL == "" || baseURL == base.PreferredCheap || baseURL == base.PreferredGood || baseURL == base.PreferredSOTA {
		baseURL = "http://localhost:8080"
	}
	t := base.DefaultTransport
	if wrapper != nil {
		t = wrapper(t)
	}
	return &Client{
		Provider: base.Provider[*ErrorResponse]{
			ClientJSON: httpjson.Client{
				Lenient: internal.BeLenient,
				Client:  &http.Client{Transport: &roundtrippers.RequestID{Transport: t}},
			},
		},
		baseURL:        baseURL,
		completionsURL: baseURL + "/completion",
		modelsURL:      baseURL + "/v1/models",
		encoding:       encoding,
	}, nil
}

func (c *Client) Name() string {
	return "llamacpp"
}

func (c *Client) Scoreboard() genai.Scoreboard {
	return Scoreboard
}

func (c *Client) ListModels(ctx context.Context) ([]genai.Model, error) {
	// https://api-docs.deepseek.com/api/list-models
	return base.ListModels[*ErrorResponse, *ModelsResponse](ctx, &c.Provider, c.modelsURL)
}

func (c *Client) GenSync(ctx context.Context, msgs genai.Messages, opts genai.Options) (genai.Result, error) {
	// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-completion-given-a-prompt-it-returns-the-predicted-completion
	// Doc mentions Cache:true causes non-determinism even if a non-zero seed is
	// specified. Disable if it becomes a problem.
	if err := msgs.Validate(); err != nil {
		return genai.Result{}, err
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return genai.Result{}, err
		}
		if m := opts.Modality(); m != genai.ModalityText {
			return genai.Result{}, fmt.Errorf("modality %s not supported", m)
		}
	}
	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return genai.Result{}, fmt.Errorf("message #%d content #%d: field Opaque not supported", i, j)
			}
		}
		for j, tool := range msg.ToolCalls {
			if len(tool.Opaque) != 0 {
				return genai.Result{}, fmt.Errorf("message #%d tool call #%d: field Opaque not supported", i, j)
			}
		}
	}
	rpcin := CompletionRequest{CachePrompt: true}
	if err := rpcin.Init(msgs, opts, ""); err != nil {
		return genai.Result{}, err
	}
	if err := c.initPrompt(ctx, &rpcin, opts, msgs); err != nil {
		return genai.Result{}, err
	}
	rpcout := CompletionResponse{}
	if err := c.CompletionRaw(ctx, &rpcin, &rpcout); err != nil {
		return genai.Result{}, fmt.Errorf("failed to get llama server response: %w", err)
	}
	return rpcout.ToResult()
}

func (c *Client) CompletionRaw(ctx context.Context, in *CompletionRequest, out *CompletionResponse) error {
	// TODO: Distinguish between completion and chat. Chat is completion with the template applied.
	in.Stream = false
	return c.DoRequest(ctx, "POST", c.completionsURL, in, out)
}

func (c *Client) GenStream(ctx context.Context, msgs genai.Messages, chunks chan<- genai.ContentFragment, opts genai.Options) (genai.Result, error) {
	result := genai.Result{}
	if err := msgs.Validate(); err != nil {
		return result, err
	}
	if opts != nil {
		if err := opts.Validate(); err != nil {
			return result, err
		}
		if m := opts.Modality(); m != genai.ModalityText {
			return genai.Result{}, fmt.Errorf("modality %s not supported", m)
		}
	}

	for i, msg := range msgs {
		for j, content := range msg.Contents {
			if len(content.Opaque) != 0 {
				return result, fmt.Errorf("message #%d content #%d: Opaque field not supported", i, j)
			}
		}
		for j, tool := range msg.ToolCalls {
			if len(tool.Opaque) != 0 {
				return result, fmt.Errorf("message #%d tool call #%d: field Opaque not supported", i, j)
			}
		}
	}

	in := CompletionRequest{}
	var continuableErr error
	if err := in.Init(msgs, opts, ""); err != nil {
		// If it's an UnsupportedContinuableError, we can continue
		if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
			// Store the error to return later if no other error occurs
			continuableErr = uce
			// Otherwise log the error but continue
		} else {
			return result, err
		}
	}
	if err := c.initPrompt(ctx, &in, opts, msgs); err != nil {
		return result, err
	}
	ch := make(chan CompletionStreamChunkResponse)
	eg, ctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		return processCompletionsStreamPackets(ch, chunks, &result)
	})
	err := c.CompletionStreamRaw(ctx, &in, ch)
	close(ch)
	if err2 := eg.Wait(); err2 != nil {
		err = err2
	}
	// Return the continuable error if no other error occurred
	if err == nil && continuableErr != nil {
		return result, continuableErr
	}
	return result, err
}

func (c *Client) CompletionStreamRaw(ctx context.Context, in *CompletionRequest, out chan<- CompletionStreamChunkResponse) error {
	in.Stream = true
	resp, err := c.ClientJSON.Request(ctx, "POST", c.completionsURL, nil, in)
	if err != nil {
		return fmt.Errorf("failed to get llama server response: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return c.DecodeError(c.completionsURL, resp)
	}
	return sse.Process(resp.Body, out, nil, c.ClientJSON.Lenient)
}

func (c *Client) GetHealth(ctx context.Context) (string, error) {
	msg, err := c.GetHealthRaw(ctx)
	return msg.Status, err
}

func (c *Client) GetHealthRaw(ctx context.Context) (HealthResponse, error) {
	msg := HealthResponse{}
	if err := c.DoRequest(ctx, "GET", c.baseURL+"/health", nil, &msg); err != nil {
		return msg, fmt.Errorf("failed to get health response: %w", err)
	}
	return msg, nil
}

// GetMetrics retrieves the performance statistics from the server.
func (c *Client) GetMetrics(ctx context.Context, m *Metrics) error {
	// TODO: Generalize.
	req, err := http.NewRequestWithContext(ctx, "GET", c.baseURL+"/metrics", nil)
	if err != nil {
		return fmt.Errorf("failed to create HTTP request: %w", err)
	}
	// This is not a JSON response.
	resp, err := c.ClientJSON.Client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to get metrics response: %w", err)
	}
	b, err := io.ReadAll(resp.Body)
	if err2 := resp.Body.Close(); err == nil {
		err = err2
	}
	if err != nil {
		return fmt.Errorf("failed to get metrics response: %w", err)
	}
	// We hardcode things here since we know which server we are talking to. See
	// the commit history if you want the generic prometheus style data.
	for l := range strings.SplitSeq(strings.TrimSpace(string(b)), "\n") {
		if strings.HasPrefix(l, "#") {
			continue
		}
		parts := strings.Split(l, " ")
		if len(parts) != 2 {
			return fmt.Errorf("failed to parse line %q: %w", l, err)
		}
		// Search for these strings in
		// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server.cpp
		f := 0.0
		if parts[1] == "nan" || parts[1] == "-nan" {
			f = math.NaN()
		} else {
			if f, err = strconv.ParseFloat(parts[1], 64); err != nil {
				return fmt.Errorf("failed to parse line %q: %w", l, err)
			}
		}
		i, _ := strconv.Atoi(parts[1])
		switch parts[0] {
		case "llamacpp:prompt_tokens_total":
			m.Prompt.Count = i
		case "llamacpp:prompt_seconds_total":
			m.Prompt.Duration = time.Duration(f*1000) * time.Millisecond
		case "llamacpp:tokens_predicted_total":
			m.Generated.Count = i
		case "llamacpp:tokens_predicted_seconds_total":
			m.Generated.Duration = time.Duration(f*1000) * time.Millisecond
		case "llamacpp:prompt_tokens_seconds", "llamacpp:predicted_tokens_seconds":
			// Ignore.
		case "llamacpp:kv_cache_usage_ratio":
			m.KVCacheUsage = f
		case "llamacpp:kv_cache_tokens":
			m.KVCacheTokens = i
		case "llamacpp:requests_processing":
			m.RequestsProcessing = i
		case "llamacpp:requests_deferred":
			m.RequestedPending = i
		case "llamacpp:n_decode_total":
		case "llamacpp:n_busy_slots_per_decode":
		default:
			return fmt.Errorf("unknown metric %q", l)
		}
	}
	return nil
}

func (c *Client) ModelID() string {
	return ""
}

func (c *Client) initPrompt(ctx context.Context, in *CompletionRequest, opts genai.Options, msgs genai.Messages) error {
	if c.encoding == nil {
		// Use the server to convert the OpenAI style format into a templated form.
		in2 := applyTemplateRequest{}
		var continuableErr error
		if err := in2.Init(opts, msgs); err != nil {
			// If it's an UnsupportedContinuableError, we can continue
			if uce, ok := err.(*genai.UnsupportedContinuableError); ok {
				// Store the error to return later if no other error occurs
				continuableErr = uce
				// Otherwise log the error but continue
			} else {
				return err
			}
		}
		out := applyTemplateResponse{}
		if err := c.DoRequest(ctx, "POST", c.baseURL+"/apply-template", &in2, &out); err != nil {
			return err
		}
		in.Prompt = out.Prompt
		// Return the continuable error if no other error occurred
		if continuableErr != nil {
			return continuableErr
		}
		return nil
	}

	in.Prompt = c.encoding.BeginOfText
	for _, m := range msgs {
		switch m.Role {
		/* TODO
		case genai.AvailableTools:
			in.Prompt += c.encoding.ToolsAvailableTokenStart + m.Text + c.encoding.ToolsAvailableTokenEnd
		*/
		case "system":
			for _, b := range m.Contents {
				in.Prompt += c.encoding.SystemTokenStart + b.Text + c.encoding.SystemTokenEnd
			}
		case genai.User:
			for _, b := range m.Contents {
				in.Prompt += c.encoding.UserTokenStart + b.Text + c.encoding.UserTokenEnd
			}
			// in.Prompt += c.encoding.ToolCallResultTokenStart + m.Text + c.encoding.ToolCallResultTokenEnd
		case genai.Assistant:
			for _, b := range m.Contents {
				in.Prompt += c.encoding.AssistantTokenStart + b.Text + c.encoding.AssistantTokenEnd
			}
			// in.Prompt += c.encoding.ToolCallTokenStart + m.Text + c.encoding.ToolCallTokenEnd
		case genai.Computer:
			fallthrough
		default:
			return fmt.Errorf("unexpected role %q", m.Role)
		}
	}
	return nil
}

func processCompletionsStreamPackets(ch <-chan CompletionStreamChunkResponse, chunks chan<- genai.ContentFragment, result *genai.Result) error {
	for msg := range ch {
		if msg.Timings.PredictedN != 0 {
			result.InputTokens = msg.Timings.PromptN
			result.OutputTokens = msg.Timings.PredictedN
		}
		if msg.StopType != "" {
			result.FinishReason = msg.StopType.ToFinishReason()
		}
		f := genai.ContentFragment{TextFragment: msg.Content}
		if !f.IsZero() {
			if err := result.Accumulate(f); err != nil {
				return err
			}
			chunks <- f
		}
	}
	return nil
}

var (
	_ genai.Provider           = &Client{}
	_ genai.ProviderGen        = &Client{}
	_ genai.ProviderScoreboard = &Client{}
)
