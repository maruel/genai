// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Section row types: chat, embed, systemone and realtime.

package scoreboard

import (
	"errors"
	"fmt"
	"maps"
	"slices"
)

// ScoreRow is what the code that handles any section needs to know about a row.
type ScoreRow interface {
	// Base returns the shared fields.
	Base() *Row
	// Reasoning reports the reasoning mode, always false where the suite has none.
	Reasoning() bool
	// Outputs reports the output modalities, derived where the suite has none.
	Outputs() map[Modality]ModalCapability
	// Untested reports whether the row holds no measurement yet.
	Untested() bool
	// Key returns the identity of the row for model, within its section.
	Key(model string) Key
	// SetReasoning sets the reasoning mode.
	SetReasoning(reason bool)
	// SetTokens sets the delimiters of an inline reasoning reply.
	SetTokens(start, end string)
	// Tokens returns the delimiters of an inline reasoning reply, empty where the
	// suite has none.
	Tokens() (start, end string)
}

// Row holds the fields every section row shares.
type Row struct {
	// Comments are notes about the row. For example, if a model is known to be
	// bugged, deprecated, expensive, etc.
	Comments string `json:"comments,omitzero"`
	// Models is a *non exhaustive* list of models that support the row. It can't be
	// exhaustive since providers continuously release new models. It is still
	// valuable to use the first value. Required.
	Models []string `json:"models"`

	// These mean that the model is automatically selected. There can be multiple
	// SOTA models, one per modality (e.g. text output vs image output). They must
	// be first. Models must be a list of one model in this case.
	SOTA  bool `json:"sota,omitzero"`
	Good  bool `json:"good,omitzero"`
	Cheap bool `json:"cheap,omitzero"`

	// In describes what the model accepts, and how.
	In map[Modality]ModalCapability `json:"in,omitzero,omitempty"`

	// Refused records why the model declines this suite. It is a result: the probes
	// asked and the model said no. An empty value means the model was not asked, or
	// answered.
	Refused string `json:"refused,omitzero"`

	// Wire records how the provider exchanged bytes with the model. An empty value
	// means the wire of the section, so a provider that answers a suite over another
	// wire writes it on every row of that section.
	Wire Wire `json:"wire,omitzero"`

	_ struct{}
}

func (r *Row) validate() error {
	if len(r.Models) == 0 {
		return errors.New("scenario must have at least one model")
	}
	for k := range r.In {
		if err := k.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// WireOf returns the wire the row ran over.
func (r *Row) WireOf(fallback Wire) Wire {
	if r.Wire != "" {
		return r.Wire
	}
	return fallback
}

// Reasoned adds the reasoning mode of a suite that can think.
type Reasoned struct {
	Row
	// Reason means that the model does either explicit chain-of-thought or hidden
	// reasoning. For some providers, this is controlled via OptionsText. For some
	// models (like Qwen3), a token "/no_think" or "/think" is used to control.
	Reason bool `json:"reason,omitzero"`

	_ struct{}
}

// Chat declares a conversational suite, measured through GenSync and GenStream.
type Chat struct {
	Reasoned
	// Out describes what the model returns.
	Out map[Modality]ModalCapability `json:"out,omitzero,omitempty"`

	// ReasoningTokenStart and ReasoningTokenEnd delimit an inline reasoning reply.
	// They are only set on explicit inline reasoning models, which often use
	// <think> and </think>.
	ReasoningTokenStart string `json:"reasoningTokenStart,omitzero"`
	ReasoningTokenEnd   string `json:"reasoningTokenEnd,omitzero"`

	// GenSync declares features supported when using Provider.GenSync.
	// An empty value requests smoke-test qualification before modalities are known.
	GenSync *Functionality `json:"GenSync,omitzero,omitempty"`
	// GenStream declares features supported when using Provider.GenStream.
	// An empty value requests smoke-test qualification before modalities are known.
	GenStream *Functionality `json:"GenStream,omitzero,omitempty"`

	_ struct{}
}

// Base implements [ScoreRow].
func (r *Chat) Base() *Row { return &r.Row }

// Reasoning implements [ScoreRow].
func (r *Chat) Reasoning() bool { return r.Reason }

// Outputs implements [ScoreRow].
func (r *Chat) Outputs() map[Modality]ModalCapability { return r.Out }

// Untested implements [ScoreRow].
func (r *Chat) Untested() bool {
	return r.GenSync == nil && r.GenStream == nil && len(r.In) == 0 && len(r.Out) == 0 && r.Refused == ""
}

// Key implements [ScoreRow].
func (r *Chat) Key(model string) Key {
	return Key{Model: model, Reason: r.Reason, Wire: r.WireOf(WireHTTP)}
}

// Tokens implements [ScoreRow].
func (r *Chat) Tokens() (string, string) { return r.ReasoningTokenStart, r.ReasoningTokenEnd }

// SetReasoning implements [ScoreRow].
func (r *Chat) SetReasoning(reason bool) { r.Reason = reason }

// SetTokens implements [ScoreRow].
func (r *Chat) SetTokens(start, end string) { r.ReasoningTokenStart, r.ReasoningTokenEnd = start, end }

// Validate returns an error if the Chat is not correctly configured.
func (r *Chat) Validate() error {
	if err := r.validate(); err != nil {
		return err
	}
	if (len(r.In) == 0) != (len(r.Out) == 0) {
		return errors.New("scenario must have either both or none of In or Out")
	}
	if len(r.In) == 0 {
		for _, f := range []*Functionality{r.GenSync, r.GenStream} {
			if f != nil && *f != (Functionality{}) {
				return errors.New("scenario with functionality must define input and output modalities")
			}
		}
	}
	for _, f := range []*Functionality{r.GenSync, r.GenStream} {
		if f != nil {
			if err := f.Validate(); err != nil {
				return err
			}
		}
	}
	return nil
}

// Embed declares a text and media embedding suite, measured through Embed.
//
// It declares no output modality: an embedding answer is a vector of the measured
// dimensions.
type Embed struct {
	Row
	// Embed declares embedding support. An empty value requests qualification.
	Embed *EmbeddingFunctionality `json:"Embed,omitzero,omitempty"`

	_ struct{}
}

// Base implements [ScoreRow].
func (r *Embed) Base() *Row { return &r.Row }

// Reasoning implements [ScoreRow].
func (r *Embed) Reasoning() bool { return false }

// Outputs implements [ScoreRow].
func (r *Embed) Outputs() map[Modality]ModalCapability {
	return map[Modality]ModalCapability{ModalityEmbedding: {Inline: true}}
}

// Untested implements [ScoreRow].
func (r *Embed) Untested() bool { return r.Embed == nil && len(r.In) == 0 && r.Refused == "" }

// Key implements [ScoreRow].
func (r *Embed) Key(model string) Key {
	return Key{Model: model, Wire: r.WireOf(WireHTTP)}
}

// Tokens implements [ScoreRow].
func (r *Embed) Tokens() (string, string) { return "", "" }

// SetReasoning implements [ScoreRow].
func (r *Embed) SetReasoning(bool) {}

// SetTokens implements [ScoreRow].
func (r *Embed) SetTokens(string, string) {}

// Validate returns an error if the Embed is not correctly configured.
func (r *Embed) Validate() error {
	if err := r.validate(); err != nil {
		return err
	}
	if r.Embed == nil {
		return nil
	}
	if err := r.Embed.Validate(); err != nil {
		return err
	}
	if r.Embed.Dimensions == 0 {
		if len(r.In) != 0 || r.Embed.RequestedDimensions != nil || r.Embed.ReportTokenUsage != False {
			return errors.New("unqualified embedding functionality must be empty")
		}
		return nil
	}
	if !r.In[ModalityText].Inline {
		return errors.New("embedding functionality requires inline text input")
	}
	return nil
}

// SystemOne declares a typed decision suite.
type SystemOne struct {
	Row
	// Out describes the decision the model returns.
	Out map[Modality]ModalCapability `json:"out,omitzero,omitempty"`
	// SystemOne declares typed decision features. An empty value requests
	// qualification.
	SystemOne *DecisionFunctionality `json:"SystemOne,omitzero,omitempty"`

	_ struct{}
}

// Base implements [ScoreRow].
func (r *SystemOne) Base() *Row { return &r.Row }

// Reasoning implements [ScoreRow].
func (r *SystemOne) Reasoning() bool { return false }

// Outputs implements [ScoreRow].
func (r *SystemOne) Outputs() map[Modality]ModalCapability { return r.Out }

// Untested implements [ScoreRow].
func (r *SystemOne) Untested() bool {
	return r.SystemOne == nil && len(r.In) == 0 && len(r.Out) == 0 && r.Refused == ""
}

// Key implements [ScoreRow].
func (r *SystemOne) Key(model string) Key {
	return Key{Model: model, Wire: r.WireOf(WireHTTP)}
}

// Tokens implements [ScoreRow].
func (r *SystemOne) Tokens() (string, string) { return "", "" }

// SetReasoning implements [ScoreRow].
func (r *SystemOne) SetReasoning(bool) {}

// SetTokens implements [ScoreRow].
func (r *SystemOne) SetTokens(string, string) {}

// Validate returns an error if the SystemOne is not correctly configured.
func (r *SystemOne) Validate() error {
	if err := r.validate(); err != nil {
		return err
	}
	if (len(r.In) == 0) != (len(r.Out) == 0) {
		return errors.New("scenario must have either both or none of In or Out")
	}
	if r.SystemOne == nil {
		return nil
	}
	if *r.SystemOne != (DecisionFunctionality{}) && len(r.In) == 0 {
		return errors.New("scenario with decision functionality must define input and output modalities")
	}
	return r.SystemOne.Validate()
}

// Realtime declares a bidirectional session suite.
type Realtime struct {
	Reasoned
	// Out describes what the session can return.
	Out map[Modality]ModalCapability `json:"out,omitzero,omitempty"`
	// Realtime declares bidirectional realtime session support. An empty value
	// requests qualification through genai.Provider.ConnectRealtime.
	Realtime *RealtimeFunctionality `json:"Realtime,omitzero,omitempty"`

	_ struct{}
}

// Base implements [ScoreRow].
func (r *Realtime) Base() *Row { return &r.Row }

// Reasoning implements [ScoreRow].
func (r *Realtime) Reasoning() bool { return r.Reason }

// Outputs implements [ScoreRow].
func (r *Realtime) Outputs() map[Modality]ModalCapability { return r.Out }

// Untested implements [ScoreRow].
func (r *Realtime) Untested() bool {
	return r.Realtime == nil && len(r.In) == 0 && len(r.Out) == 0 && r.Refused == ""
}

// Key implements [ScoreRow].
func (r *Realtime) Key(model string) Key {
	return Key{Model: model, Reason: r.Reason, Wire: r.WireOf(WireSession)}
}

// Tokens implements [ScoreRow].
func (r *Realtime) Tokens() (string, string) { return "", "" }

// SetReasoning implements [ScoreRow].
func (r *Realtime) SetReasoning(reason bool) { r.Reason = reason }

// SetTokens implements [ScoreRow].
func (r *Realtime) SetTokens(string, string) {}

// Validate returns an error if the Realtime is not correctly configured.
func (r *Realtime) Validate() error {
	if err := r.validate(); err != nil {
		return err
	}
	if (len(r.In) == 0) != (len(r.Out) == 0) {
		return errors.New("scenario must have either both or none of In or Out")
	}
	if r.Realtime == nil {
		return nil
	}
	if !r.Realtime.empty() && len(r.In) == 0 {
		return errors.New("realtime functionality requires input and output modalities")
	}
	return r.Realtime.Validate()
}

// Score is a snapshot of the capabilities of the provider. These are smoke tested
// to confirm the accuracy.
//
// A scoreboard holds one section per probe suite, and one family fills each:
// Chat holds GenSync and GenStream, Embed holds embedding, SystemOne holds typed
// decisions, and Realtime holds sessions. The section is the suite, so a row carries
// the payloads of its section alone; a row without a payload is a queue entry for
// models that suite has not measured, unless it records a refusal.
type Score struct {
	// Warnings lists concerns the user should be aware of.
	Warnings []string `json:"warnings,omitzero,omitempty"`
	// Country where the provider is based, e.g. "US", "CN", "EU". Two exceptions:
	// "Local" for local and "N/A" for pure routers.
	Country string `json:"country"`
	// DashboardURL is the URL to the provider's dashboard, if available.
	DashboardURL string `json:"dashboardURL"`

	Chat      []Chat      `json:"chat,omitzero,omitempty"`
	Embed     []Embed     `json:"embed,omitzero,omitempty"`
	SystemOne []SystemOne `json:"systemone,omitzero,omitempty"`
	Realtime  []Realtime  `json:"realtime,omitzero,omitempty"`

	_ struct{}
}

// Rows returns the shared fields of every row, in section order.
func (s *Score) Rows() []Row {
	out := make([]Row, 0, len(s.Chat)+len(s.Embed)+len(s.SystemOne)+len(s.Realtime))
	for i := range s.Chat {
		out = append(out, s.Chat[i].Row)
	}
	for i := range s.Embed {
		out = append(out, s.Embed[i].Row)
	}
	for i := range s.SystemOne {
		out = append(out, s.SystemOne[i].Row)
	}
	for i := range s.Realtime {
		out = append(out, s.Realtime[i].Row)
	}
	return out
}

// Models returns every model the scoreboard names, with the reasoning mode it
// records. A model that holds rows in several sections appears once per mode.
func (s Score) Models() []Model { //nolint:gocritic // hugeParam: the sections are read, not copied.
	var out []Model
	seen := map[Model]struct{}{}
	add := func(rows []Model) {
		for _, m := range rows {
			if _, ok := seen[m]; ok {
				continue
			}
			seen[m] = struct{}{}
			out = append(out, m)
		}
	}
	for i := range s.Chat {
		for _, m := range s.Chat[i].Models {
			add([]Model{{Model: m, Reason: s.Chat[i].Reason}})
		}
	}
	for i := range s.Embed {
		for _, m := range s.Embed[i].Models {
			add([]Model{{Model: m}})
		}
	}
	for i := range s.SystemOne {
		for _, m := range s.SystemOne[i].Models {
			add([]Model{{Model: m}})
		}
	}
	for i := range s.Realtime {
		for _, m := range s.Realtime[i].Models {
			add([]Model{{Model: m, Reason: s.Realtime[i].Reason}})
		}
	}
	return out
}

// Embedding returns measured embedding functionality for model, or nil when
// support is unknown. Missing rows and empty qualification markers are unknown;
// they do not establish unsupported model behavior.
func (s *Score) Embedding(model string) *EmbeddingFunctionality {
	for i := range s.Embed {
		r := &s.Embed[i]
		if r.Embed != nil && r.Embed.Dimensions > 0 && slices.Contains(r.Models, model) {
			return r.Embed
		}
	}
	return nil
}

// Validate returns an error if the Score is not correctly configured.
func (s *Score) Validate() error {
	if err := validateSection[Chat](s.Chat); err != nil {
		return fmt.Errorf("chat: %w", err)
	}
	if err := validateSection[Embed](s.Embed); err != nil {
		return fmt.Errorf("embed: %w", err)
	}
	if err := validateSection[SystemOne](s.SystemOne); err != nil {
		return fmt.Errorf("systemone: %w", err)
	}
	return validateSection[Realtime](s.Realtime)
}

// rowPointer is the pointer type of a section row. The methods of a row take a
// pointer, so a generic helper over a section names both.
type rowPointer[T any] interface {
	*T
	ScoreRow
}

// validateSection checks one section: its rows, their keys and their tiers.
func validateSection[T any, PT interface {
	rowPointer[T]
	Validate() error
}](rows []T) error {
	seen := make(map[Key]struct{})
	for i := range rows {
		r := PT(&rows[i])
		if err := r.Validate(); err != nil {
			return err
		}
		for _, model := range r.Base().Models {
			key := r.Key(model)
			if _, ok := seen[key]; ok {
				return fmt.Errorf("duplicate model in scoreboard: %q (reason=%v, wire=%v)", model, key.Reason, key.Wire)
			}
			seen[key] = struct{}{}
		}
	}
	return validateTiers[T, PT](rows)
}

// validateTiers checks that each output modality picks at most one SOTA, Good and
// Cheap model, in that order.
func validateTiers[T any, PT interface {
	*T
	ScoreRow
}](rows []T) error {
	groups := make(map[Modality][]int)
	for i := range rows {
		out := PT(&rows[i]).Outputs()
		if len(out) == 0 {
			groups[""] = append(groups[""], i)
			continue
		}
		for modality := range out {
			groups[modality] = append(groups[modality], i)
		}
	}
	for modality, indices := range groups {
		name := string(modality)
		if name == "" {
			name = "text"
		}
		count := map[string]int{}
		first := map[string]int{"sota": -1, "good": -1, "cheap": -1}
		for _, i := range indices {
			r := PT(&rows[i]).Base()
			if len(r.Models) == 0 {
				continue
			}
			for tier, set := range map[string]bool{"sota": r.SOTA, "good": r.Good, "cheap": r.Cheap} {
				if set {
					count[tier]++
					if first[tier] < 0 {
						first[tier] = i
					}
				}
			}
		}
		for _, tier := range []string{"sota", "good", "cheap"} {
			if count[tier] > 1 {
				return fmt.Errorf("multiple %s models marked for modality %q (count: %d)", tier, name, count[tier])
			}
		}
		if count["sota"] > 0 && first["sota"] != indices[0] {
			return fmt.Errorf("sota model %q for modality %q should be first in that modality's scenarios (at position %d)",
				PT(&rows[first["sota"]]).Base().Models[0], name, first["sota"])
		}
		if first["good"] >= 0 && first["cheap"] >= 0 && first["good"] > first["cheap"] {
			return fmt.Errorf("good model comes after cheap model for modality %q (good at %d, cheap at %d)",
				name, first["good"], first["cheap"])
		}
	}
	return nil
}

// SortRows sorts one section: SOTA, then Good, then Cheap, then the rest; measured
// rows before queue entries; then grouped by model with reasoning first.
func SortRows[T any, PT interface {
	*T
	ScoreRow
}](rows []T) {
	slices.SortStableFunc(rows, func(a, b T) int {
		return compareRows(PT(&a), PT(&b))
	})
}

func compareRows[T ScoreRow](a, b T) int {
	if c := preference(a.Base()) - preference(b.Base()); c != 0 {
		return c
	}
	if a.Untested() != b.Untested() {
		if a.Untested() {
			return 1
		}
		return -1
	}
	am, bm := firstModel(a.Base()), firstModel(b.Base())
	if am != bm {
		if am < bm {
			return -1
		}
		return 1
	}
	if a.Reasoning() != b.Reasoning() {
		if a.Reasoning() {
			return -1
		}
		return 1
	}
	return 0
}

func preference(r *Row) int {
	switch {
	case r.SOTA:
		return 0
	case r.Good:
		return 1
	case r.Cheap:
		return 2
	}
	return 999
}

func firstModel(r *Row) string {
	if len(r.Models) > 0 {
		return r.Models[0]
	}
	return ""
}

// ConsolidateQueue merges the queue entries of one section that share their notes.
//
// Rows with matching Comments are merged, with their models combined and sorted.
// A row with a preference flag is kept alone.
func ConsolidateQueue[T any, PT interface {
	*T
	ScoreRow
}](rows []T) []T {
	// Queue entries carry no payload, so they consolidate by their notes and wire
	// alone: a queue entry waits for one wire.
	byKey := map[string]int{}
	var out []T
	for i := range rows {
		r := PT(&rows[i])
		if !r.Untested() {
			continue
		}
		if r.Base().SOTA || r.Base().Good || r.Base().Cheap {
			out = append(out, rows[i])
			continue
		}
		key := fmt.Sprintf("%s|%v|%v", r.Base().Comments, r.Reasoning(), r.Base().Wire)
		if idx, found := byKey[key]; found {
			existing := PT(&out[idx])
			set := make(map[string]struct{}, len(existing.Base().Models)+len(r.Base().Models))
			for _, m := range existing.Base().Models {
				set[m] = struct{}{}
			}
			for _, m := range r.Base().Models {
				set[m] = struct{}{}
			}
			existing.Base().Models = slices.Sorted(maps.Keys(set))
		} else {
			out = append(out, rows[i])
			byKey[key] = len(out) - 1
		}
	}
	return out
}
