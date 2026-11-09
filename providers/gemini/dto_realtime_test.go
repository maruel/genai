// Copyright 2026 Marc-Antoine Ruel. All rights reserved.
// Use of this source code is governed under the Apache License, Version 2.0
// that can be found in the LICENSE file.

// Tests for Gemini Live server message decoding.

package gemini

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/maruel/genai"
)

// decodeLiveStrict mirrors the strict decoding the session applies to server frames.
func decodeLiveStrict(data string, out any) error {
	d := json.NewDecoder(strings.NewReader(data))
	d.DisallowUnknownFields()
	return d.Decode(out)
}

func TestLiveServerMessage(t *testing.T) {
	t.Parallel()
	// Every field of every modeled server message, as the Live endpoint sends them.
	data := `{
	"setupComplete":{"sessionId":"live-test","voiceConsentSignature":{"signature":"c2ln"}},
	"serverContent":{
		"modelTurn":{"role":"model","parts":[
			{"thought":true,"thoughtSignature":"c2ln","text":"thinking"},
			{"text":"answer"},
			{"inlineData":{"mimeType":"audio/pcm;rate=24000","data":"AQACAA=="}},
			{"functionCall":{"id":"call-1","name":"get_weather","args":{"city":"Montreal"},"thoughtSignature":"c2ln"}}
		]},
		"turnComplete":true,
		"turnCompleteReason":"NEED_MORE_INPUT",
		"interrupted":true,
		"generationComplete":true,
		"waitingForInput":true,
		"interactionStatus":"IN_PROGRESS",
		"inputTranscription":{"text":"hello","languageCode":"en-US","speakerLabel":"spk_1","startOffset":"0.5s","endOffset":"1.5s","words":[{"word":"hello","startOffset":"0.5s","endOffset":"1s"}]},
		"interimInputTranscription":{"text":"hel"},
		"outputTranscription":{"text":"world","finished":true},
		"groundingMetadata":{
			"webSearchQueries":["weather Montreal"],
			"groundingChunks":[{"web":{"uri":"https://example.com/a","title":"A"}}],
			"groundingSupports":[{"groundingChunkIndices":[0],"confidenceScores":[0.5],"segment":{"partIndex":0,"startIndex":0,"endIndex":6,"text":"answer"}}],
			"searchEntryPoint":{"renderedContent":"<div></div>","sdkBlob":"e30="},
			"retrievalMetadata":{"googleSearchDynamicRetrievalScore":0.25}
		},
		"urlContextMetadata":{"urlMetadata":[{"retrievedUrl":"https://example.com/b","urlRetrievalStatus":"URL_RETRIEVAL_STATUS_SUCCESS"}]}
	},
	"toolCall":{"functionCalls":[{"id":"call-1","name":"get_weather","args":{"city":"Montreal"},"thoughtSignature":"c2ln"}]},
	"toolCallCancellation":{"ids":["call-2"]},
	"usageMetadata":{
		"promptTokenCount":1,
		"cachedContentTokenCount":2,
		"responseTokenCount":3,
		"thoughtsTokenCount":4,
		"totalTokenCount":10,
		"toolUsePromptTokenCount":5,
		"serviceTier":"standard",
		"trafficType":"ON_DEMAND_FLEX",
		"promptTokensDetails":[{"modality":"AUDIO","tokenCount":1}],
		"cacheTokensDetails":[{"modality":"TEXT","tokenCount":2}],
		"responseTokensDetails":[{"modality":"TEXT","tokenCount":3}],
		"toolUsePromptTokensDetails":[{"modality":"TEXT","tokenCount":5}]
	},
	"goAway":{"timeLeft":"5.5s"},
	"sessionResumptionUpdate":{"newHandle":"resume","resumable":true,"lastConsumedClientMessageIndex":"12"},
	"voiceActivityDetectionSignal":{"vadSignalType":"VAD_SIGNAL_TYPE_SOS"},
	"voiceActivity":{"type":"ACTIVITY_END","audioOffset":"1.5s"},
	"error":{"code":400,"message":"bad","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/x"}]}
}`
	var msg LiveServerMessage
	if err := decodeLiveStrict(data, &msg); err != nil {
		t.Fatal(err)
	}

	if msg.SetupComplete.SessionID != "live-test" || msg.SetupComplete.VoiceConsentSignature.Signature != "c2ln" {
		t.Errorf("setup complete: %+v", msg.SetupComplete)
	}
	c := msg.ServerContent
	if c == nil {
		t.Fatal("missing server content")
	}
	if !c.TurnComplete || !c.Interrupted || !c.GenerationComplete || !c.WaitingForInput || c.TurnCompleteReason != LiveTurnCompleteNeedMoreInput || c.InteractionStatus != InteractionStatusInProgress {
		t.Errorf("content flags: %+v", *c)
	}
	if len(c.ModelTurn.Parts) != 4 || !c.ModelTurn.Parts[0].Thought || c.ModelTurn.Parts[1].Text != "answer" || c.ModelTurn.Parts[2].InlineData.MIMEType != "audio/pcm;rate=24000" || c.ModelTurn.Parts[3].FunctionCall.Name != "get_weather" {
		t.Errorf("model turn: %+v", c.ModelTurn)
	}
	if c.InputTranscription.LanguageCode != "en-US" || c.InputTranscription.Words[0].Word != "hello" || c.InterimInputTranscription.Text != "hel" || !c.OutputTranscription.Finished {
		t.Errorf("transcriptions: %+v", *c)
	}
	if len(c.GroundingMetadata.GroundingChunks) != 1 || c.GroundingMetadata.GroundingChunks[0].Web.URI != "https://example.com/a" || c.UrlContextMetadata.UrlMetadata[0].UrlRetrievalStatus != "URL_RETRIEVAL_STATUS_SUCCESS" {
		t.Errorf("citations: %+v %+v", c.GroundingMetadata, c.UrlContextMetadata)
	}
	if len(msg.ToolCall.FunctionCalls) != 1 || msg.ToolCallCancellation.IDs[0] != "call-2" {
		t.Errorf("tool calls: %+v %+v", msg.ToolCall, msg.ToolCallCancellation)
	}
	if msg.UsageMetadata.ServiceTier != LiveServiceTierStandard || msg.UsageMetadata.TrafficType != LiveTrafficTypeOnDemandFlex || msg.UsageMetadata.PromptTokensDetails[0].Modality != ModalityAudio {
		t.Errorf("usage: %+v", *msg.UsageMetadata)
	}
	if msg.GoAway.TimeLeft != "5.5s" || msg.SessionResumptionUpdate.LastConsumedClientMessageIndex != 12 || !msg.SessionResumptionUpdate.Resumable {
		t.Errorf("session control: %+v %+v", msg.GoAway, msg.SessionResumptionUpdate)
	}
	if msg.VoiceActivityDetectionSignal.VADSignalType != LiveVADSignalSOS || msg.VoiceActivity.Type != LiveVoiceActivityEnd || msg.VoiceActivity.AudioOffset != "1.5s" {
		t.Errorf("activity: %+v %+v", msg.VoiceActivityDetectionSignal, msg.VoiceActivity)
	}
	if msg.Error.Code != 400 || msg.Error.Status != "INVALID_ARGUMENT" || len(msg.Error.Details) != 1 {
		t.Errorf("error: %+v", *msg.Error)
	}
}

func TestLiveServerMessageUnknownField(t *testing.T) {
	t.Parallel()
	// Decoding is strict: a field the SDK does not declare is a protocol error.
	var msg LiveServerMessage
	if err := decodeLiveStrict(`{"serverContent":{"notInTheSdk":true}}`, &msg); err == nil {
		t.Fatal("expected an unknown field error")
	}
}

func TestLiveSpeechConfig(t *testing.T) {
	t.Parallel()
	// VoiceConfig is a union: the unset members are not emitted.
	data := []struct {
		name string
		in   LiveSpeechConfig
		want string
	}{
		{
			name: "prebuilt",
			in:   LiveSpeechConfig{VoiceConfig: LiveVoiceConfig{PrebuiltVoiceConfig: LivePrebuiltVoiceConfig{VoiceName: "Aoede"}}},
			want: `{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Aoede"}}}`,
		},
		{
			name: "replicated",
			in:   LiveSpeechConfig{VoiceConfig: LiveVoiceConfig{ReplicatedVoiceConfig: &LiveReplicatedVoiceConfig{MIMEType: "audio/wav", VoiceSampleAudio: []byte{1}, VoiceConsentSignature: &LiveVoiceConsentSignature{Signature: "c2ln"}}}},
			want: `{"voiceConfig":{"replicatedVoiceConfig":{"mimeType":"audio/wav","voiceSampleAudio":"AQ==","voiceConsentSignature":{"signature":"c2ln"}}}}`,
		},
		{
			name: "voice",
			in:   LiveSpeechConfig{VoiceConfig: LiveVoiceConfig{Voice: "Aoede"}},
			want: `{"voiceConfig":{"voice":"Aoede"}}`,
		},
		{
			name: "multi speaker",
			in: LiveSpeechConfig{MultiSpeakerVoiceConfig: &LiveMultiSpeakerVoiceConfig{SpeakerVoiceConfigs: []LiveSpeakerVoiceConfig{
				{Speaker: "Alice", VoiceConfig: LiveVoiceConfig{Voice: "Aoede"}},
				{Speaker: "Bob", VoiceConfig: LiveVoiceConfig{PrebuiltVoiceConfig: LivePrebuiltVoiceConfig{VoiceName: "Puck"}}},
			}}},
			want: `{"multiSpeakerVoiceConfig":{"speakerVoiceConfigs":[{"speaker":"Alice","voiceConfig":{"voice":"Aoede"}},{"speaker":"Bob","voiceConfig":{"prebuiltVoiceConfig":{"voiceName":"Puck"}}}]}}`,
		},
	}
	for _, c := range data {
		t.Run(c.name, func(t *testing.T) {
			t.Parallel()
			b, err := json.Marshal(c.in)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != c.want {
				t.Errorf("got %s, want %s", b, c.want)
			}
		})
	}
}

func TestLiveSetup(t *testing.T) {
	t.Parallel()
	// SafetySettings and TranslationConfig are forwarded in the setup message.
	setup := LiveSetup{
		Model:          "models/gemini-3.8-live",
		SafetySettings: []SafetySetting{{Category: "HARM_CATEGORY_HARASSMENT", Threshold: 2}},
		GenerationConfig: LiveGenerationConfig{
			TranslationConfig: &LiveTranslationConfig{TargetLanguageCode: "fr", EchoTargetLanguage: new(bool)},
		},
	}
	b, err := json.Marshal(setup)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"model":"models/gemini-3.8-live","generationConfig":{"translationConfig":{"targetLanguageCode":"fr","echoTargetLanguage":false}},"safetySettings":[{"category":"HARM_CATEGORY_HARASSMENT","threshold":2}],"realtimeInputConfig":{"activityHandling":"","turnCoverage":""}}`
	if string(b) != want {
		t.Errorf("got %s, want %s", b, want)
	}
}

func TestLiveFunctionResponse(t *testing.T) {
	t.Parallel()
	// A media function response carries parts instead of a JSON response.
	data := `{"functionResponses":[{"name":"snap","response":{},"parts":[{"inlineData":{"mimeType":"image/png","data":"AQACAA=="}},{"fileData":{"fileUri":"https://example.com/a.png","mimeType":"image/png"}}]}]}`
	var msg LiveToolResponse
	if err := decodeLiveStrict(data, &msg); err != nil {
		t.Fatal(err)
	}
	parts := msg.FunctionResponses[0].Parts
	if len(parts) != 2 || parts[0].InlineData.MIMEType != "image/png" || len(parts[0].InlineData.Data) != 4 || parts[0].InlineData.Data[2] != 2 || parts[1].FileData.FileURI != "https://example.com/a.png" {
		t.Errorf("parts: %+v", parts)
	}
}

func TestLiveContextUpdate(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		msg  LiveClientMessage
		want string
	}{
		{
			name: "instructions",
			msg:  LiveClientMessage{ContextUpdate: LiveContextUpdate{SystemInstruction: LiveContent{Parts: []LivePart{{Text: "be terse"}}}}},
			want: `{"contextUpdate":{"systemInstruction":{"parts":[{"text":"be terse"}]}}}`,
		},
		{
			name: "clear tools",
			msg:  LiveClientMessage{ContextUpdate: LiveContextUpdate{Tools: &LiveContextUpdateTools{}}},
			want: `{"contextUpdate":{"tools":{}}}`,
		},
		{
			name: "replace tools",
			msg:  LiveClientMessage{ContextUpdate: LiveContextUpdate{Tools: &LiveContextUpdateTools{Tools: LiveTools{{FunctionDeclarations: []LiveFunctionDeclaration{{Name: "lookup"}}}}}}},
			want: `{"contextUpdate":{"tools":{"tools":[{"functionDeclarations":[{"name":"lookup","description":"","parametersJsonSchema":null,"behavior":""}]}]}}}`,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			b, err := json.Marshal(&tc.msg)
			if err != nil {
				t.Fatal(err)
			}
			if string(b) != tc.want {
				t.Fatalf("%s != %s", b, tc.want)
			}
		})
	}
}

func TestLiveAvatarConfig(t *testing.T) {
	t.Parallel()
	audio, video := int32(32000), int32(64000)
	cfg := LiveAvatarConfig{
		AvatarName:       "avatar_1",
		CustomizedAvatar: &LiveCustomizedAvatar{ImageMIMEType: "image/jpeg", ImageData: []byte{1, 2}},
		AudioBitrateBPS:  &audio,
		VideoBitrateBPS:  &video,
	}
	b, err := json.Marshal(&cfg)
	if err != nil {
		t.Fatal(err)
	}
	want := `{"avatarName":"avatar_1","customizedAvatar":{"imageMimeType":"image/jpeg","imageData":"AQI="},"audioBitrateBps":32000,"videoBitrateBps":64000}`
	if string(b) != want {
		t.Fatalf("%s != %s", b, want)
	}
}

func TestLiveToolsFrom(t *testing.T) {
	t.Parallel()
	var got LiveTools
	tools := []genai.ToolDef{{Name: "lookup", Description: "Lookup", InputSchemaOverride: genai.JSONSchema(`{"type":"object"}`)}}
	if err := got.From(tools, LiveFunctionNonBlocking); err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || len(got[0].FunctionDeclarations) != 1 {
		t.Fatalf("declarations: %#v", got)
	}
	d := got[0].FunctionDeclarations[0]
	if d.Name != "lookup" || d.Description != "Lookup" || d.Behavior != LiveFunctionNonBlocking || string(d.ParametersJSONSchema) != `{"type":"object"}` {
		t.Fatalf("declaration: %#v", d)
	}
}
