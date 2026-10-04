# TypeSafe AI

- **Official Documentation**: https://docs.typesafe.ai/
- **API Reference**: https://docs.typesafe.ai/api
- **Python SDK**: https://github.com/typesafe-ai/typesafe-sdk-python
- **JavaScript SDK**: https://github.com/typesafe-ai/typesafe-sdk-js

## Implementation Notes

TypeSafe is not a chat provider. It answers typed questions about a state, so it does not fit the rest
of the provider interfaces directly:

- `POST /v1/systemone` takes `state` plus a map of named, typed questions, and returns one typed answer
  per question. `state` is a string, a JSON object, or a JSON array. Only text states are supported,
  images, audio and video are rejected by the API.
- `GET /v1/models` lists the aliases available to the account. Versioned model IDs are also accepted by
  the `model` field even though they are not listed.
- There is no batching, no caching, no tool calling, no seed and no token or stop limits.

## Mapping to genai

- `SystemOne` accepts a shared `genai.SystemOneRequest` and returns typed answers. `GenSync` and
  `GenStream` return `base.ErrNotSupported`; TypeSafe does not generate text.
- `genai.Questions` and `genai.Answers` use dynamic names and typed values. See `example_test.go`.
- `genai.DecisionContent` is a closed interface, so a value the API does not accept is a compile error instead of a
  runtime 400. Widening it to every JSON value, and typing `genai.Object` and `genai.Array` as
  `map[string]genai.DecisionContent`/`[]genai.DecisionContent` instead of `any`, were considered and rejected: they would only turn a
  compile error into a runtime one, and break passing an existing `map[string]any` or a struct as the
  state.

## Scoreboard

`providers/typesafe/scoreboard.json` is authored by hand. `smoke/smoketest` drives chat providers
through `genai.Provider.GenSync` with text prompts, which cannot exercise a provider that only answers
questions, so `-update-scoreboard` is not wired up here. The legacy scoreboard metadata (`jsonSchema`,
`reportTokenUsage`) describes the decision protocol. The recorded tests in `client_test.go` verify it;
it does not declare working chat generation. Re-check it by hand
when the API changes.
