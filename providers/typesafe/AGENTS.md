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

- `GenSync` takes exactly one message, whose state is its request: text or a JSON document. A message with
  several requests becomes an array state, which the API takes as a sequence of messages or records. The API
  evaluates a string as text, so structured data must use the document, which must be `application/json`:
  `genai.Doc` derives the media type from the filename, so it must be named `*.json`. Replies and tool call
  results are rejected rather than silently dropped.
- `genai.GenOptionText` with `DecodeAs` is the only option accepted. Any other option returns
  `base.ErrNotSupported`, and duplicate options follow genai's convention: the last one wins.
- `Content` is a closed interface, so a value the API does not accept is a compile error instead of a
  runtime 400. Widening it to every JSON value, and typing `Object` and `Array` as
  `map[string]Content`/`[]Content` instead of `any`, were considered and rejected: they would only turn a
  compile error into a runtime one, and break passing an existing `map[string]any` or a struct as the
  state.
- Only `Noul`, `Choice` and `Score` fields are supported. Plain Go types would have to be derived from a
  JSON schema, which loses the probabilities the API reports.

## Scoreboard

`providers/typesafe/scoreboard.json` is authored by hand. `smoke/smoketest` drives chat providers
through `genai.Provider.GenSync` with text prompts, which cannot exercise a provider that only answers
questions, so `-update-scoreboard` is not wired up here. The declared functionality (`jsonSchema`,
`reportTokenUsage`) is verified by the recorded tests in `client_test.go` instead. Re-check it by hand
when the API changes.
