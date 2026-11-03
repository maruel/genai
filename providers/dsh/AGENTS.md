# DeepSeek Harness CLI Provider

Implements `genai.Provider` backed by the DeepSeek Harness CLI (`dsh --profile acp`). The package
documentation and [`client.go`](client.go) own the protocol, model routes, sessions, and usage behavior;
[`acp`](../../acp/acp.go) owns the shared ACP engine and [`dto.go`](../../acp/dto.go)
owns the wire types.

## Install

```sh
npm install --global @deepseek-ai/dsh
export DEEPSEEK_API_KEY=...
dsh --profile acp
```

The first boot creates `$DSH_HOME/profiles/acp` from the shipped profile. The default profile routes
`deepseek-official` with model `deepseek-v4-flash`. Fixture recording requires `dsh` on `PATH`,
`DEEPSEEK_API_KEY`, and this profile; replay needs none of them. `$DSH_HOME` also stores the sessions that
`session/resume` restores, so keep it stable while recording multi-turn cases.

## Recording

```sh
RECORD=failure_only go test ./providers/dsh/
```

`TestClient/Scoreboard` records one fixture per probe under
`testdata/TestClient/Scoreboard/<provider>/<model>/`. The scoreboard tests `deepseek-v4-pro` and
`deepseek-flash`; `deepseek-v4-flash` is the default route, is covered by the client tests, and stays
untested. Model discovery requires network access to the configured provider catalog.

## Keeping DTOs in sync

The wire contract is pinned to DeepSeek Harness `dsh-v0.2.1-rc.5`, commit
`b830ec950faba0a46dda564be3e2d854afed94f1`, and `@agentclientprotocol/sdk` `1.4.0`, commit
`e6463f444093ed7c5f1cc937c3f32afb5853e906`. To check a new release:

1. Clone https://github.com/deepseek-ai/deepseek-harness at the release tag.
2. Diff these sources against [`dto.go`](../../acp/dto.go):
   `packages/acp/acp/src/index.ts` (method dispatch and permissions),
   `packages/acp/acp/src/model-control.ts` (`model` and `reasoning_effort` options),
   `packages/acp/acp/src/updates.ts` (session updates), and
   `packages/acp/acp/src/codec.ts` (stop reasons).
3. Diff the ACP SDK schema at the pinned SDK commit.
4. Record fresh fixtures and update the header comment.

## References

- https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/acp/acp/README.md: ACP profile contract
- https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/cli-help.md: `dsh` command grammar
- https://agentclientprotocol.com: ACP specification
