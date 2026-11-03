# OpenCode Package

Implements `genai.Provider` backed by the OpenCode CLI (`opencode acp`). The protocol and session behavior
are owned by [`acp`](../../acp/acp.go); [`client.go`](client.go) owns the OpenCode process, model, effort,
and mode settings, and [`dto.go`](dto.go) holds only the OpenCode-specific option values.

## Install

Install OpenCode from https://opencode.ai and authenticate once:

```sh
opencode auth login
opencode acp
```

## Recording

```sh
RECORD=failure_only go test ./providers/opencode/
```

`TestClient/Scoreboard` records one fixture per probe under
`testdata/TestClient/Scoreboard/<provider>/<model>/`. Replay needs no `opencode` binary.

## Keeping DTOs in sync

The wire types live in [`acp/dto.go`](../../acp/dto.go), pinned there to OpenCode `1.19.4`, commit
`148842ef7cb1bc34192b3fff57d0f9721ab9c801`, and `@agentclientprotocol/sdk` `0.21.0`, commit
`74372b59f8f56eb30e72f2538df2671a0adb376f`. The shared DTO also serves providers that group
`session/new` configuration values; keep grouped and flat `ConfigOptionValue` options working.

When updating the shared DTO:

1. Clone https://github.com/anomalyco/opencode at the pinned commit.
2. Diff `packages/opencode/src/acp/agent.ts`, `service.ts`, `event.ts`, `permission.ts`, `content.ts`, and
   `tool.ts` against [`acp/dto.go`](../../acp/dto.go) and [`acp/acp.go`](../../acp/acp.go).
3. Compare the exact ACP SDK version in `packages/opencode/package.json`. Do not update from ACP
   specification HEAD alone; it can be ahead of the SDK shipped by OpenCode.
4. Record fresh fixtures.

## References

- https://opencode.ai/docs/acp/: ACP documentation
- https://opencode.ai/docs/config/: configuration format
- https://opencode.ai/docs/providers/: provider configurations
- https://agentclientprotocol.com: ACP specification
