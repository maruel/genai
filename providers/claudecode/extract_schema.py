#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Check Claude Code Go discriminators against the published TypeScript SDK.

Claude Code binaries no longer retain the Zod source fragments that the old
binary-string extractor depended on. The npm package's sdk.d.ts is the released
wire contract and can be checked reproducibly instead.

Usage:
  npm pack @anthropic-ai/claude-agent-sdk@0.3.270
  tar -xzf anthropic-ai-claude-agent-sdk-0.3.270.tgz
  ./extract_schema.py package/sdk.d.ts dto.go --version 0.3.270
"""

import argparse
import json
import re
import sys
from pathlib import Path

TYPE_DECLARATION = re.compile(
    r"(?:export\s+)?declare\s+type\s+(?P<name>[A-Za-z0-9_]+)\s*=\s*(?P<body>.*?);",
    re.DOTALL,
)
LITERAL_FIELD = re.compile(r"\b(?:type|subtype):\s*'([^']+)'")
QUOTED_LITERAL = re.compile(r"'([^']+)'")
GO_LITERAL = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
TS_FIELD = re.compile(r"^    ([A-Za-z_][A-Za-z0-9_]*)(?:\?)?:", re.MULTILINE)
GO_JSON_TAG = re.compile(r"json:\"([^,\"]+)")


def _declarations(source: str) -> dict[str, str]:
    return {match.group("name"): match.group("body") for match in TYPE_DECLARATION.finditer(source)}


def _union_members(body: str) -> list[str]:
    return re.findall(r"\bSDK[A-Za-z0-9_]+\b", body)


def _discriminators(name: str, declarations: dict[str, str], seen: set[str]) -> set[str]:
    if name in seen:
        return set()
    seen.add(name)
    body = declarations.get(name, "")
    values = set(LITERAL_FIELD.findall(body))
    for member in _union_members(body):
        values.update(_discriminators(member, declarations, seen))
    return values


def _fields(name: str, declarations: dict[str, str], seen: set[str]) -> set[str]:
    if name in seen:
        return set()
    seen.add(name)
    body = declarations.get(name, "")
    values = set(TS_FIELD.findall(body))
    for member in _union_members(body):
        values.update(_fields(member, declarations, seen))
    return values


def _sdk_version(path: Path) -> str | None:
    package_json = path.parent / "package.json"
    if not package_json.is_file():
        return None
    with package_json.open(encoding="utf-8") as stream:
        value = json.load(stream).get("version")
    return value if isinstance(value, str) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdk", type=Path, help="sdk.d.ts from @anthropic-ai/claude-agent-sdk")
    parser.add_argument("dto", type=Path, nargs="?", default=Path(__file__).with_name("dto.go"))
    parser.add_argument("--version", help="require this SDK package version")
    args = parser.parse_args()

    sdk_source = args.sdk.read_text(encoding="utf-8")
    dto_source = args.dto.read_text(encoding="utf-8")
    version = _sdk_version(args.sdk)
    if args.version and version != args.version:
        print(
            f"error: SDK version is {version or 'unknown'}, expected {args.version}",
            file=sys.stderr,
        )
        return 2

    declarations = _declarations(sdk_source)
    expected = _discriminators("SDKMessage", declarations, set())
    expected.update(_discriminators("SDKControlRequestInner", declarations, set()))
    expected_fields = _fields("SDKMessage", declarations, set())
    expected_fields.update(_fields("SDKControlRequestInner", declarations, set()))

    hooks_match = re.search(r"HOOK_EVENTS:\s*readonly\s*\[(.*?)\]", sdk_source, re.DOTALL)
    if hooks_match:
        expected.update(QUOTED_LITERAL.findall(hooks_match.group(1)))

    present = {bytes(value, "utf-8").decode("unicode_escape") for value in GO_LITERAL.findall(dto_source)}
    missing = sorted(expected - present)
    missing_fields = sorted(expected_fields - set(GO_JSON_TAG.findall(dto_source)))
    print(f"Claude Agent SDK {version or 'unknown'}: {len(expected)} discriminators")
    if missing or missing_fields:
        if missing:
            print("Missing discriminators from dto.go:")
            for value in missing:
                print(f"  {value}")
        if missing_fields:
            print("Missing top-level JSON fields from dto.go:")
            for value in missing_fields:
                print(f"  {value}")
        return 1
    print("All SDK message/control discriminators and top-level fields are declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
