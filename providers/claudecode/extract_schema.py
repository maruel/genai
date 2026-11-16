#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Check Claude Code Go DTOs against the published TypeScript SDK.

The npm package's sdk.d.ts is the released wire contract. This verifies that
dto.go declares every message, control request and hook discriminator, and
every field reachable from them at any depth. Values that dto.go keeps as raw
JSON wherever they occur are not descended into. zod_schema.py covers the
@internal fields that sdk.d.ts omits.

Usage:
  npm pack @anthropic-ai/claude-agent-sdk@<version>
  tar -xzf anthropic-ai-claude-agent-sdk-<version>.tgz
  ./extract_schema.py package/sdk.d.ts dto.go --version <version>
"""

import argparse
import json
import re
import sys
from pathlib import Path

TYPE_DECLARATION = re.compile(r"(?:export\s+)?declare\s+type\s+(?P<name>[A-Za-z0-9_]+)\s*=")
TYPE_TOKEN = re.compile(
    r"'[^'\\]*(?:\\.[^'\\]*)*'|\"[^\"\\]*(?:\\.[^\"\\]*)*\"|`[^`]*`|/\*.*?\*/|//[^\n]*|[{}()[\];]",
    re.DOTALL,
)
LITERAL_FIELD = re.compile(r"\b(?:type|subtype):\s*'([^']+)'")
QUOTED_LITERAL = re.compile(r"'([^']+)'")
GO_LITERAL = re.compile(r'^\s*(?:const\s+)?\w+(?:\s+\w+)?\s*=\s*"([^"\\]*(?:\\.[^"\\]*)*)"', re.MULTILINE)
TS_FIELD = re.compile(r"^\s+([A-Za-z_][A-Za-z0-9_]*)\??:", re.MULTILINE)
TS_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
TS_IDENTIFIER = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")
GO_JSON_TAG = re.compile(r"json:\"([^,\"]+)")
GO_FIELD = re.compile(r"^\s*\w+\s+(\S+)\s+`json:\"([^,\"]+)", re.MULTILINE)
GO_OPAQUE_TYPE = re.compile(r"(?:\[\])?(?:map\[string\])?(?:json\.RawMessage|base\.Unknown)")


def _declarations(source: str) -> dict[str, str]:
    declarations = {}
    for match in TYPE_DECLARATION.finditer(source):
        depth = 0
        for token in TYPE_TOKEN.finditer(source, match.end()):
            value = token.group()
            if value in {"{", "(", "["}:
                depth += 1
            elif value in {"}", ")", "]"}:
                depth -= 1
            elif value == ";" and depth == 0:
                declarations[match.group("name")] = source[match.end() : token.start()].strip()
                break
        else:
            raise ValueError(f"unterminated type declaration: {match.group('name')}")
    return declarations


def _union_members(body: str) -> list[str]:
    if body.startswith("{"):
        return []
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


def _skip_value(body: str, start: int) -> int:
    """Return the end of the field value starting at start: the next ';' or newline outside brackets."""
    depth = 0
    for index in range(start, len(body)):
        char = body[index]
        if char in "{([<":
            depth += 1
        elif char in "})]>":
            depth -= 1
        elif depth <= 0 and char in ";\n" and index > start:
            return index
    return len(body)


def _drop_opaque(body: str, opaque: set[str]) -> str:
    """Remove the values of fields that dto.go keeps as raw JSON."""
    out = []
    position = 0
    for match in TS_FIELD.finditer(body):
        if match.start() < position or match.group(1) not in opaque:
            continue
        out.append(body[position : match.end()])
        position = _skip_value(body, match.end())
    out.append(body[position:])
    return "".join(out)


def _fields(
    name: str,
    declarations: dict[str, str],
    opaque: set[str],
    fields: dict[str, set[str]],
    seen: set[str],
) -> None:
    """Map each field of name and of the types it references, at every depth, to its declaring types."""
    if name in seen:
        return
    seen.add(name)
    body = _drop_opaque(TS_COMMENT.sub("", declarations.get(name, "")), opaque)
    for field in TS_FIELD.findall(body):
        fields.setdefault(field, set()).add(name)
    for reference in TS_IDENTIFIER.findall(body):
        if reference in declarations:
            _fields(reference, declarations, opaque, fields, seen)


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
    # dto.go keeps these values as raw JSON wherever they occur, so their
    # contents are not checked.
    go_fields = GO_FIELD.findall(dto_source)
    typed = {tag for go_type, tag in go_fields if not GO_OPAQUE_TYPE.fullmatch(go_type)}
    opaque = {tag for go_type, tag in go_fields if GO_OPAQUE_TYPE.fullmatch(go_type)} - typed
    fields: dict[str, set[str]] = {}
    seen: set[str] = set()
    _fields("SDKMessage", declarations, opaque, fields, seen)
    _fields("SDKControlRequestInner", declarations, opaque, fields, seen)

    hooks_match = re.search(r"HOOK_EVENTS:\s*readonly\s*\[(.*?)\]", sdk_source, re.DOTALL)
    if hooks_match:
        expected.update(QUOTED_LITERAL.findall(hooks_match.group(1)))

    present = {bytes(value, "utf-8").decode("unicode_escape") for value in GO_LITERAL.findall(dto_source)}
    missing = sorted(expected - present)
    missing_fields = sorted(set(fields) - set(GO_JSON_TAG.findall(dto_source)))
    print(f"Claude Agent SDK {version or 'unknown'}: {len(expected)} discriminators")
    if missing or missing_fields:
        if missing:
            print("Missing discriminators from dto.go:")
            for value in missing:
                print(f"  {value}")
        if missing_fields:
            print("Missing JSON fields from dto.go:")
            for value in missing_fields:
                print(f"  {value} ({', '.join(sorted(fields[value]))})")
        return 1
    print("All SDK message/control discriminators and fields are declared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
