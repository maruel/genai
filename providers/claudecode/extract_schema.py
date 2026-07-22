#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Extract Zod wire-protocol schemas from the claude binary.

The claude binary is a Bun SEA that compiles JS to native x86-64 via JSC.
No bytecode survives but Zod schema property names are stored as string
literals. This script extracts Zod object bodies via brace-counting and parses
the top-level field names without evaluating the code.

Run after upgrading claude to spot new fields vs dto.go.

Usage: ./extract_schema.py [path-to-claude-binary]
"""

import argparse
import re
import shutil
import sys
from dataclasses import dataclass


_ZOD_NAME = r"(?P<zod>[a-zA-Z_$][a-zA-Z0-9_$]*)"


@dataclass
class Schema:
    label: str
    type_name: str
    subtype: str | None = None
    subtype_enum_first: str | None = None


def _schema_pattern(schema: Schema) -> re.Pattern[str]:
    discriminator = rf'type:(?P=zod)\.literal\("{re.escape(schema.type_name)}"\)'
    if schema.subtype is not None:
        discriminator += rf',subtype:(?P=zod)\.literal\("{re.escape(schema.subtype)}"\)'
    elif schema.subtype_enum_first is not None:
        discriminator += (
            rf',subtype:(?P=zod)\.enum\(\["{re.escape(schema.subtype_enum_first)}"'
        )
    return re.compile(rf"{_ZOD_NAME}\.object\(\{{(?P<body>{discriminator})")


# Tracked non-system messages. System subtypes are discovered automatically.
SCHEMAS: list[Schema] = [
    Schema("stream_event", "stream_event"),
    Schema("result/success", "result", subtype="success"),
    Schema("result/error", "result", subtype_enum_first="error_during_execution"),
    Schema("assistant", "assistant"),
    Schema("rate_limit_event", "rate_limit_event"),
]


def _skip_string(data: str, i: int) -> int:
    """Advance i past a JS string literal starting at data[i]. Returns new i."""
    quote = data[i]
    i += 1
    while i < len(data):
        c = data[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    return i


def find_schema_body(data: str, schema: Schema, window: int = 16_000) -> str | None:
    """Return a Zod object body for schema up to its closing brace."""
    match = _schema_pattern(schema).search(data)
    if match is None:
        return None
    chunk = data[match.start("body") : match.start("body") + window]
    depth = 0
    i = 0
    while i < len(chunk):
        c = chunk[i]
        if c in ('"', "'", "`"):
            i = _skip_string(chunk, i)
            continue
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
            if depth < 0:
                return chunk[:i]
        i += 1
    return chunk


def extract_top_level_keys(body: str) -> list[str]:
    """Extract property names at brace-depth 0 from a JS object literal body."""
    keys: list[str] = []
    depth = 0
    i = 0
    while i < len(body):
        c = body[i]
        if c in ('"', "'", "`"):
            i = _skip_string(body, i)
            continue
        if c in "{([":
            depth += 1
            i += 1
            continue
        if c in "})]":
            depth -= 1
            i += 1
            continue
        if depth == 0:
            match = re.match(r"([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:", body[i:])
            if match:
                keys.append(match.group(1))
                i += len(match.group(0))
                continue
        i += 1
    return keys


def find_system_schemas(data: str) -> list[Schema]:
    """Discover every literal system subtype schema in binary order."""
    pattern = re.compile(
        rf'{_ZOD_NAME}\.object\(\{{type:(?P=zod)\.literal\("system"\),'
        rf'subtype:(?P=zod)\.literal\("(?P<subtype>[^"]+)"\)'
    )
    schemas: list[Schema] = []
    seen: set[str] = set()
    for match in pattern.finditer(data):
        subtype = match.group("subtype")
        if subtype not in seen:
            schemas.append(Schema(f"system/{subtype}", "system", subtype=subtype))
            seen.add(subtype)
    return schemas


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract Zod wire-protocol field names from the claude binary."
    )
    parser.add_argument(
        "binary",
        nargs="?",
        help="Path to claude binary (default: which claude)",
    )
    args = parser.parse_args()

    binary = args.binary or shutil.which("claude")
    if not binary:
        print("error: claude binary not found in PATH", file=sys.stderr)
        return 1

    print(f"Extracting wire-protocol schemas from {binary}")

    with open(binary, "rb") as f:
        data = f.read().decode("latin-1")

    schemas = [*find_system_schemas(data), *SCHEMAS]
    found = 0
    for schema in schemas:
        body = find_schema_body(data, schema)
        if body is None:
            print(f"\n{schema.label}: NOT FOUND")
            continue
        found += 1
        keys = extract_top_level_keys(body)
        print(f"\n{schema.label}:")
        for key in keys:
            print(f"  {key}")

    if found == 0:
        print("error: no wire-protocol schemas found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
