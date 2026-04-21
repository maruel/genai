#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Extract Zod wire-protocol schemas from the claude binary.

The claude binary is a Bun SEA that compiles JS to native x86-64 via JSC.
No bytecode survives but Zod schema property names are stored as string
literals. This script extracts E.object({...}) bodies via brace-counting
and parses the top-level field names without evaluating the code.

Run after upgrading claude to spot new fields vs dto.go.

Usage: ./extract_schema.py [path-to-claude-binary]
"""

import argparse
import re
import shutil
import sys
from dataclasses import dataclass


@dataclass
class Schema:
    label: str
    needle: str


def _needle(type_str: str, subtype: str | None = None) -> str:
    if subtype is None:
        return f'type:E.literal("{type_str}")'
    return f'type:E.literal("{type_str}"),subtype:E.literal("{subtype}")'


# Tracked message types. result/error uses E.enum() for its subtype discriminator.
SCHEMAS: list[Schema] = [
    Schema("system/init", _needle("system", "init")),
    Schema("stream_event", _needle("stream_event")),
    Schema("result/success", _needle("result", "success")),
    Schema(
        "result/error",
        'type:E.literal("result"),subtype:E.enum(["error_during_execution',
    ),
    Schema("assistant", _needle("assistant")),
    Schema("rate_limit_event", _needle("rate_limit_event")),
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


def find_schema_body(data: str, needle: str, window: int = 4000) -> str | None:
    """Return the E.object body starting at needle up to the closing brace."""
    idx = data.find(needle)
    if idx < 0:
        return None
    chunk = data[idx : idx + window]
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
            m = re.match(r"([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:", body[i:])
            if m:
                keys.append(m.group(1))
                i += len(m.group(0))
                continue
        i += 1
    return keys


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

    for schema in SCHEMAS:
        body = find_schema_body(data, schema.needle)
        if body is None:
            print(f"\n{schema.label}: NOT FOUND")
            continue
        keys = extract_top_level_keys(body)
        print(f"\n{schema.label}:")
        for k in keys:
            print(f"  {k}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
