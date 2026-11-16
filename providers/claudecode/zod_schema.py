#!/usr/bin/env python3
# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Look up and check the Zod output schemas embedded in the claude binary.

The claude binary is a Bun executable that embeds the minified Zod schemas of
its stream-json output, including the @internal messages and fields that
sdk.d.ts omits. extract_schema.py checks the SDK; this script covers the rest.

lookup prints the schema object enclosing a pattern, without descriptions, to
type a field seen in a log:

  ./zod_schema.py ~/.local/bin/claude lookup 'type:T("command_lifecycle")'

check lists, per output message that dto.go declares, the schema fields that
no dto.go JSON tag names. Most are @internal telemetry that stream-json omits;
add a field when a log shows it. Values that dto.go keeps as raw JSON are not
descended into.

  ./zod_schema.py ~/.local/bin/claude check dto.go
"""

import argparse
import mmap
import re
import sys
from pathlib import Path

from extract_schema import GO_FIELD, GO_JSON_TAG, GO_OPAQUE_TYPE

DESCRIBE = re.compile(
    rb"\.describe\((?:\"(?:[^\"\\]|\\.)*\"|`(?:[^`\\]|\\.)*`|'(?:[^'\\]|\\.)*'|[\w$]+)"
    rb"(?:\+(?:\"(?:[^\"\\]|\\.)*\"|[\w$]+))*\)"
)
MESSAGE = re.compile(rb'u\(\{type:T\("(\w+)"\)(?:,subtype:T\("(\w+)"\))?')
DEFINITION = re.compile(rb"[,;\s]([\w$]{1,4})=p\(\(\)=>")
KEY = re.compile(r"[{,]([A-Za-z_]\w*):")
REFERENCE = re.compile(r"(?<![\w$.])([\w$]{1,4})\(\)")
# Largest distance between a schema and a definition it references, in bytes.
MODULE_SPAN = 1_000_000
GO_OUTPUT_TYPE = re.compile(r'^\s*Output\w+\s+OutputType\s*=\s*"(\w+)"', re.MULTILINE)
GO_SYSTEM_SUBTYPE = re.compile(r'^\s*System\w+\s+SystemSubtype\s*=\s*"(\w+)"', re.MULTILINE)


def _call_end(data: bytes, start: int) -> int:
    """Return the index after the ')' closing the '(' at start, skipping string literals."""
    depth = 0
    quote = 0
    escaped = False
    for index in range(start, len(data)):
        char = data[index]
        if quote:
            if escaped:
                escaped = False
            elif char == 0x5C:
                escaped = True
            elif char == quote:
                quote = 0
        elif char in b"\"'`":
            quote = char
        elif char == 0x28:
            depth += 1
        elif char == 0x29:
            depth -= 1
            if depth == 0:
                return index + 1
    raise ValueError(f"unbalanced expression at offset {start}")


def _schema(data: bytes, call: int) -> str:
    """Return the call starting at call ('u(' or 'p('), without descriptions."""
    open_paren = data.find(b"(", call)
    return DESCRIBE.sub(b"", data[call : _call_end(data, open_paren)]).decode("utf-8", "replace")


def _value_end(text: str, start: int) -> int:
    """Return the end of the object value starting at start: the next ',' or '}' outside brackets."""
    depth = 0
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in "\"'`":
            quote = char
        elif char in "({[":
            depth += 1
        elif char in ")}]":
            if depth == 0:
                return index
            depth -= 1
        elif char == "," and depth == 0:
            return index
    return len(text)


def _drop_opaque(text: str, opaque: set[str]) -> str:
    out = []
    position = 0
    for match in KEY.finditer(text):
        if match.start() < position or match.group(1) not in opaque:
            continue
        out.append(text[position : match.end()])
        position = _value_end(text, match.end())
    out.append(text[position:])
    return "".join(out)


def lookup(data: bytes, pattern: bytes) -> str:
    """Return the innermost lazy schema definition containing pattern."""
    offset = data.find(pattern)
    if offset < 0:
        raise LookupError(f"{pattern.decode()} not found")
    start = data.rfind(b"=p(()=>", max(0, offset - 1_000_000), offset)
    if start < 0:
        raise LookupError(f"no schema definition encloses {pattern.decode()}")
    return _schema(data, start + 1)


def check(data: bytes, dto_source: str) -> dict[str, list[str]]:
    """Map each output message that dto.go declares to its schema fields missing from dto.go."""
    types = set(GO_OUTPUT_TYPE.findall(dto_source))
    subtypes = set(GO_SYSTEM_SUBTYPE.findall(dto_source))
    tags = set(GO_JSON_TAG.findall(dto_source))
    go_fields = GO_FIELD.findall(dto_source)
    typed = {tag for go_type, tag in go_fields if not GO_OPAQUE_TYPE.fullmatch(go_type)}
    opaque = {tag for go_type, tag in go_fields if GO_OPAQUE_TYPE.fullmatch(go_type)} - typed
    # Minified names repeat across bundled modules; resolve a reference to the
    # definition nearest the message that uses it, within the same module. A
    # name defined only far away is a Zod primitive here, such as o() for
    # z.string().
    definitions: dict[str, list[int]] = {}
    for match in DEFINITION.finditer(data):
        definitions.setdefault(match.group(1).decode(), []).append(match.start(1))

    report: dict[str, list[str]] = {}
    for match in MESSAGE.finditer(data):
        kind = match.group(1).decode()
        subtype = (match.group(2) or b"").decode()
        if kind not in types or (kind == "system" and subtype not in subtypes):
            continue
        keys: set[str] = set()
        seen: set[str] = set()
        pending = [_schema(data, match.start())]
        while pending:
            text = _drop_opaque(pending.pop(), opaque)
            keys.update(KEY.findall(text))
            for name in REFERENCE.findall(text):
                if name in seen or name not in definitions:
                    continue
                seen.add(name)
                nearest = min(definitions[name], key=lambda position: abs(position - match.start()))
                if abs(nearest - match.start()) > MODULE_SPAN:
                    continue
                pending.append(_schema(data, data.find(b"p(()=>", nearest)))
        missing = sorted(keys - tags)
        if missing:
            label = f"{kind}/{subtype}" if subtype else kind
            report[label] = sorted(set(report.get(label, [])) | set(missing))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("binary", type=Path, help="claude executable")
    commands = parser.add_subparsers(dest="command", required=True)
    lookup_parser = commands.add_parser("lookup", help="print the schema enclosing a pattern")
    lookup_parser.add_argument("pattern", help='minified source, e.g. type:T("command_lifecycle")')
    check_parser = commands.add_parser("check", help="list schema fields missing from dto.go")
    check_parser.add_argument("dto", type=Path, nargs="?", default=Path(__file__).with_name("dto.go"))
    args = parser.parse_args()

    with args.binary.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if args.command == "lookup":
            try:
                print(lookup(data, args.pattern.encode()))
            except LookupError as error:
                print(f"error: {error}", file=sys.stderr)
                return 1
            return 0
        report = check(data, args.dto.read_text(encoding="utf-8"))
    for label, fields in sorted(report.items()):
        print(f"{label}: {' '.join(fields)}")
    return 1 if report else 0


if __name__ == "__main__":
    sys.exit(main())
