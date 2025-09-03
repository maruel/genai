#!/usr/bin/env python3
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Print the request and response bodies of an interaction file.

Recommended usage:
  ./internal/print_bodies.py \
    providers/cerebras/testdata/TestClient/Scoreboard/gpt-oss-120b_thinking/GenStream-Tools-SquareRoot-1-any.yaml
    | less -R

"""

import json
import sys
import yaml


def print_line(line):
    try:
        d = json.loads(line)
        print(json.dumps(d, indent=2))
    except ValueError:
        print(line)


def main():
    if len(sys.argv) < 2:
        print("Usage: print_bodies.py <filename>")
        return 1
    with open(sys.argv[1], "r") as f:
        data = yaml.safe_load(f)
    for interaction in data["interactions"]:
        req = interaction["request"].get("body")
        if req:
            print("\033[1mRequest:\033[0m")
            print_line(req)
        print("\033[1mResponse:\033[0m")
        body = interaction["response"]["body"]
        if body.startswith(("data:", "event:")):
            for line in body.split("\n"):
                if line.startswith("event:"):
                    continue
                if line.startswith("data:"):
                    line = line[5:]
                line = line.strip()
                if line:
                    print_line(line)
        else:
            print_line(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
