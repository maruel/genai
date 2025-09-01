#!/usr/bin/env python3
# Copyright 2025 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Print the response body of an interaction file.

Recommended usage:
  ./internal/print_response_body.py \
    providers/cerebras/testdata/TestClient/Scoreboard/gpt-oss-120b_thinking/GenStream-Tools-SquareRoot-1-any.yaml
    | less

"""

import json
import sys
import yaml


def main():
    if len(sys.argv) < 2:
        print("Usage: print_response_body.py <filename>")
        return 1
    with open(sys.argv[1], "r") as f:
        data = yaml.safe_load(f)
    for interaction in data["interactions"]:
        body = interaction["response"]["body"]
        if body.startswith("data:"):
            for line in body.split("\n"):
                if line.startswith("data:"):
                    line = line[5:]
                line = line.strip()
                if line:
                    try:
                        d = json.loads(line)
                        print(json.dumps(d, indent=2))
                    except ValueError:
                        print(line)
        else:
            print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
