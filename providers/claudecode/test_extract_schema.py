# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Regression tests for the released Claude SDK schema checker."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import extract_schema


class SchemaCheckerTest(unittest.TestCase):
    def test_complete_declarations(self):
        source = """export declare type SDKUser = {
    type: 'user';
    // A semicolon ; in a comment is not a type boundary.
    message: { content: 'text;value'; nested: string[] };
    agent_id?: string;
};
declare type SDKMessage = SDKUser | { type: 'assistant'; uuid: string };
"""
        declarations = extract_schema._declarations(source)
        self.assertEqual(extract_schema._fields("SDKUser", declarations, set()), {"type", "message", "agent_id"})
        self.assertIn("nested: string[]", declarations["SDKUser"])
        self.assertIn("uuid: string", declarations["SDKMessage"])

    def test_missing_field_after_discriminator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "sdk.d.ts"
            dto = root / "dto.go"
            sdk.write_text(
                "export declare type SDKMessage = {\n    type: 'user';\n    tool_result_meta?: string;\n};\n"
            )
            # A Go rune containing a quote must not hide a subsequent constant.
            dto.write_text('if data[0] == \'"\' {}\nconst OutputUser OutputType = "user"\n')
            result = subprocess.run(
                [sys.executable, str(Path(extract_schema.__file__)), str(sdk), str(dto)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("tool_result_meta", result.stdout)
            self.assertNotIn("Missing discriminators", result.stdout)


if __name__ == "__main__":
    unittest.main()
