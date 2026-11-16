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

    def test_nested_and_opaque_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "sdk.d.ts"
            dto = root / "dto.go"
            sdk.write_text(
                "export declare type SDKMessage = {\n"
                "    type: 'user';\n"
                "    /** Mentions Hidden, which is not referenced. */\n"
                "    period: Period;\n"
                "    raw: {\n        raw_inner: string;\n    };\n"
                "    shared: {\n        shared_inner: string;\n    };\n"
                "};\n"
                "export declare type Period = {\n    resets_at: number;\n};\n"
                "export declare type Hidden = {\n    hidden_field: string;\n};\n"
            )
            # raw is opaque everywhere; shared is typed in one struct, so its
            # contents are checked.
            dto.write_text(
                'const OutputUser OutputType = "user"\n'
                "type A struct {\n"
                '\tPeriod json.RawMessage `json:"period"`\n'
                '\tPeriodTyped Period `json:"period"`\n'
                '\tRaw json.RawMessage `json:"raw"`\n'
                '\tShared base.Unknown `json:"shared"`\n'
                "}\n"
                'type B struct {\n\tShared Shared `json:"shared"`\n}\n'
            )
            result = subprocess.run(
                [sys.executable, str(Path(extract_schema.__file__)), str(sdk), str(dto)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("  resets_at (Period)\n", result.stdout)
            self.assertIn("  shared_inner (SDKMessage)\n", result.stdout)
            self.assertNotIn("raw_inner", result.stdout)
            self.assertNotIn("hidden_field", result.stdout)


if __name__ == "__main__":
    unittest.main()
