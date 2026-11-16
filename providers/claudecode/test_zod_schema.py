# Copyright 2026 Marc-Antoine Ruel. All rights reserved.
# Use of this source code is governed under the Apache License, Version 2.0
# that can be found in the LICENSE file.

"""Regression tests for the claude binary Zod schema lookup and check."""

import unittest

import zod_schema

DTO = """
const (
	OutputToolProgress OutputType = "tool_progress"
)
type OutputToolProgressMsg struct {
	Type   OutputType      `json:"type"`
	TaskID string          `json:"task_id"`
	Extra  json.RawMessage `json:"extra,omitempty"`
}
type Period struct {
	Utilization float64 `json:"utilization"`
}
"""


def _binary(far_definition: bytes = b"") -> bytes:
    schema = (
        b'var Ab=p(()=>u({type:T("tool_progress"),task_id:o().describe("Owning task."),'
        b"period:Pr(),extra:u({ignored:o()}),o_field:o(),added:E().optional()}))"
        b",Pr=p(()=>u({utilization:E(),resets_at:E().describe(`When.`)}));"
    )
    # A definition beyond MODULE_SPAN belongs to another module.
    return schema + b" " * (zod_schema.MODULE_SPAN + 1) + far_definition


class ZodSchemaTest(unittest.TestCase):
    def test_lookup(self):
        self.assertEqual(
            zod_schema.lookup(_binary(), b"resets_at"),
            "p(()=>u({utilization:E(),resets_at:E()}))",
        )
        with self.assertRaises(LookupError):
            zod_schema.lookup(_binary(), b"absent_key")

    def test_check(self):
        report = zod_schema.check(_binary(b",o=p(()=>u({foreign:o()}))"), DTO)
        # extra is raw JSON in dto.go, so ignored is not reported; o() is a
        # primitive here because its only definition is in another module.
        self.assertEqual(report, {"tool_progress": ["added", "o_field", "period", "resets_at"]})


if __name__ == "__main__":
    unittest.main()
