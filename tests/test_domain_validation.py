from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.generate_domain_validation import build_cases, write_cases


class DomainValidationTests(unittest.TestCase):
    def test_curated_domain_validation_schema_and_labels(self) -> None:
        cases = build_cases()
        self.assertEqual(len(cases), 10)
        self.assertEqual({case["domain"] for case in cases}, {
            "devops_incident",
            "fintech_risk",
            "healthcare_ops",
            "saas_billing_ops",
            "supply_chain",
        })

        for case in cases:
            traversal = case["traversal"]
            attach = case["attach"]
            self.assertEqual(len(traversal["query"]), 32)
            self.assertEqual(len(traversal["current_summary"]), 32)
            self.assertEqual(len(traversal["path"]), 32)
            self.assertEqual(len(attach["new_summary"]), 32)
            self.assertEqual(len(attach["new_full"]), 64)
            self.assertEqual(len(traversal["candidates"]), 12)
            self.assertEqual(len(attach["candidates"]), 12)
            self.assertGreaterEqual(sum(candidate["result_label"] for candidate in traversal["candidates"]), 2)
            self.assertGreaterEqual(sum(candidate["label"] for candidate in attach["candidates"]), 2)
            kinds = {candidate["kind"] for candidate in traversal["candidates"]}
            self.assertIn("hard_same_domain_negative", kinds)
            self.assertIn("cross_domain_negative", kinds)

    def test_write_cases_outputs_benchmark_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            write_cases(build_cases(), output_dir=output_dir)

            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["traversal_ranking_cases"], 10)
            self.assertEqual(manifest["attach_ranking_cases"], 10)
            self.assertTrue((output_dir / "traversal_ranking.jsonl").exists())
            self.assertTrue((output_dir / "attach_ranking.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
