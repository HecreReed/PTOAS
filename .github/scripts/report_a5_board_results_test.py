#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import importlib.util
import pathlib
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).with_name("report_a5_board_results.py")
SPEC = importlib.util.spec_from_file_location("report_a5_board_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPORT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORT
SPEC.loader.exec_module(REPORT)


class BoardResultReportTest(unittest.TestCase):
    def test_load_and_render_results(self) -> None:
        with tempfile.TemporaryDirectory(prefix="a5-board-report-") as temp_dir:
            results = pathlib.Path(temp_dir) / "results.tsv"
            results.write_text(
                "testcase\tstatus\tstage\tinfo\n"
                "Abs/abs\tOK\trun\tvalidation=independent-golden\n"
                "Add/add\tFAIL\trun\texit=1\n"
                "Print/print\tSKIP\trun\tin SKIP_CASES\n",
                encoding="utf-8",
            )

            summary = REPORT.load_results(results)
            self.assertTrue(summary.results_found)
            self.assertEqual(summary.total, 3)
            self.assertEqual(summary.counts["OK"], 1)
            self.assertEqual(summary.counts["FAIL"], 1)
            self.assertEqual(summary.failed_cases, ("Add/add",))

            markdown = REPORT.render_markdown(
                summary,
                conclusion="failure",
                run_url="https://github.example/actions/runs/1",
                sha="0123456789abcdef",
            )
            self.assertIn("Status: **FAIL**", markdown)
            self.assertIn("`Add/add`", markdown)
            self.assertIn("`0123456789ab`", markdown)

    def test_missing_results_are_reported_as_failure(self) -> None:
        summary = REPORT.load_results(pathlib.Path("/path/that/does/not/exist"))
        self.assertFalse(summary.results_found)
        self.assertFalse(summary.passed)
        payload = REPORT.build_feishu_payload(
            summary,
            conclusion="failure",
            run_url="",
            sha="",
        )
        self.assertEqual(payload["card"]["header"]["template"], "red")

    def test_successful_results_are_reported_as_pass(self) -> None:
        summary = REPORT.BoardSummary(
            True,
            REPORT.Counter({"OK": 2, "SKIP": 1}),
            (),
        )
        self.assertTrue(summary.passed)
        payload = REPORT.build_feishu_payload(
            summary,
            conclusion="success",
            run_url="https://github.example/actions/runs/2",
            sha="abcdef0123456789",
        )
        self.assertEqual(payload["card"]["header"]["template"], "green")


if __name__ == "__main__":
    unittest.main()
