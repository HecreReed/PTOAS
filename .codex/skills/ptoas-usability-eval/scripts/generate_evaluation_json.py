#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path


TOP_LEVEL_FIELDS = [
    "evaluation_id",
    "operator_name",
    "batch_id",
    "development_success",
    "test_results",
    "summary",
    "documentation_retrieval",
    "code_examples",
    "build_configuration",
    "functional_testing",
    "key_findings",
    "dimension_tables",
    "evaluated_at",
    "report_file",
]

SUMMARY_FIELDS = [
    "support_score",
    "measured_score",
    "documentation_score",
    "example_score",
    "build_score",
    "debugging_score",
    "efficiency_score",
    "prompt_tokens",
    "completion_tokens",
]

DIMENSION_TABLES = {
    "discoverability": [
        "doc_search",
        "navigation_depth",
        "multi_entry_access",
        "version_lookup",
    ],
    "consistency": [
        "document_structure",
        "concept_alignment",
        "interface_layering",
    ],
    "accuracy": [
        "doc_correctness",
        "version_matrix",
        "build_signal_accuracy",
    ],
    "completeness": [
        "document_coverage",
        "sample_coverage",
        "tool_coverage",
        "deliverable_completeness",
    ],
    "learnability": [
        "progressive_guidance",
        "cognitive_load",
        "key_path_visibility",
    ],
    "practicability": [
        "one_shot_success",
        "example_reuse",
        "operation_steps",
        "deployment_readiness",
    ],
    "debuggability": [
        "error_context",
        "remediation_hint",
        "signal_to_noise",
    ],
}

REPRESENTATIVE_CASES = [
    "test/samples/AddPtr",
    "test/samples/AllocTile",
    "test/samples/AsyncComm",
    "test/samples/Bf16",
    "test/samples/CommSync",
    "test/samples/Complex",
    "test/samples/ControlFlow",
    "test/samples/Cvt",
    "test/samples/Dequant",
    "test/samples/DynamicTailMatmul",
    "test/samples/FFN",
    "test/samples/FlashAttention",
    "test/samples/Gather",
    "test/samples/Gemv",
    "test/samples/GQA",
    "test/samples/LayoutInference",
    "test/samples/MatMul",
    "test/samples/Mgather",
    "test/samples/Mscatter",
    "test/samples/Partition5D",
    "test/samples/PyPTOIRParser",
    "test/samples/Quant",
    "test/samples/Qwen3DecodeA3",
    "test/samples/Qwen3DecodeA5",
    "test/samples/Scatter",
    "test/samples/SetValidShape",
    "test/samples/Sync",
    "test/samples/SyncAll",
    "test/samples/TPrefetch",
    "test/samples/TPrefetchAsync",
    "test/samples/TPushTPop",
    "test/samples/planmemory",
]


def _build_evaluation_json_schema():
    dimension_properties = {}
    for dimension_name, subdimensions in DIMENSION_TABLES.items():
        dimension_properties[dimension_name] = {
            "type": "object",
            "required": subdimensions,
            "properties": {
                subdimension: {"type": "integer", "minimum": 1, "maximum": 10}
                for subdimension in subdimensions
            },
        }

    return {
        "type": "object",
        "required": TOP_LEVEL_FIELDS,
        "properties": {
            "evaluation_id": {"type": "string"},
            "operator_name": {"type": "string"},
            "batch_id": {"type": "string"},
            "development_success": {"type": "boolean"},
            "test_results": {
                "type": "object",
                "required": [
                    "level_0",
                    "level_1",
                    "level_2",
                    "level_3",
                    "passed_count",
                    "total_count",
                    "pass_rate",
                ],
            },
            "summary": {
                "type": "object",
                "required": SUMMARY_FIELDS,
            },
            "documentation_retrieval": {
                "type": "object",
                "required": [
                    "total_searches",
                    "effective_searches",
                    "effectiveness_rate",
                ],
            },
            "code_examples": {
                "type": "object",
                "required": [
                    "sampled_case_count",
                    "sampled_cases",
                    "modified_examples",
                    "modification_rate",
                ],
            },
            "build_configuration": {
                "type": "object",
                "required": ["config_lines", "macro_count", "macros"],
            },
            "functional_testing": {
                "type": "object",
                "required": [
                    "compile_run_count",
                    "cycles",
                    "sampled_case_count",
                    "sampled_cases",
                ],
            },
            "key_findings": {"type": "array"},
            "dimension_tables": {
                "type": "object",
                "required": list(DIMENSION_TABLES.keys()),
                "properties": dimension_properties,
            },
            "evaluated_at": {"type": "string"},
            "report_file": {"type": "string"},
        },
    }


def _clamp(number, lower_bound, upper_bound):
    return max(lower_bound, min(upper_bound, number))


def _safe_ratio(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _postprocess_evaluation_json(payload):
    processed = copy.deepcopy(payload)

    test_results = processed["test_results"]
    level_passed_total = 0
    level_case_total = 0
    for level_name in ("level_0", "level_1", "level_2", "level_3"):
        level_result = test_results[level_name]
        level_result["passed_count"] = max(0, int(level_result.get("passed_count", 0)))
        level_result["total_count"] = max(0, int(level_result.get("total_count", 0)))
        if level_result["passed_count"] > level_result["total_count"]:
            level_result["passed_count"] = level_result["total_count"]
        level_passed_total += level_result["passed_count"]
        level_case_total += level_result["total_count"]

    test_results["passed_count"] = level_passed_total
    test_results["total_count"] = level_case_total
    test_results["pass_rate"] = round(
        _clamp(_safe_ratio(level_passed_total, level_case_total), 0.0, 1.0), 4
    )

    summary = processed["summary"]
    for score_field in SUMMARY_FIELDS:
        if score_field.endswith("_tokens"):
            summary[score_field] = max(0, int(summary.get(score_field, 0)))
        else:
            summary[score_field] = round(
                _clamp(float(summary.get(score_field, 1)), 1.0, 10.0), 2
            )

    documentation = processed["documentation_retrieval"]
    documentation["total_searches"] = max(0, int(documentation.get("total_searches", 0)))
    documentation["effective_searches"] = max(
        0, int(documentation.get("effective_searches", 0))
    )
    if documentation["effective_searches"] > documentation["total_searches"]:
        documentation["effective_searches"] = documentation["total_searches"]
    documentation["effectiveness_rate"] = round(
        _clamp(
            _safe_ratio(
                documentation["effective_searches"], documentation["total_searches"]
            ),
            0.0,
            1.0,
        ),
        4,
    )

    code_examples = processed["code_examples"]
    code_examples["sampled_case_count"] = max(
        0, int(code_examples.get("sampled_case_count", 0))
    )
    code_examples["modified_examples"] = max(
        0, int(code_examples.get("modified_examples", 0))
    )
    if code_examples["modified_examples"] > code_examples["sampled_case_count"]:
        code_examples["modified_examples"] = code_examples["sampled_case_count"]
    code_examples["modification_rate"] = round(
        _clamp(
            _safe_ratio(
                code_examples["modified_examples"], code_examples["sampled_case_count"]
            ),
            0.0,
            1.0,
        ),
        4,
    )

    build_configuration = processed["build_configuration"]
    build_configuration["config_lines"] = max(
        0, int(build_configuration.get("config_lines", 0))
    )
    build_configuration["macro_count"] = max(
        0, int(build_configuration.get("macro_count", 0))
    )

    functional_testing = processed["functional_testing"]
    functional_testing["compile_run_count"] = max(
        0, int(functional_testing.get("compile_run_count", 0))
    )
    functional_testing["cycles"] = max(0, int(functional_testing.get("cycles", 0)))
    functional_testing["sampled_case_count"] = max(
        0, int(functional_testing.get("sampled_case_count", 0))
    )

    dimension_tables = processed["dimension_tables"]
    for dimension_name, subdimensions in DIMENSION_TABLES.items():
        for subdimension in subdimensions:
            dimension_tables[dimension_name][subdimension] = int(
                _clamp(
                    int(dimension_tables[dimension_name].get(subdimension, 1)), 1, 10
                )
            )

    return processed


def _validate_evaluation_json(payload):
    missing_top_level = [field for field in TOP_LEVEL_FIELDS if field not in payload]
    if missing_top_level:
        raise ValueError(f"missing top-level fields: {missing_top_level}")

    for string_field in (
        "evaluation_id",
        "operator_name",
        "batch_id",
        "evaluated_at",
        "report_file",
    ):
        if not isinstance(payload[string_field], str):
            raise ValueError(f"{string_field} must be a string")

    if not isinstance(payload["development_success"], bool):
        raise ValueError("development_success must be a bool")

    summary = payload["summary"]
    missing_summary_fields = [field for field in SUMMARY_FIELDS if field not in summary]
    if missing_summary_fields:
        raise ValueError(f"missing summary fields: {missing_summary_fields}")

    dimension_tables = payload["dimension_tables"]
    missing_dimensions = [
        field for field in DIMENSION_TABLES.keys() if field not in dimension_tables
    ]
    if missing_dimensions:
        raise ValueError(f"missing dimensions: {missing_dimensions}")

    efficiency_score = float(summary["efficiency_score"])
    if not 1.0 <= efficiency_score <= 10.0:
        raise ValueError(f"efficiency_score out of range: {efficiency_score}")

    pass_rate = float(payload["test_results"]["pass_rate"])
    if not 0.0 <= pass_rate <= 1.0:
        raise ValueError(f"pass_rate out of range: {pass_rate}")

    count_fields = [
        payload["test_results"]["passed_count"],
        payload["test_results"]["total_count"],
        payload["documentation_retrieval"]["total_searches"],
        payload["documentation_retrieval"]["effective_searches"],
        payload["code_examples"]["sampled_case_count"],
        payload["code_examples"]["modified_examples"],
        payload["build_configuration"]["config_lines"],
        payload["build_configuration"]["macro_count"],
        payload["functional_testing"]["compile_run_count"],
        payload["functional_testing"]["cycles"],
        payload["functional_testing"]["sampled_case_count"],
        summary["prompt_tokens"],
        summary["completion_tokens"],
    ]
    if any(int(value) < 0 for value in count_fields):
        raise ValueError("token/count fields must be non-negative")

    for dimension_name, subdimensions in DIMENSION_TABLES.items():
        dimension_values = payload["dimension_tables"][dimension_name]
        if not 3 <= len(dimension_values) <= 4:
            raise ValueError(
                f"dimension {dimension_name} must contain 3-4 subdimensions"
            )
        for subdimension in subdimensions:
            subdimension_score = dimension_values[subdimension]
            if not isinstance(subdimension_score, int) or not 1 <= subdimension_score <= 10:
                raise ValueError(
                    f"subdimension score out of range: {dimension_name}.{subdimension}"
                )

    test_results = payload["test_results"]
    if int(test_results["passed_count"]) > int(test_results["total_count"]):
        raise ValueError("passed_count cannot exceed total_count")

    recomputed_pass_rate = round(
        _safe_ratio(int(test_results["passed_count"]), int(test_results["total_count"])),
        4,
    )
    if abs(recomputed_pass_rate - float(test_results["pass_rate"])) > 0.01:
        raise ValueError(
            f"pass_rate mismatch: expected {recomputed_pass_rate}, got {test_results['pass_rate']}"
        )

    documentation = payload["documentation_retrieval"]
    if int(documentation["effective_searches"]) > int(documentation["total_searches"]):
        raise ValueError("effective_searches cannot exceed total_searches")


def _existing_representative_cases(repo_root):
    existing_cases = []
    for case_path in REPRESENTATIVE_CASES:
        if (repo_root / case_path).exists():
            existing_cases.append(case_path)
    return existing_cases


def _build_template_payload(repo_root, output_path):
    representative_cases = _existing_representative_cases(repo_root)
    representative_case_count = len(representative_cases)

    try:
        report_file = output_path.relative_to(repo_root).as_posix()
    except ValueError:
        report_file = output_path.as_posix()

    payload = {
        "evaluation_id": "ptoas-touchpoint-eval-template",
        "operator_name": "PTOAS",
        "batch_id": f"representative-samples-{representative_case_count}",
        "development_success": False,
        "test_results": {
            "level_0": {
                "label": "L1 文档审阅层",
                "passed_count": 0,
                "total_count": 0,
            },
            "level_1": {
                "label": "L2 本地最小运行层",
                "passed_count": 0,
                "total_count": 0,
            },
            "level_2": {
                "label": "L3 Linux compile-only 层",
                "passed_count": 0,
                "total_count": 0,
            },
            "level_3": {
                "label": "L4 NPU 上板层",
                "passed_count": 0,
                "total_count": 0,
            },
            "passed_count": 0,
            "total_count": 0,
            "pass_rate": 0.0,
        },
        "summary": {
            "support_score": 5,
            "measured_score": 5,
            "documentation_score": 5,
            "example_score": 5,
            "build_score": 5,
            "debugging_score": 5,
            "efficiency_score": 5,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        },
        "documentation_retrieval": {
            "total_searches": 0,
            "effective_searches": 0,
            "effectiveness_rate": 0.0,
            "tracked_queries": [],
        },
        "code_examples": {
            "sampled_case_count": representative_case_count,
            "sampled_cases": representative_cases,
            "modified_examples": 0,
            "modification_rate": 0.0,
        },
        "build_configuration": {
            "config_lines": 0,
            "macro_count": 0,
            "macros": [],
        },
        "functional_testing": {
            "compile_run_count": 0,
            "cycles": 0,
            "sampled_case_count": representative_case_count,
            "sampled_cases": representative_cases,
        },
        "key_findings": [
            {
                "rank": 1,
                "finding": "This file is a starter template. Replace placeholder scores and counts after actual touchpoint scoring.",
                "category": "process",
            },
            {
                "rank": 2,
                "finding": "Representative case pack is preselected from PTOAS samples to cover compile, validation, shape, sync, layout, precision and model-style paths.",
                "category": "sampling",
            },
            {
                "rank": 3,
                "finding": "Use real retrieval counts, compile cycles and run outcomes before treating this JSON as a measured report.",
                "category": "validation",
            },
        ],
        "dimension_tables": {
            "discoverability": {
                "doc_search": 5,
                "navigation_depth": 5,
                "multi_entry_access": 5,
                "version_lookup": 5,
            },
            "consistency": {
                "document_structure": 5,
                "concept_alignment": 5,
                "interface_layering": 5,
            },
            "accuracy": {
                "doc_correctness": 5,
                "version_matrix": 5,
                "build_signal_accuracy": 5,
            },
            "completeness": {
                "document_coverage": 5,
                "sample_coverage": 5,
                "tool_coverage": 5,
                "deliverable_completeness": 5,
            },
            "learnability": {
                "progressive_guidance": 5,
                "cognitive_load": 5,
                "key_path_visibility": 5,
            },
            "practicability": {
                "one_shot_success": 5,
                "example_reuse": 5,
                "operation_steps": 5,
                "deployment_readiness": 5,
            },
            "debuggability": {
                "error_context": 5,
                "remediation_hint": 5,
                "signal_to_noise": 5,
            },
        },
        "evaluated_at": "1970-01-01T00:00:00Z",
            "report_file": report_file,
    }

    processed = _postprocess_evaluation_json(payload)
    _validate_evaluation_json(processed)
    return processed


def main():
    parser = argparse.ArgumentParser(
        description="Generate PTOAS touchpoint evaluation JSON template."
    )
    parser.add_argument(
        "--repo-root",
        default=str(Path(__file__).resolve().parents[3]),
        help="PTOAS repository root.",
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parents[1]
            / "assets"
            / "ptoas_touchpoint_evaluation_template.json"
        ),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--emit-schema",
        action="store_true",
        help="Print the JSON schema-like contract instead of the template payload.",
    )
    arguments = parser.parse_args()

    output_path = Path(arguments.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if arguments.emit_schema:
        payload = _build_evaluation_json_schema()
    else:
        repo_root = Path(arguments.repo_root).resolve()
        payload = _build_template_payload(repo_root, output_path)

    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")

    print(output_path)


if __name__ == "__main__":
    main()
