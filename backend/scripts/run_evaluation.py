import argparse
import logging
import sys
from pathlib import Path

from app.config.settings import get_settings
from app.container import create_application_container
from app.evaluation.dataset import load_cases, resolve_document_ids
from app.evaluation.service import EvaluationService


def main() -> None:
    args = _parse_args()

    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    print("=== Evaluation Harness ===")

    container = create_application_container(settings)

    indexed_page_count = container.initialize()

    print(f"Indexed pages loaded: {indexed_page_count}")

    print(f"\nLoading golden dataset: {args.dataset}")

    cases = load_cases(args.dataset)

    print(f"Loaded {len(cases)} case(s).")

    try:
        cases = resolve_document_ids(cases, container.repository)
    except ValueError as exc:
        print(f"\nERROR: {exc}")
        print(
            "\nIngest the referenced document(s) first (POST /documents "
            "or IngestionService.ingest), then rerun this script."
        )
        sys.exit(1)

    evaluation_service = EvaluationService(
        retrieval_service=container.retrieval_service,
        agent_orchestrator=container.agent_orchestrator,
    )

    print("\n--- Retrieval evaluation (no LLM, deterministic) ---")

    retrieval_summary = evaluation_service.evaluate_retrieval(cases)

    for result in retrieval_summary.results:
        status = "PASS" if result.hit else "FAIL"

        print(
            f"[{status}] {result.case_id}: recall={result.recall:.2f} "
            f"expected={list(result.expected_pages)} "
            f"retrieved={list(result.retrieved_pages)}"
        )

    print(
        f"\nMean recall: {retrieval_summary.mean_recall:.2f}  "
        f"Hit rate: {retrieval_summary.hit_rate:.2f}"
    )

    retrieval_passed = retrieval_summary.hit_rate == 1.0

    if not args.full:
        _print_final_status(retrieval_passed)
        sys.exit(0 if retrieval_passed else 1)

    print(
        "\n--- Full agent evaluation (requires a reachable "
        "generation provider, e.g. Ollama) ---"
    )

    try:
        agent_summary = evaluation_service.evaluate_agent(cases)
    except Exception as exc:
        print(
            f"\nERROR: Full agent evaluation could not complete: {exc}"
        )
        print(
            "This usually means the configured generation provider "
            "is unreachable -- for the default Ollama setup, run "
            "'ollama serve' and confirm 'curl "
            "http://localhost:11434/api/tags' responds, then retry."
        )
        print(
            "\nRetrieval evaluation above is unaffected and still "
            "reflects real results."
        )
        sys.exit(2)

    for result in agent_summary.results:
        status = "PASS" if result.page_hit else "FAIL"

        print(
            f"[{status}] {result.case_id}: "
            f"iterations={result.iterations} "
            f"cited={list(result.cited_pages)} "
            f"keyword_coverage={result.keyword_coverage:.2f}"
        )
        print(f"       answer: {result.answer[:200]!r}")

    print(
        f"\nPage hit rate: {agent_summary.page_hit_rate:.2f}  "
        f"Mean keyword coverage: "
        f"{agent_summary.mean_keyword_coverage:.2f}  "
        f"Mean iterations: {agent_summary.mean_iterations:.2f}"
    )

    agent_passed = agent_summary.page_hit_rate == 1.0

    _print_final_status(retrieval_passed and agent_passed)
    sys.exit(0 if (retrieval_passed and agent_passed) else 1)


def _print_final_status(passed: bool) -> None:
    print("\n=== " + ("PASS" if passed else "FAIL") + " ===")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the golden-dataset evaluation against a live, "
            "already-ingested system."
        ),
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Path to a golden-dataset JSON file. Defaults to "
            "app/evaluation/golden_dataset.json."
        ),
    )

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Also run the full agent pipeline (LLM-backed decision, "
            "refine, and answer generation), not just retrieval. "
            "Requires a reachable generation provider."
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()