
import json
from pathlib import Path

from sbert_evaluation import concept_coverage
from SBERT.sbert_analyzer import SBERTAnalyzer


TEST_CASES = [
    {
        "label": "Good answer",
        "answer": (
            "I reproduce the problem, read the error message, "
            "inspect the code and inputs, identify the cause, "
            "fix it, and test the solution again."
        ),
    },
    {
        "label": "Paraphrased answer",
        "answer": (
            "I recreate the problem, examine the error details, "
            "check the program and its data, find the cause, "
            "correct it, and run the program again."
        ),
    },
    {
        "label": "Weak answer",
        "answer": "I look at the code and try to fix the bug.",
    },
    {
        "label": "Irrelevant answer",
        "answer": "I enjoy cricket and watching movies.",
    },
]

KEY_CONCEPTS = [
    "error message",
    "code",
    "inputs",
    "run the code again",
]


def main():
    analyzer = SBERTAnalyzer()
    results = []

    print("\n=== CONCEPT COVERAGE COMPARISON ===")

    for case in TEST_CASES:
        score, covered = concept_coverage(
            case["answer"], KEY_CONCEPTS
        )

        reference_similarity = analyzer.answer_ideal_similarity(
            case["answer"],
            (
                "I reproduce the error, read the error message, "
                "inspect the relevant code and inputs, identify "
                "the cause, make a change, and test the fix."
            ),
        )

        result = {
            "type": case["label"],
            "answer": case["answer"],
            "exact_phrase_coverage": score,
            "covered_concepts": covered,
            "reference_similarity": round(
                reference_similarity, 4
            ),
        }
        results.append(result)

        print(f"\n{case['label']}")
        print("Exact phrase coverage:", score)
        print("Covered concepts:", covered)
        print(
            "Reference similarity:",
            round(reference_similarity, 4),
        )

    output_file = (
        Path(__file__).resolve().parent
        / "sbert_concept_coverage_results.json"
    )

    output_file.write_text(
        json.dumps({"results": results}, indent=2),
        encoding="utf-8",
    )

    print("\nSaved results to:", output_file)


if __name__ == "__main__":
    main()
