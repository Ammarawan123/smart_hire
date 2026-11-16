
import json
from pathlib import Path

from SBERT.sbert_analyzer import SBERTAnalyzer


REFERENCE_ANSWER = (
    "I reproduce the error, read the error message, "
    "inspect the relevant code and inputs, identify "
    "the cause, make a change, and test the fix."
)

REQUIRED_CONCEPTS = [
    "error message",
    "code",
    "inputs",
    "run the code again",
]

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


def semantic_concept_coverage(
    analyzer, answer, concepts, threshold=0.40
):
    """Experimental semantic matching, not proof of correctness."""

    if not answer.strip() or not concepts:
        return {
            "score": 0.0 if concepts else None,
            "matched": [],
            "details": [],
        }

    answer_embedding = analyzer.get_embedding(answer)
    matched = []
    details = []

    for concept in concepts:
        concept_embedding = analyzer.get_embedding(concept)

        similarity = analyzer.model.similarity(
            answer_embedding, concept_embedding
        )
        score = float(similarity.item())

        is_matched = score >= threshold

        details.append({
            "concept": concept,
            "similarity": round(score, 4),
            "matched": is_matched,
        })

        if is_matched:
            matched.append(concept)

    coverage = len(matched) / len(concepts)

    return {
        "score": round(coverage, 4),
        "matched": matched,
        "details": details,
    }


def main():
    analyzer = SBERTAnalyzer()
    results = []

    print("\n=== EXPERIMENTAL SEMANTIC CONCEPT TEST ===")
    print("Threshold:", 0.40)
    print("This threshold is experimental, not validated.")

    for case in TEST_CASES:
        answer = case["answer"]

        exact_matches = [
            concept for concept in REQUIRED_CONCEPTS
            if concept.lower() in answer.lower()
        ]

        semantic = semantic_concept_coverage(
            analyzer, answer, REQUIRED_CONCEPTS
        )

        reference_similarity = (
            analyzer.answer_ideal_similarity(
                answer, REFERENCE_ANSWER
            )
        )

        result = {
            "type": case["label"],
            "answer": answer,
            "exact_phrase_coverage": round(
                len(exact_matches) / len(REQUIRED_CONCEPTS), 4
            ),
            "semantic_concept_coverage": semantic["score"],
            "semantic_matches": semantic["matched"],
            "concept_similarity_details": semantic["details"],
            "reference_similarity": round(
                reference_similarity, 4
            ),
        }

        results.append(result)

        print(f"\n{case['label']}")
        print("Exact coverage:", result["exact_phrase_coverage"])
        print("Semantic coverage:", semantic["score"])
        print("Semantic matches:", semantic["matched"])
        print(
            "Reference similarity:",
            result["reference_similarity"],
        )

        for detail in semantic["details"]:
            print(
                f"  {detail['concept']}: "
                f"{detail['similarity']} "
                f"(matched={detail['matched']})"
            )

    output_file = (
        Path(__file__).resolve().parent
        / "sbert_semantic_concept_results.json"
    )

    output_file.write_text(
        json.dumps({"results": results}, indent=2),
        encoding="utf-8",
    )

    print("\nSaved results to:", output_file)
    print(
        "Warning: semantic matches can be false positives. "
        "Do not use these scores as hiring decisions."
    )


if __name__ == "__main__":
    main()
