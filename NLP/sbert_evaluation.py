
import json
from pathlib import Path

from SBERT.sbert_analyzer import SBERTAnalyzer


def concept_coverage(answer, key_concepts):
    """Estimate how many required concepts are mentioned."""
    answer_lower = answer.lower()

    if not key_concepts:
        return None, []

    covered = [
        concept
        for concept in key_concepts
        if concept.lower() in answer_lower
    ]

    score = len(covered) / len(key_concepts)

    return round(score, 4), covered


def evaluate_answer(
    analyzer,
    question,
    answer,
    ideal_answer,
    key_concepts
):
    """Evaluate one answer using existing SBERT functions."""

    relevance = analyzer.question_answer_similarity(
        question, answer
    )

    reference_similarity = analyzer.answer_ideal_similarity(
        answer, ideal_answer
    )

    coverage, covered_concepts = concept_coverage(
        answer, key_concepts
    )

    return {
        "question": question,
        "answer": answer,
        "question_relevance_similarity": round(relevance, 4),
        "reference_answer_similarity": round(
            reference_similarity, 4
        ),
        "concept_coverage": coverage,
        "covered_concepts": covered_concepts,
        "required_concepts": key_concepts,
        "note": (
            "Similarity and concept coverage are indicators, "
            "not percentages of correctness."
        )
    }

def check_answer_consistency(analyzer, previous_answer, new_answer):
    """Flag potentially inconsistent answers for human review."""

    if not previous_answer.strip() or not new_answer.strip():
        return {
            "similarity": None,
            "review_flag": False,
            "message": "Both answers must contain text."
        }

    similarity = analyzer.semantic_consistency(
        previous_answer,
        new_answer
    )

    return {
        "semantic_similarity": round(float(similarity), 4),
        "review_flag": True,
        "message": (
            "Review both answers for possible contradictions. "
            "Similarity alone cannot establish whether claims conflict."
        ),
        "previous_answer": previous_answer,
        "new_answer": new_answer
    }


def main():
    analyzer = SBERTAnalyzer()

    previous_answer = input(
    "\nEnter the candidate's earlier claim: "
    ).strip()

    new_answer = input(
    "Enter the candidate's later claim: "
    ).strip()

    consistency_report = check_answer_consistency(
        analyzer,
        previous_answer,
        new_answer
    )

    print("\n--- Consistency Review Test ---")
    print(json.dumps(consistency_report, indent=2))


    question = "How do you find and fix a bug in Python?"

    ideal_answer = (
        "I reproduce the error, read the error message, "
        "inspect the relevant code and inputs, identify "
        "the cause, make a change, and test the fix."
    )

    key_concepts = [
        "error message",
        "code",
        "inputs",
        "run the code again"
    ]

    print("\nQuestion:")
    print(question)

    answer = input("\nEnter a candidate answer: ").strip()

    if not answer:
        print("No answer entered. Exiting.")
        return

    report = evaluate_answer(
        analyzer,
        question,
        answer,
        ideal_answer,
        key_concepts
    )

    print("\n--- Evaluation Results ---")
    print(json.dumps(report, indent=2))

    output_path = Path("sbert_evaluation_result.json")
    output_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print(f"\nReport saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
