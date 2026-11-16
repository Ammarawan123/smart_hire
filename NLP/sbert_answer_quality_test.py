
import json
from pathlib import Path
from SBERT.sbert_analyzer import SBERTAnalyzer

BASE_DIR = Path(__file__).resolve().parent
REFERENCE_FILE = BASE_DIR / "sbert_reference_answers.json"
OUTPUT_FILE = BASE_DIR / "sbert_answer_quality_results.json"

TEST_ANSWERS = {
    1: {
        "good": (
            "I am an Artificial Intelligence student. "
            "I use Python for programming, machine learning, "
            "and data analysis in academic projects."
        ),
        "weak": (
            "I study AI and know a little Python."
        ),
        "irrelevant": (
            "My favorite food is biryani, and I enjoy watching movies."
        ),
    },
    2: {
        "good": (
            "I worked on a computer vision project using Python. "
            "I helped prepare the dataset, implement an object "
            "detection model, and evaluate its performance."
        ),
        "weak": (
            "I worked on an AI project with Python."
        ),
        "irrelevant": (
            "I enjoy playing cricket and watching television."
        ),
    },
    3: {
        "good": (
            "I reproduce the error, read the error message, "
            "inspect the code, identify the cause, fix it, "
            "and test the solution."
        ),
        "weak": (
            "I look at the code and try to fix the bug."
        ),
        "irrelevant": (
            "Python is a popular programming language used for websites."
        ),
    },
}


def main():
    with REFERENCE_FILE.open("r", encoding="utf-8") as file:
        questions = json.load(file)["questions"]

    analyzer = SBERTAnalyzer()
    results = []

    print("\n=== SBERT ANSWER QUALITY TEST ===")

    for item in questions:
        question_id = item["id"]
        question = item["question"]
        reference = item["reference_answer"]

        for answer_type, answer in TEST_ANSWERS[question_id].items():
            relevance = analyzer.question_answer_similarity(
                question, answer
            )
            reference_similarity = analyzer.answer_ideal_similarity(
                answer, reference
            )

            result = {
                "question_id": question_id,
                "question": question,
                "answer_type": answer_type,
                "answer": answer,
                "question_answer_similarity": round(
                    relevance, 4
                ),
                "reference_answer_similarity": round(
                    reference_similarity, 4
                ),
            }
            results.append(result)

            print(
                f"\nQ{question_id} | {answer_type.upper()}"
                f"\nQuestion relevance: {relevance:.4f}"
                f"\nReference similarity: {reference_similarity:.4f}"
            )

    report = {
        "results": results,
        "note": (
            "Similarity scores are indicators, not correctness "
            "percentages. Compare patterns across answer types."
        ),
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print(f"\nAll 9 cases completed.")
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
