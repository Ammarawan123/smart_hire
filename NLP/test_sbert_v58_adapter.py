
import json

from sbert_v58_adapter import SBERTV58Adapter


REPORT_PATH = "SmartHire_V58_032005/interview_report_v58.json"


def main():
    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    transcript = report.get("transcript", [])

    # Find the first turn with both a question and an answer.
    entry = next(
        (
            item for item in transcript
            if item.get("interviewer", "").strip()
            and item.get("answer", "").strip()
            and item.get("phase") != "closing"
            and item.get("kind") != "closing"
        ),
        None,
    )

    if entry is None:
        print("No completed interview turn found.")
        return

    print("Loading SBERT adapter...")
    adapter = SBERTV58Adapter()

    result = adapter.analyze_turn(
        question=entry["interviewer"],
        answer=entry["answer"],
        targets=entry.get("targets", []),
        kind=entry.get("kind"),
        phase=entry.get("phase"),
    )

    print("\nSBERT V58 adapter test successful.")
    print("Turn:", entry.get("turn"))
    print("Question type:", result["kind"])
    print("Phase:", result["phase"])
    print("Targets:", result["targets"])
    print(
        "Question-answer similarity:",
        round(result["question_answer_similarity"], 4),
    )
    print("Embedding dimensions:", result["embedding_dimensions"])
    print("Reference-answer score:", result["reference_answer_similarity"])

    with open(
        "sbert_v58_adapter_test_result.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("\nSaved: sbert_v58_adapter_test_result.json")


if __name__ == "__main__":
    main()
