
import json
from pathlib import Path
from datetime import datetime

from SBERT.sbert_analyzer import SBERTAnalyzer

RECORDINGS_DIR = Path("recordings")

# Sample values for testing. We'll connect these to actual
# interview questions and reference answers in the next stage.
SAMPLE_QUESTION = "Can you introduce yourself and describe your background?"
SAMPLE_REFERENCE = (
    "My name is Abdul Ahad. I studied at a university and "
    "have experience working as a data scientist."
)


def find_latest_transcript():
    transcripts = sorted(
        RECORDINGS_DIR.glob("transcript*_corrected.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return transcripts[0] if transcripts else None


def main():
    transcript_path = find_latest_transcript()

    if transcript_path is None:
        print("No corrected transcript found in the recordings folder.")
        return

    with transcript_path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        print("The selected transcript is empty.")
        return

    print(f"Transcript selected: {transcript_path}")
    print("Loading your existing SBERT analyzer...")
    analyzer = SBERTAnalyzer()

    results = []

    for index, answer in enumerate(lines, start=1):
        question_score = analyzer.question_answer_similarity(
            SAMPLE_QUESTION, answer
        )
        reference_score = analyzer.answer_ideal_similarity(
            answer, SAMPLE_REFERENCE
        )
        embedding = analyzer.get_embedding(answer)

        results.append({
            "line_number": index,
            "text": answer,
            "question_similarity": question_score,
            "reference_answer_similarity": reference_score,
            "embedding_dimensions": len(embedding),
        })

        print(f"\nTranscript line {index}: {answer}")
        print(f"Question similarity: {question_score:.4f}")
        print(f"Reference similarity: {reference_score:.4f}")
        print(f"Embedding dimensions: {len(embedding)}")

    report = {
        "transcript_file": str(transcript_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note": (
            "Similarity scores are not correctness percentages. "
            "The question and reference answer are sample values."
        ),
        "results": results,
    }

    report_path = Path("sbert_integration_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nReport saved to: {report_path}")
    print("SBERT transcript integration test completed.")


if __name__ == "__main__":
    main()
