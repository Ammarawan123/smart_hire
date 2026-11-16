

import json
from pathlib import Path
from datetime import datetime

from SBERT.sbert_analyzer import SBERTAnalyzer


# Input: the saved SmartHire V58 interview report
INPUT_REPORT = Path(
    "SmartHire_V58_032005/interview_report_v58.json"
)

# Output: a separate SBERT analysis report
OUTPUT_REPORT = Path("sbert_real_interview_analysis.json")


def main():
    # Check that the input report exists
    if not INPUT_REPORT.exists():
        print(f"ERROR: Interview report not found: {INPUT_REPORT}")
        print("Check that you are running this script from the SmartHire folder.")
        return

    # Load the saved interview report
    try:
        with INPUT_REPORT.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (json.JSONDecodeError, OSError) as error:
        print(f"ERROR: Could not read the interview report: {error}")
        return

    transcript = report.get("transcript", [])

    print(f"Loaded {len(transcript)} interview turns.")

    if not transcript:
        print("No transcript entries found.")
        return

    # Load the SBERT model
    print("Loading SBERT model...")

    try:
        analyzer = SBERTAnalyzer()
    except Exception as error:
        print(f"ERROR: Could not load SBERT: {error}")
        return

    results = []

    # Analyze each interview question and answer
    for entry in transcript:
        question = str(entry.get("interviewer") or "").strip()
        answer = str(entry.get("answer") or "").strip()

        # Skip entries without a question or answer
        if not question or not answer:
            print(f"Skipping turn {entry.get('turn')}: missing question or answer.")
            continue

        # Skip the closing turn
        if entry.get("phase") == "closing" or entry.get("kind") == "closing":
            print(f"Skipping closing turn {entry.get('turn')}.")
            continue

        try:
            # Measure semantic similarity between question and answer
            score = analyzer.question_answer_similarity(
                question,
                answer
            )

            # Generate a 384-dimensional answer embedding
            embedding = analyzer.get_embedding(answer)

        except Exception as error:
            print(f"Error analyzing turn {entry.get('turn')}: {error}")
            continue

        # Get SmartHire's existing interpretation, if available
        interpretation = entry.get("interpretation") or {}

        # Keep the original interpretation as well as selected fields
        if not isinstance(interpretation, dict):
            interpretation = {}

        result = {
            "turn": entry.get("turn"),
            "phase": entry.get("phase"),
            "kind": entry.get("kind"),
            "targets": entry.get("targets", []),
            "question": question,
            "answer": answer,
            "question_answer_similarity": score,
            "embedding_dimensions": len(embedding),

            # No reference answers are available in these transcript entries
            "reference_answer_similarity": None,

            # Existing SmartHire analysis; not an SBERT score
            "existing_interpretation": {
                "intent": interpretation.get("intent"),
                "quality": interpretation.get("quality"),
                "specificity": interpretation.get("specificity"),
                "question_answered": interpretation.get("question_answered"),
                "missing_information": interpretation.get(
                    "missing_information", []
                ),
            },

            # Preserve the full existing interpretation for later analysis
            "full_interpretation": interpretation,

            # Preserve SmartHire's existing evidence changes
            "evidence_changes": entry.get("evidence_changes", []),
        }

        results.append(result)

        print(
            f"Turn {entry.get('turn')}: "
            f"similarity={score:.4f}, "
            f"kind={entry.get('kind')}, "
            f"targets={entry.get('targets', [])}, "
            f"embedding={len(embedding)} dimensions"
        )

    # Build the output report
    output = {
        "source_report": str(INPUT_REPORT),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "analyzer": "all-MiniLM-L6-v2",
        "analyzed_turns": len(results),
        "note": (
            "Question-answer similarity measures semantic similarity, "
            "not correctness, answer quality, or a hiring score. "
            "Reference-answer similarity was not calculated because "
            "ideal/reference answers were not available in the saved "
            "transcript. Existing SmartHire interpretations are preserved "
            "separately and are not SBERT-generated scores."
        ),
        "results": results,
    }

    # Save the report without changing the original SmartHire report
    try:
        OUTPUT_REPORT.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as error:
        print(f"ERROR: Could not save output report: {error}")
        return

    print(f"\nSaved results to {OUTPUT_REPORT}")
    print(f"Successfully analyzed {len(results)} interview turns.")
    print("Original SmartHire files were not modified.")


if __name__ == "__main__":
    main()

