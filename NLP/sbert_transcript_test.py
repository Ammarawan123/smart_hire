import os
from SBERT.sbert_analyzer import SBERTAnalyzer

TRANSCRIPT_PATH = os.path.join(
    "recordings",
    "transcript_20261017_161649.txt"
)

def main():
    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"Transcript not found: {TRANSCRIPT_PATH}")
        return

    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    if not lines:
        print("The transcript is empty.")
        return

    print("Loading your existing SBERT analyzer...")
    analyzer = SBERTAnalyzer()

    question = "Can you introduce yourself and describe your background?"
    ideal_answer = (
        "My name is Abdul Ahad. I studied at a university and "
        "have experience working as a data scientist."
    )

    print("\n========== SBERT TRANSCRIPT ANALYSIS ==========")
    print(f"Transcript: {TRANSCRIPT_PATH}")
    print(f"Number of transcript lines: {len(lines)}")

    for index, answer in enumerate(lines, start=1):
        print(f"\n--- Transcript Line {index} ---")
        print(f"Text: {answer}")

        question_score = analyzer.question_answer_similarity(
            question, answer
        )
        ideal_score = analyzer.answer_ideal_similarity(
            answer, ideal_answer
        )
        embedding = analyzer.get_embedding(answer)

        print(f"Question similarity: {question_score:.4f}")
        print(f"Reference answer similarity: {ideal_score:.4f}")
        print(f"Embedding dimensions: {len(embedding)}")

    print("\nAnalysis completed successfully.")

if __name__ == "__main__":
    main()
