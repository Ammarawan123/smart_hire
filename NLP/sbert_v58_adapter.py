
from SBERT.sbert_analyzer import SBERTAnalyzer


class SBERTV58Adapter:
    """Separate SBERT adapter for SmartHire V58 question-answer pairs."""

    def __init__(self):
        self.analyzer = SBERTAnalyzer()

    def analyze_turn(
        self,
        question,
        answer,
        targets=None,
        kind=None,
        phase=None,
    ):
        """Analyze one real interview question and candidate answer."""

        question = str(question or "").strip()
        answer = str(answer or "").strip()
        targets = list(targets or [])

        if not question:
            raise ValueError("Question cannot be empty.")

        if not answer:
            raise ValueError("Answer cannot be empty.")

        similarity = self.analyzer.question_answer_similarity(
            question, answer
        )

        embedding = self.analyzer.get_embedding(answer)

        return {
            "question": question,
            "answer": answer,
            "question_answer_similarity": similarity,
            "embedding_dimensions": len(embedding),
            "targets": targets,
            "kind": kind,
            "phase": phase,
            "reference_answer_similarity": None,
            "note": (
                "Question-answer similarity is not a correctness "
                "percentage. No reference answer was provided."
            ),
        }
