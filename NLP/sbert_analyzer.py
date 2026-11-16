
from sentence_transformers import SentenceTransformer


class SBERTAnalyzer:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print("Loading SBERT model...")
        try:
            import config as _root
            if model_name == "all-MiniLM-L6-v2":
                model_name = _root.local_or(_root.SBERT_DIR, model_name)
        except Exception:
            pass
        self.model = SentenceTransformer(model_name)
        print("SBERT model loaded successfully!")

    def get_embedding(self, text):
        """
        Convert text into a 384-dimensional SBERT embedding.
        """
        return self.model.encode(text)

    def question_answer_similarity(self, question, answer):
        """
        Calculate semantic similarity between a question and answer.
        """
        question_embedding = self.model.encode(question)
        answer_embedding = self.model.encode(answer)

        similarity = self.model.similarity(
            question_embedding,
            answer_embedding
        )

        return float(similarity.item())

    def answer_ideal_similarity(self, answer, ideal_answer):
        """
        Calculate semantic similarity between a candidate answer
        and an ideal/reference answer.
        """
        answer_embedding = self.model.encode(answer)
        ideal_embedding = self.model.encode(ideal_answer)

        similarity = self.model.similarity(
            answer_embedding,
            ideal_embedding
        )

        return float(similarity.item())

    def semantic_consistency(self, answer_1, answer_2):
        """
        Calculate semantic similarity between two answers.
        """
        embedding_1 = self.model.encode(answer_1)
        embedding_2 = self.model.encode(answer_2)

        similarity = self.model.similarity(
            embedding_1,
            embedding_2
        )

        return float(similarity.item())

    def analyze_answer(self, question, answer, ideal_answer=None):
        """
        Perform complete SBERT analysis on a candidate answer.

        Returns:
            A dictionary containing:
            - Question-answer similarity
            - Candidate vs ideal answer similarity (if provided)
            - 384-dimensional text embedding
        """

        # Question vs Candidate Answer
        question_answer_score = self.question_answer_similarity(
            question,
            answer
        )

        # Generate text embedding
        embedding = self.get_embedding(answer)

        # Store the main results
        result = {
            "question_answer_similarity": question_answer_score,
            "embedding": embedding
        }

        # Candidate Answer vs Ideal Answer
        if ideal_answer is not None:
            ideal_score = self.answer_ideal_similarity(
                answer,
                ideal_answer
            )

            result["ideal_answer_similarity"] = ideal_score

        return result

