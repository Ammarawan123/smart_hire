
from sbert_analyzer import SBERTAnalyzer


# Create SBERT analyzer
analyzer = SBERTAnalyzer()


# Interview question
question = "What experience do you have with Python?"


# Candidate's answer
candidate_answer = """
I have used Python for machine learning, data analysis,
and computer vision projects.
"""


# Ideal/reference answer
ideal_answer = """
I have experience using Python for machine learning,
data analysis, and computer vision.
"""


# Perform complete SBERT analysis
result = analyzer.analyze_answer(
    question=question,
    answer=candidate_answer,
    ideal_answer=ideal_answer
)


# Display results
print("\n========== SBERT ANALYSIS ==========")

print("\nQuestion:")
print(question)

print("\nCandidate Answer:")
print(candidate_answer)

print("\nQuestion-Answer Similarity:")
print(result["question_answer_similarity"])

print("\nCandidate vs Ideal Answer Similarity:")
print(result["ideal_answer_similarity"])

print("\nText Feature Vector Size:")
print(len(result["embedding"]))

print("\n====================================")

