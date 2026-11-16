from pathlib import Path

p = Path("Interviewer/answer_understanding_v58.py")
s = p.read_text(encoding="utf-8")

# 1. "I know basic C#" remains a claim.
old = '''    knowledge_only = bool(re.search(r"\\b(?:i|we)\\s+(?:understand|know|am aware|are aware|am familiar|are familiar)\\b", cl)) and not re.search(r"\\b(?:built|created|developed|implemented|wrote|trained|used|applied|worked on)\\b", cl)'''

new = '''    knowledge_only = bool(re.search(r"\\b(?:i|we)\\s+(?:understand|am aware|are aware|am familiar|are familiar)\\b", cl)) and not re.search(r"\\b(?:built|created|developed|implemented|wrote|trained|used|applied|worked on)\\b", cl)'''

if s.count(old) != 2:
    raise SystemExit(f"Expected knowledge_only condition twice; found {s.count(old)}. No changes made.")

s = s.replace(old, new)

# 2. "I need to learn more" is not evidence of active learning.
old_learn = r'|\blearn(?:ing)?\s+more\b|\bto learn\b'
new_learn = r''

if old_learn not in s:
    raise SystemExit("Expected broad learning patterns not found. No changes made.")

s = s.replace(old_learn, new_learn, 1)

p.write_text(s, encoding="utf-8")

# 3. Keep the curated YOLO -> Computer Vision implication,
# while limiting unrelated implications to tracked competencies.
p = Path("Interviewer/answer_understanding_v58.py")
s = p.read_text(encoding="utf-8")

old = '''                if k2 in tracked and k2 not in claims and w >= 0.3:'''

new = '''                if (k2 in tracked or (key == "yolo" and k2 == "cv")) and k2 not in claims and w >= 0.3:'''

if old not in s:
    raise SystemExit("Expected implication condition not found. Learning/knowledge edits were applied; implication edit was not.")

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("Competency test fixes applied.")