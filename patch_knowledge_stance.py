from pathlib import Path

path = Path("Interviewer/answer_understanding_v58.py")
source = path.read_text(encoding="utf-8")

old = '''    if neg_know:
        return "unknown", own, 0.0, ""
'''
new = '''    if neg_know:
        return "unknown", own, 0.0, ""
    # Knowledge statements are awareness, not hands-on experience.
    knowledge_only = bool(re.search(r"\\b(?:i|we)\\s+(?:understand|know|am aware|are aware|am familiar|are familiar)\\b", cl)) and not re.search(r"\\b(?:built|created|developed|implemented|wrote|trained|used|applied|worked on)\\b", cl)
    if knowledge_only:
        return "theoretical", own, 0.1, "knowledge or conceptual understanding"
'''

if source.count(old) != 1:
    raise SystemExit("STOP: first anchor is not unique; no changes made.")

source = source.replace(old, new, 1)

old2 = '''            for cl in clauses:
                if _FIRST.search(cl) and re.search(r"\\b" + _BUILD + r"\\b", cl) and not _NEG_USE.search(cl) and not _HYPO.search(cl) and K.facets_hit(key, cl) and not re.search(r"\\bi (?:know|learned|studied)\\b", cl):
'''
new2 = '''            for cl in clauses:
                knowledge_only = bool(re.search(r"\\b(?:i|we)\\s+(?:understand|know|am aware|are aware|am familiar|are familiar)\\b", cl)) and not re.search(r"\\b(?:built|created|developed|implemented|wrote|trained|used|applied|worked on)\\b", cl)
                if _FIRST.search(cl) and re.search(r"\\b" + _BUILD + r"\\b", cl) and not knowledge_only and not _NEG_USE.search(cl) and not _HYPO.search(cl) and K.facets_hit(key, cl) and not re.search(r"\\bi (?:know|learned|studied)\\b", cl):
'''

if source.count(old2) != 1:
    raise SystemExit("STOP: second anchor is not unique; no changes made.")

source = source.replace(old2, new2, 1)
path.write_text(source, encoding="utf-8")
print("Knowledge-statement classification patch applied.")