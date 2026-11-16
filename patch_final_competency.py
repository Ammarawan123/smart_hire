
from pathlib import Path

p = Path("Interviewer/answer_understanding_v58.py")
s = p.read_text(encoding="utf-8")

old = '''    # implied evidence from practical claims (YOLO => Computer Vision ...), only for tracked concepts, never overriding a direct claim
    tracked = set(graph.comp) if graph is not None else set()
    for key, c in list(claims.items()):
        if c.stance == "practical":
            for k2, w in K.closure(key).items():
                if k2 not in claims and w >= 0.3:
                    claims[k2] = Claim(k2, "practical", True, c.ownership, [], c.quote, 0.2, f"implied by {K.name_of(key)}")
    # family credit: SQL answer supports "SQL Server" requirement at reduced weight
    for child, parent in K.FAMILY.items():
        if parent in claims and child not in claims and graph is not None and child in graph.job.requirements and claims[parent].stance in ("practical", "theoretical", "learning"):
            claims[child] = Claim(child, claims[parent].stance, True, claims[parent].ownership, [], claims[parent].quote, 0.2, f"family credit from {K.name_of(parent)}")
'''

new = '''    # Implied evidence is limited to competencies tracked by this interview.
    tracked = set(graph.comp) if graph is not None else set()
    for key, c in list(claims.items()):
        if c.stance == "practical":
            for k2, w in K.closure(key).items():
                if k2 in tracked and k2 not in claims and w >= 0.3:
                    claims[k2] = Claim(k2, "practical", True, c.ownership, [], c.quote, 0.2, f"implied by {K.name_of(key)}")

    # Generic SQL can indicate awareness of a database family, not hands-on
    # experience with a specific vendor product such as SQL Server.
    for child, parent in K.FAMILY.items():
        if parent in claims and child in tracked and child not in claims and claims[parent].stance in ("practical", "theoretical", "learning"):
            claims[child] = Claim(child, "theoretical", True, claims[parent].ownership, [], claims[parent].quote, 0.1, f"general {K.name_of(parent)} knowledge; product-specific experience not established")
'''

if old not in s:
    raise SystemExit("ERROR: Expected code block not found. No changes made.")

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("Final competency patch applied.")

