from pathlib import Path

p = Path("Interviewer/answer_understanding_v58.py")
s = p.read_text(encoding="utf-8")

old = r'''_FUTURE_LEARN = re.compile(r"\b(?:need|want|have|has|going|plan|hope|like|try|trying|should|will|would|wish|ready|eager|willing)\s+(?:to\s+)?(?:learn|study|read|know)\w*|\blearn(?:ing)?\s+more\b|\bto learn\b")'''

new = r'''_FUTURE_LEARN = re.compile(r"\b(?:need|want|have|has|going|plan|hope|like|try|trying|should|will|would|wish|ready|eager|willing|keen|interested)\s+(?:to\s+)?(?:learn|study|read|know)\w*|\b(?:eager|willing|keen|interested)\b[^.!?]{0,80}\blearn\w*|\blearn(?:ing)?\s+more\b|\bto learn\b")'''

if old not in s:
    raise SystemExit("Expected pattern not found. No change made.")

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("Future-learning pattern updated successfully.")