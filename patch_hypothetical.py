from pathlib import Path

p = Path("Interviewer/answer_understanding_v58.py")
s = p.read_text(encoding="utf-8")

old = r'''_HYPO = re.compile(r"\b(would|will|i'?ll|i'?d|should|might|suppose|imagine|first i|then i|we can|you can|you would|one can|one would|going to|plan to|want to|i think we|let'?s say|i could|we could|could)\b")'''

new = r'''_HYPO = re.compile(r"\b(i can|i could|i would|i will|would|will|i'?ll|i'?d|should|might|suppose|imagine|first i|then i|we can|you can|you would|one can|one would|going to|plan to|want to|i think we|let'?s say|we could|could)\b")'''

if old not in s:
    raise SystemExit("Expected hypothetical pattern not found. No change made.")

p.write_text(s.replace(old, new, 1), encoding="utf-8")
print("First-person hypothetical detection updated.")