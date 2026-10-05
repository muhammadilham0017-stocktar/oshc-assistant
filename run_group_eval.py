"""
Run the group's 27 questions through the routing pipeline.

    python run_group_eval.py

This measures routing, not answer text. For each question it reports which
tier the question lands in, and for questions that reach retrieval, which
section comes back first.

Put group_test_set.json in eval/ first.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline import Retriever, safety_gate

try:
    from app.pipeline import crisis_check
except ImportError:
    crisis_check = lambda q: None
try:
    from app.pipeline import wellbeing_check
except ImportError:
    wellbeing_check = lambda q: None

MIN_CONFIDENCE = 0.018
TESTS = json.loads(Path("eval/group_test_set.json").read_text())


def route(q, retriever, product="comprehensive"):
    """Mirrors the order in main.py. Tier 1 first, always."""
    tier = crisis_check(q)
    if tier:
        return tier, None, None
    if wellbeing_check(q):
        return "wellbeing", None, None
    blocked, term = safety_gate(q)
    if blocked:
        return "gate", None, term
    hits = retriever.search(q, product=product, k=3)
    if not hits:
        return "unsure", 0.0, None
    conf = float(hits[0][1])
    if conf < MIN_CONFIDENCE:
        return "unsure", conf, hits[0][0]["section"]
    return "answer", conf, hits[0][0]["section"]


def main():
    r = Retriever()
    rows, mismatch = [], []

    for t in TESTS:
        got, conf, extra = route(t["question"], r)
        want = t["expect"]
        ok = got == want
        rows.append((t, got, want, conf, extra, ok))
        if not ok:
            mismatch.append((t, got, want, extra))

    group = None
    for t, got, want, conf, extra, ok in rows:
        if t["group"] != group:
            group = t["group"]
            print(f"\n--- {group.upper()}")
            print(f"{'id':<5}{'got':<11}{'want':<11}{'conf':<8}{'top section / trigger'}")
            print("-" * 76)
        c = f"{conf:.4f}" if conf is not None else ""
        mark = " " if ok else "*"
        print(f"{mark}{t['id']:<4}{got:<11}{want:<11}{c:<8}{str(extra or '')[:34]}")

    agree = sum(1 for *_, ok in rows if ok)
    print("\n" + "=" * 76)
    print(f"Routing matches the group's expectation: {agree}/{len(rows)}")

    if mismatch:
        print("\nWhere it differs, and why that may be correct:\n")
        for t, got, want, extra in mismatch:
            print(f"  {t['id']}  expected {want}, got {got}")
            print(f"      \"{t['question'][:66]}\"")
            if t.get("note"):
                print(f"      {t['note']}")
            print()

    print("A mismatch is not automatically a bug. Several of these questions")
    print("are genuinely ambiguous, and the group's expectation may be the")
    print("thing that needs changing rather than the code.")


if __name__ == "__main__":
    main()
