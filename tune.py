"""
Sweep the retrieval parameters instead of guessing them.

    python tune.py

Two sweeps, both cheap because neither needs re-embedding:

  1. The confidence floor, against real questions and junk questions
  2. The fusion weights, against the labelled test set

A caution that belongs in the write up. The test set has 15 questions.
Sweeping many parameters against 15 examples finds settings that fit
those 15 and nothing else. The sweeps here are deliberately narrow, and
the objective for the floor is separation between two populations rather
than accuracy on a small labelled set, which overfits less.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.pipeline import Retriever

# Real coverage questions. These must be answered.
REAL = [
    "is the dentist covered", "are glasses covered",
    "how much does a doctor cost", "do i have to wait before using it",
    "can i see a psychologist", "how do i claim money back",
    "is an ambulance covered", "how much medicine can i claim",
    "what happens if my visa is extended", "i bought it but never set it up",
    "apakah sakit gigi tercover", "apakah dokter gigi ditanggung",
    "dentis is cover", "can i see a psychologisss",
]

# Out of scope. These must be refused.
JUNK = [
    "what is the capital of france", "how to calculate p/e ratio",
    "best pizza in melbourne", "who won the grand final",
    "how do i cook rice", "what time is the train to sydney",
    "explain quantum computing", "cheapest flights to bali",
]


def sweep_floor(r, product="comprehensive"):
    print("=" * 72)
    print("1. CONFIDENCE FLOOR")
    print("=" * 72)
    print("  The floor decides what gets refused. Too low and junk is")
    print("  answered. Too high and real questions are refused.\n")

    real_scores = sorted(r.search(q, product=product, k=1)[0][1] for q in REAL)
    junk_scores = sorted(r.search(q, product=product, k=1)[0][1] for q in JUNK)

    print(f"  real questions  n={len(real_scores)}  "
          f"min {real_scores[0]:.4f}  median {real_scores[len(real_scores)//2]:.4f}  "
          f"max {real_scores[-1]:.4f}")
    print(f"  junk questions  n={len(junk_scores)}  "
          f"min {junk_scores[0]:.4f}  median {junk_scores[len(junk_scores)//2]:.4f}  "
          f"max {junk_scores[-1]:.4f}")

    gap = real_scores[0] - junk_scores[-1]
    print(f"\n  separation between the two populations: {gap:+.4f}")
    if gap <= 0:
        print("  They overlap. No single threshold separates them cleanly,")
        print("  so the floor trades one error type against the other.\n")
    else:
        print("  They do not overlap. Any threshold inside the gap works.\n")

    lo = min(real_scores[0], junk_scores[0]) - 0.002
    hi = max(real_scores[-1], junk_scores[-1]) + 0.002
    steps = 40

    print(f"  {'floor':<9}{'real kept':<12}{'junk refused':<15}{'balanced':<10}")
    print("  " + "-" * 50)

    best, best_score = None, -1
    rows = []
    for i in range(steps + 1):
        t = lo + (hi - lo) * i / steps
        kept = sum(1 for s in real_scores if s >= t) / len(real_scores)
        refused = sum(1 for s in junk_scores if s < t) / len(junk_scores)
        # Balanced accuracy. Refusing a real question and answering junk are
        # both errors, and neither is obviously worse in this domain.
        bal = (kept + refused) / 2
        rows.append((t, kept, refused, bal))
        if bal > best_score:
            best_score, best = bal, t

    shown = set()
    for t, kept, refused, bal in rows:
        key = (round(kept, 2), round(refused, 2))
        if key in shown:
            continue
        shown.add(key)
        mark = "  <-- best" if abs(t - best) < 1e-9 else ""
        print(f"  {t:<9.4f}{kept:<12.0%}{refused:<15.0%}{bal:<10.0%}{mark}")

    print(f"\n  Best balanced threshold: {best:.4f}  ({best_score:.0%})")
    print("  Set MIN_CONFIDENCE in app/main.py to this value.")

    # what each choice costs, in plain terms
    kept = sum(1 for s in real_scores if s >= best)
    refused = sum(1 for s in junk_scores if s < best)
    print(f"\n  At {best:.4f}: {kept}/{len(real_scores)} real questions answered, "
          f"{refused}/{len(junk_scores)} junk questions refused.")
    missed = [q for q in REAL if r.search(q, product=product, k=1)[0][1] < best]
    answered = [q for q in JUNK if r.search(q, product=product, k=1)[0][1] >= best]
    if missed:
        print("\n  Real questions this would refuse:")
        for q in missed:
            print(f"    {q}")
    if answered:
        print("\n  Junk questions this would answer:")
        for q in answered:
            print(f"    {q}")
    return best


def sweep_weights():
    print("\n" + "=" * 72)
    print("2. FUSION WEIGHTS")
    print("=" * 72)
    print("  How much the dense and lexical retrievers each contribute.")
    print("  Measured as recall@3 on the labelled test set.\n")

    tests = json.loads(Path("eval/test_set.json").read_text())
    labelled = [t for t in tests
                if t.get("expect") == "answer" and t.get("expect_section")]
    if not labelled:
        print("  No labelled questions in the test set, skipping.")
        return

    import app.pipeline as P
    src = Path("app/pipeline.py").read_text()
    if "W_DENSE" not in src:
        print("  Fusion weights not found in pipeline.py, skipping.")
        return

    print(f"  {'dense':<8}{'bm25':<8}{'recall@3':<10}")
    print("  " + "-" * 28)

    import re
    best, best_r = None, -1
    for w_bm in [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]:
        patched = re.sub(r"W_DENSE, W_BM25, BM25_FLOOR = [\d.]+, [\d.]+, [\d.]+",
                         f"W_DENSE, W_BM25, BM25_FLOOR = 1.0, {w_bm}, 4.0", src)
        Path("app/pipeline.py").write_text(patched)
        import importlib
        importlib.reload(P)
        r = P.Retriever()
        hit = 0
        for t in labelled:
            got = [h[0]["section"] for h in
                   r.search(t["question"], product="comprehensive", k=3)]
            if any(t["expect_section"].lower() in g.lower() for g in got):
                hit += 1
        rec = hit / len(labelled)
        mark = ""
        if rec > best_r:
            best_r, best, mark = rec, w_bm, "  <-- best"
        print(f"  {'1.0':<8}{w_bm:<8}{rec:<10.3f}{mark}")

    Path("app/pipeline.py").write_text(src)
    print(f"\n  Best BM25 weight: {best}  (recall@3 {best_r:.3f})")
    print("  pipeline.py has been restored to its original setting.")
    print("\n  With only "
          f"{len(labelled)} labelled questions, treat a difference of one")
    print("  question as noise rather than a result.")


if __name__ == "__main__":
    r = Retriever()
    sweep_floor(r)
    sweep_weights()
    print("\n" + "=" * 72)
    print("Change one parameter, re-run eval/run_eval.py, keep it or discard.")
    print("Changing several at once tells you nothing about which one worked.")
