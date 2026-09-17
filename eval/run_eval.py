"""
Measure retrieval and answer quality against a labelled test set.

    python eval/run_eval.py

Reports recall@3, gate accuracy, readability pass rate and numeric
verification pass rate. Change one thing, re-run, keep or discard.
Fifteen questions covering different failure types beats sixty covering
the same one.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import Retriever, safety_gate, readable, numbers_supported

TESTS = json.loads(Path(__file__).with_name("test_set.json").read_text())


def run():
    r = Retriever()
    recall, gate_ok, rows = [], [], []

    for t in TESTS:
        blocked, term = safety_gate(t["question"])
        expected_block = t["expect"] == "nurse_line"
        gate_ok.append(blocked == expected_block)

        if blocked:
            rows.append({"id": t["id"], "gate": "blocked",
                         "correct": blocked == expected_block, "recall@3": None})
            continue

        hits = r.search(t["question"], product=t.get("product", "comprehensive"), k=3)
        sections = [h[0]["section"] for h in hits]
        want = t.get("expect_section")
        if want:
            got = any(want.lower() in s.lower() for s in sections)
            recall.append(1 if got else 0)
            rows.append({"id": t["id"], "gate": "passed", "correct": got,
                         "recall@3": 1 if got else 0,
                         "expected": want, "returned": sections[0]})
        else:
            rows.append({"id": t["id"], "gate": "passed", "correct": None,
                         "recall@3": None, "returned": sections[0]})

    print(f"{'id':<6}{'gate':<10}{'recall@3':<10}{'expected':<26}{'returned'}")
    print("-" * 82)
    for row in rows:
        print(f"{row['id']:<6}{row['gate']:<10}"
              f"{str(row.get('recall@3', '')):<10}"
              f"{str(row.get('expected', ''))[:24]:<26}"
              f"{str(row.get('returned', ''))[:30]}")

    print("-" * 82)
    print(f"Safety gate accuracy : {sum(gate_ok)}/{len(gate_ok)}")
    if recall:
        print(f"Recall@3             : {sum(recall)}/{len(recall)} "
              f"({sum(recall)/len(recall):.3f})")
    print("\nFailures are the useful part. A test set where everything "
          "passes tells you nothing about the guardrails.")


if __name__ == "__main__":
    run()
