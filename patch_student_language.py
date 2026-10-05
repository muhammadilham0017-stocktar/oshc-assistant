"""
Handle the three ways a student's question differs from the policy wording.

    python patch_student_language.py

1. Shorthand. "dent", "psych", "meds" are not misspelled, so spelling
   correction cannot help. They go in the synonym map.

2. Typos. Already handled by app/spelling.py, which corrects against the
   policy vocabulary before retrieval runs.

3. Grammar. Missing articles and auxiliaries, wrong tense, word order.
   This script tests whether they actually break retrieval before adding
   any machinery for them. Embeddings are usually robust to this and BM25
   drops stopwords anyway, so the expectation is that they do not.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SYN = Path("app/synonyms.py")

# Shorthand a student actually types. Each maps to the words the policy
# documents use. These are additions to the existing map, not replacements.
SHORTHAND = {
    # dental
    "dent": "dental dentist general treatment ancillary excluded",
    "dents": "dental dentist general treatment ancillary excluded",
    "tooth": "dental dentist general treatment ancillary",
    "teeth": "dental dentist general treatment ancillary",
    # mental health
    "psych": "psychologist psychology mental health counselling",
    "psycho": "psychologist psychology mental health counselling",
    "counsellor": "counselling psychologist mental health",
    "therapy": "psychologist counselling mental health",
    # medicines
    "meds": "prescription medicines pharmaceutical",
    "med": "prescription medicines pharmaceutical",
    "script": "prescription medicines pharmaceutical",
    "scripts": "prescription medicines pharmaceutical",
    "pills": "prescription medicines pharmaceutical",
    "tablets": "prescription medicines pharmaceutical",
    "pharmacy": "prescription medicines pharmaceutical chemist",
    "chemist": "prescription medicines pharmaceutical pharmacy",
    # services
    "amb": "ambulance emergency transport",
    "ambo": "ambulance emergency transport",
    "physio": "physiotherapy ancillary general treatment excluded",
    "specialist": "out-of-hospital medical services MBS referral",
    "scan": "x-ray radiology diagnostic imaging MBS",
    "xray": "x-ray radiology diagnostic imaging MBS",
    "bloods": "pathology blood tests MBS",
    "checkup": "consultation general practitioner GP",
    "check-up": "consultation general practitioner GP",
    # money
    "bulk bill": "Direct Billing no out-of-pocket",
    "bulkbill": "Direct Billing no out-of-pocket",
    "bulk billing": "Direct Billing no out-of-pocket",
    "gap": "out-of-pocket expense difference benefit MBS fee",
    "excess": "out-of-pocket expense co-payment",
    "rebate": "benefit claim reimburse",
    "reimburse": "benefit claim",
    "out of pocket": "out-of-pocket expense gap benefit",
    # membership
    "card": "membership card member number",
    "renew": "policy duration extend cover visa",
    "extend": "policy duration renew cover visa",
    "cancel": "cancelling membership refund premium",
    "switch": "transferring from another health insurer",
    "transfer": "transferring from another health insurer",
    # hospital
    "er": "accident and emergency department hospital",
    "emergency room": "accident and emergency department hospital",
    "a&e": "accident and emergency department hospital",
    "admitted": "inpatient hospital admission",
    # Indonesian shorthand
    "rs": "hospital rumah sakit",
    "resep": "prescription medicines pharmaceutical",
    "periksa": "consultation doctor GP",
}

# Grammar patterns common in second language English. These are tested,
# not corrected, because retrieval is usually robust to them.
GRAMMAR_PROBES = [
    ("Is a GP visit covered?", "OSHC cover GP visit?"),
    ("Is a GP visit covered?", "GP visit covered or not"),
    ("Is a GP visit covered?", "can i go doctor with oshc"),
    ("How do I claim?", "how to claim"),
    ("How do I claim?", "i want claim money"),
    ("How do I claim?", "claim how"),
    ("Are prescription medicines covered?", "medicine covered?"),
    ("Are prescription medicines covered?", "oshc pay medicine or no"),
    ("Is dental covered?", "dental cover or not"),
    ("Is dental covered?", "my oshc have dental?"),
    ("Is an ambulance covered?", "ambulance free?"),
    ("Do I have to wait?", "waiting time how long"),
]


def patch_synonyms():
    if not SYN.exists():
        print("  app/synonyms.py not found")
        return
    s = SYN.read_text()
    if '"dent":' in s:
        print("  shorthand already present")
        return

    lines = []
    for k, v in SHORTHAND.items():
        lines.append(f'    "{k}": "{v}",')
    block = ("    # Shorthand students actually type. Not misspellings, so the\n"
             "    # spelling corrector cannot reach them.\n" + "\n".join(lines) + "\n}")

    # close the dict on the last entry
    m = re.search(r"\n\}", s)
    if not m:
        print("  could not find the end of SYNONYMS")
        return
    s = s[:m.start()] + "\n" + block + s[m.end() + 2:]
    SYN.write_text(s)
    print(f"  added {len(SHORTHAND)} shorthand terms")


def test_grammar():
    from app.pipeline import Retriever
    r = Retriever()
    FLOOR = 0.0231

    print("\n" + "=" * 74)
    print("GRAMMAR ROBUSTNESS")
    print("=" * 74)
    print("  Does a broken sentence reach the same section as a correct one?\n")

    same, diff, refused = 0, 0, 0
    for good, bad in GRAMMAR_PROBES:
        g = r.search(good, product="comprehensive", k=1)[0]
        b = r.search(bad, product="comprehensive", k=1)[0]
        match = g[0]["section"] == b[0]["section"]
        below = b[1] < FLOOR
        if below:
            refused += 1
            tag = "REFUSED"
        elif match:
            same += 1
            tag = "same   "
        else:
            diff += 1
            tag = "DIFFERS"
        print(f"  {tag}  {b[1]:.4f}  {bad}")
        if not match or below:
            print(f"           correct form -> {g[0]['section']}")
            print(f"           broken form  -> {b[0]['section']}")

    n = len(GRAMMAR_PROBES)
    print(f"\n  same section: {same}/{n}    different: {diff}/{n}    refused: {refused}/{n}")
    if refused + diff > n * 0.3:
        print("\n  Grammar is affecting retrieval. Worth handling.")
    else:
        print("\n  Grammar is mostly not affecting retrieval, which is expected.")
        print("  Embeddings match on meaning, and stopwords are dropped before")
        print("  lexical search. Missing content words break retrieval. Missing")
        print("  articles and auxiliaries usually do not.")


def test_shorthand():
    from app.pipeline import Retriever
    r = Retriever()
    FLOOR = 0.0231
    print("\n" + "=" * 74)
    print("SHORTHAND")
    print("=" * 74)
    probes = ["hello dent cover?", "dent covered", "psych covered",
              "meds covered", "is amb free", "physio covered",
              "do you bulk bill", "whats the gap", "scripts covered"]
    for q in probes:
        top = r.search(q, product="comprehensive", k=1)[0]
        mark = "answer " if top[1] >= FLOOR else "REFUSE "
        print(f"  {mark} {top[1]:.4f}  [{top[0]['section'][:26]:28}] {q}")


if __name__ == "__main__":
    patch_synonyms()
    test_shorthand()
    test_grammar()
    print("\nThen run:  python eval/run_eval.py")
