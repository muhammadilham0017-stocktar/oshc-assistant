"""
Adds crisis and emergency routing to app/pipeline.py and app/main.py.

Run from the project root:

    python patch_crisis.py

Testing the frontend found that "i want to kill myself" received a generic
"I do not have an approved answer for that" response. The clinical list
contained the word suicide but not the sentence a person in crisis types.
So did "i cant breathe", "my chest is tight", "i took too many pills" and
the equivalents in other languages.

These two tiers run before everything else, including before the coverage
intent override. A question that mentions cover and also discloses a crisis
is still a crisis.
"""
import re
import sys
from pathlib import Path

PIPELINE = Path("app/pipeline.py")
MAIN = Path("app/main.py")

CRISIS_BLOCK = '''
# ------------------------------------------------------------------ TIER 1
# Immediate danger. Checked before anything else and before the coverage
# override. Nothing is retrieved, generated, stored or logged as text.
CRISIS = [
    # self harm, English
    "kill myself", "killing myself", "end my life", "ending my life",
    "want to die", "wanna die", "dont want to live", "don't want to live",
    "better off dead", "take my own life", "suicide", "suicidal",
    "self harm", "hurt myself", "harm myself", "cut myself",
    "overdose", "took too many", "no reason to live",
    "cant go on", "can't go on",
    # self harm, other languages
    "ingin mati", "mau mati", "bunuh diri", "akhiri hidup",
    "\\u6211\\u60f3\\u6b7b", "\\u81ea\\u6740", "\\u4e0d\\u60f3\\u6d3b",
    "mu\\u1ed1n ch\\u1ebft", "t\\u1ef1 t\\u1eed",
]

EMERGENCY = [
    "cant breathe", "can't breathe", "cannot breathe", "trouble breathing",
    "not breathing", "chest pain", "chest is tight", "chest tight",
    "heart attack", "stroke", "unconscious", "passed out",
    "wont wake up", "won't wake up", "seizure", "choking",
    "severe bleeding", "bleeding a lot", "anaphyla", "allergic reaction",
    "call an ambulance",
    "sesak napas", "sesak nafas", "nyeri dada", "serangan jantung", "pingsan",
    "\\u547c\\u5438\\u56f0\\u96be", "\\u80f8\\u75db", "\\u660f\\u8ff7",
    "kh\\u00f3 th\\u1edf", "\\u0111au ng\\u1ef1c",
]

LIFELINE = "13 11 14"
TRIPLE_ZERO = "000"

CRISIS_RESPONSE = (
    "I am not the right help for this, and I want you to talk to someone "
    f"who is. Lifeline is free, 24 hours, on {LIFELINE}. If you are in "
    f"immediate danger call {TRIPLE_ZERO}. Your cover also includes free "
    "counselling on 1800 887 283, in around 160 languages. "
    "Nothing you typed was saved."
)

EMERGENCY_RESPONSE = (
    f"This sounds like an emergency. Call {TRIPLE_ZERO} now. Do not drive "
    "yourself. Emergency ambulance is fully covered, with no limit and no "
    "waiting period. Nothing you typed was saved."
)


def _match(terms, q):
    """Word boundaries for Latin scripts, substring for scripts without
    spaces. Same rule as the clinical gate."""
    for w in terms:
        if not w.isascii():
            if w in q:
                return w
        elif re.search(rf"\\b{re.escape(w)}\\b", q):
            return w
    return None


def crisis_check(question):
    """Returns 'crisis', 'emergency' or None. Runs first, always."""
    q = question.lower()
    if _match(CRISIS, q):
        return "crisis"
    if _match(EMERGENCY, q):
        return "emergency"
    return None

'''

WELLBEING_EXTRA = '''    "panic attack", "anxiety attack", "cant cope", "can't cope",
    "breaking down", "was assaulted", "been assaulted", "attacked me",
    "abused", "unsafe at home",
    "serangan panik", "cemas", "diserang",
'''

MAIN_ROUTE = '''    # Tier 1 runs before the wellbeing and clinical checks.
    tier = crisis_check(question)
    if tier == "crisis":
        log("crisis", 1.0, page_name, "lifeline", routed=1)
        st.error(CRISIS_RESPONSE)
        return
    if tier == "emergency":
        log("emergency", 1.0, page_name, "triple_zero", routed=1)
        st.error(EMERGENCY_RESPONSE)
        return

'''


def patch_pipeline():
    s = PIPELINE.read_text()
    if "CRISIS_RESPONSE" in s:
        print("  pipeline.py already patched")
        return
    anchor = "NURSE_LINE ="
    if anchor not in s:
        sys.exit("  could not find NURSE_LINE in pipeline.py")
    s = s.replace(anchor, CRISIS_BLOCK + "\n" + anchor, 1)

    # extend the wellbeing list if it exists
    if "WELLBEING = [" in s and "panic attack" not in s:
        s = re.sub(r'(WELLBEING = \[\n)', r'\1' + WELLBEING_EXTRA, s, count=1)
        print("  wellbeing list extended")

    PIPELINE.write_text(s)
    print("  pipeline.py patched")


def patch_main():
    s = MAIN.read_text()
    if "crisis_check" in s:
        print("  main.py already patched")
        return

    # import
    s = re.sub(
        r"(from app\.pipeline import \()",
        r"\1crisis_check, CRISIS_RESPONSE, EMERGENCY_RESPONSE,\n                          ",
        s, count=1)
    if "crisis_check" not in s:
        s = s.replace(
            "from app.pipeline import Retriever, safety_gate, GATE_RESPONSE, NURSE_LINE",
            "from app.pipeline import (Retriever, safety_gate, GATE_RESPONSE, NURSE_LINE,\n"
            "                          crisis_check, CRISIS_RESPONSE, EMERGENCY_RESPONSE)")

    # route, before every other check
    hooks = [
        '    """The runtime path. Gate first, always."""\n',
        '    blocked, term = safety_gate(question)\n',
    ]
    for h in hooks:
        if h in s:
            s = s.replace(h, h + MAIN_ROUTE if h.endswith('"""\n') else MAIN_ROUTE + h, 1)
            break
    else:
        sys.exit("  could not find the ask() entry point in main.py")

    MAIN.write_text(s)
    print("  main.py patched")


if __name__ == "__main__":
    if not PIPELINE.exists():
        sys.exit("Run this from the project root, where app/ lives.")
    patch_pipeline()
    patch_main()
    print("\nNow verify:\n  python - <<'EOF'\n"
          "import sys; sys.path.insert(0,'.')\n"
          "from app.pipeline import crisis_check\n"
          "for q in ['i want to kill myself','i cant breathe','saya ingin mati',\n"
          "          'is the dentist covered','my tooth hurts']:\n"
          "    print(f'  {str(crisis_check(q)):<10} {q}')\n"
          "EOF")
