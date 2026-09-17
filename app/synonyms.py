"""
Domain synonym expansion at query time.

Students use everyday words. The policy document uses regulatory ones.
This bridges the gap for lexical search. Word boundaries and
deduplication are deliberate: an earlier version used substring matching
and fired twice on charge/charged, doubling term frequency and drowning
out the original question.
"""
import re

SYNONYMS = {
    "dentist": "dental optical ancillary extras excluded",
    "dental": "ancillary extras excluded optical",
    "teeth": "dental ancillary extras excluded",
    "glasses": "optical items ancillary extras excluded",
    "spectacles": "optical items ancillary excluded",
    "physio": "physiotherapy ancillary extras excluded",
    "medicine": "prescription medicines pharmaceutical",
    "medicines": "prescription pharmaceutical",
    "doctor": "general practitioner GP out-of-hospital medical services MBS",
    "gp": "general practitioner out-of-hospital medical services MBS",
    "cost": "out-of-pocket expense benefit MBS fee charge",
    "charge": "out-of-pocket expense benefit MBS fee",
    "charged": "out-of-pocket expense benefit MBS fee",
    "already had": "pre-existing condition",
    "activate": "Online Member Services register arrival membership",
    "set it up": "activate Online Member Services register membership",
    "sign up": "activate Online Member Services register membership",
    "therapist": "psychologist counsellor mental health",
    "counselling": "counsellor psychologist mental health",
    "refund": "claim benefit reimburse",
    "money back": "claim benefit reimburse submit",
    "claim": "benefit reimburse invoice receipt submit",
}


def expand(question):
    low = question.lower()
    seen, extra = set(), []
    for key, val in SYNONYMS.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            for term in val.split():
                if term.lower() not in seen:
                    seen.add(term.lower())
                    extra.append(term)
    return question + " " + " ".join(extra) if extra else question
