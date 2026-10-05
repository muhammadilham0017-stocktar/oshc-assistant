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
    # Indonesian and Malay. Expansion was English only, so a question
    # in another language scored at the junk baseline even though
    # retrieval found the right section.
    "gigi": "dental dentist optical ancillary extras excluded",
    "dokter": "doctor general practitioner GP medical services MBS",
    "obat": "prescription medicines pharmaceutical",
    "kacamata": "optical glasses items ancillary excluded",
    "biaya": "cost out-of-pocket expense benefit fee",
    "klaim": "claim benefit reimburse receipt",
    "menunggu": "waiting period",
    "jiwa": "mental health psychology psychiatric counselling",
    "mental": "mental health psychology psychiatric counselling",
    # Shorthand students actually type. Not misspellings, so the
    # spelling corrector cannot reach them.
    "dent": "dental dentist general treatment ancillary excluded",
    "dents": "dental dentist general treatment ancillary excluded",
    "tooth": "dental dentist general treatment ancillary",
    "teeth": "dental dentist general treatment ancillary",
    "psych": "psychologist psychology mental health counselling",
    "psycho": "psychologist psychology mental health counselling",
    "counsellor": "counselling psychologist mental health",
    "therapy": "psychologist counselling mental health",
    "meds": "prescription medicines pharmaceutical",
    "med": "prescription medicines pharmaceutical",
    "script": "prescription medicines pharmaceutical",
    "scripts": "prescription medicines pharmaceutical",
    "pills": "prescription medicines pharmaceutical",
    "tablets": "prescription medicines pharmaceutical",
    "pharmacy": "prescription medicines pharmaceutical chemist",
    "chemist": "prescription medicines pharmaceutical pharmacy",
    "amb": "ambulance emergency transport",
    "ambo": "ambulance emergency transport",
    "physio": "physiotherapy ancillary general treatment excluded",
    "specialist": "out-of-hospital medical services MBS referral",
    "scan": "x-ray radiology diagnostic imaging MBS",
    "xray": "x-ray radiology diagnostic imaging MBS",
    "bloods": "pathology blood tests MBS",
    "checkup": "consultation general practitioner GP",
    "check-up": "consultation general practitioner GP",
    "bulk bill": "Direct Billing no out-of-pocket",
    "bulkbill": "Direct Billing no out-of-pocket",
    "bulk billing": "Direct Billing no out-of-pocket",
    "gap": "out-of-pocket expense difference benefit MBS fee",
    "excess": "out-of-pocket expense co-payment",
    "rebate": "benefit claim reimburse",
    "reimburse": "benefit claim",
    "out of pocket": "out-of-pocket expense gap benefit",
    "card": "membership card member number",
    "renew": "policy duration extend cover visa",
    "extend": "policy duration renew cover visa",
    "cancel": "cancelling membership refund premium",
    "switch": "transferring from another health insurer",
    "transfer": "transferring from another health insurer",
    "er": "accident and emergency department hospital",
    "emergency room": "accident and emergency department hospital",
    "a&e": "accident and emergency department hospital",
    "admitted": "inpatient hospital admission",
    "rs": "hospital rumah sakit",
    "resep": "prescription medicines pharmaceutical",
    "periksa": "consultation doctor GP",
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
