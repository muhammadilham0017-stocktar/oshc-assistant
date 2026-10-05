"""
The parts that run when a student asks a question.

Order matters. The safety gate runs before anything is retrieved,
generated or logged, so health information is never collected.
"""
import json
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------- 1. GATE
# Clinical terms. Deliberately a plain list so a reviewer can see exactly
# what triggers it. High recall is the goal: routing a policy question to
# a nurse costs one phone call, missing a crisis message costs far more.
CLINICAL = [
    "hurt", "hurts", "hurting", "pain", "painful", "ache", "aching", "sore",
    "bleeding", "blood in", "fever", "rash", "lump", "swollen", "swelling",
    "dizzy", "vomit", "nausea", "chest", "breathe", "breathing", "cough",
    "infection", "infected", "injury", "injured", "broken", "fracture",
    "sick", "ill", "unwell", "symptom", "symptoms", "diagnos", "diagnosed",
    "emergency", "urgent", "ambulance now", "bleeding heavily",
    "depressed", "depression", "anxious", "anxiety", "panic", "suicid",
    "self harm", "harm myself", "cannot sleep", "can't sleep",
    "pregnant", "pregnancy test", "std", "sti", "contracept",
    "medication", "medicine i take", "my prescription", "insulin",
    "asthma", "diabetes", "epilep", "cancer",
    "toothache", "headache", "backache", "stomachache", "earache",
    "sore throat", "食欲", "不舒服",
    # Indonesian and Malay
    "sakit", "nyeri", "demam", "batuk", "muntah", "luka", "gigi sakit",
    "pusing", "sesak", "hamil", "depresi", "cemas",
    # Chinese
    "\u75bc", "\u75db", "\u53d1\u70e7", "\u54b3\u55fd", "\u5417\u5410",
    "\u53d7\u4f24", "\u6000\u5b55", "\u6291\u90c1", "\u7126\u8651",
    # Vietnamese
    "\u0111au", "s\u1ed1t", "ho", "n\u00f4n", "ch\u1ea5n th\u01b0\u01a1ng",
    # Hindi and Nepali romanised
    "dard", "bukhar", "khansi",
]

# Words that mark a question about the policy rather than a report
# about the person. "my tooth hurts" is a symptom. "is toothache
# covered" is a coverage question that happens to name a symptom.
COVERAGE_INTENT = [
    "cover", "covered", "coverage", "claim", "benefit", "pay", "cost",
    "price", "limit", "included", "exclude", "excluded", "policy",
    "tercover", "ditanggung", "biaya", "klaim",
    "\u4fdd\u9669", "\u62a5\u9500", "\u8d39\u7528",
    "b\u1ea3o hi\u1ec3m", "chi ph\u00ed",
]


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
    "\u6211\u60f3\u6b7b", "\u81ea\u6740", "\u4e0d\u60f3\u6d3b",
    "mu\u1ed1n ch\u1ebft", "t\u1ef1 t\u1eed",
]

EMERGENCY = [
    "cant breathe", "can't breathe", "cannot breathe", "trouble breathing",
    "not breathing", "chest pain", "chest is tight", "chest tight",
    "heart attack", "stroke", "unconscious", "passed out",
    "wont wake up", "won't wake up", "seizure", "choking",
    "severe bleeding", "bleeding a lot", "anaphyla", "allergic reaction",
    "call an ambulance",
    "sesak napas", "sesak nafas", "nyeri dada", "serangan jantung", "pingsan",
    "\u547c\u5438\u56f0\u96be", "\u80f8\u75db", "\u660f\u8ff7",
    "kh\u00f3 th\u1edf", "\u0111au ng\u1ef1c",
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
        elif re.search(rf"\b{re.escape(w)}\b", q):
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


NURSE_LINE = "1800 887 283"

GATE_RESPONSE = (
    "It sounds like you may need medical advice. Please call the free "
    f"Student Health and Support Line on {NURSE_LINE}. A registered nurse "
    "answers, 24 hours, in around 160 languages. Your message was not saved."
)



# Wellbeing is not a clinical symptom and needs a warmer response.
# Loneliness and homesickness are common among international students,
# and a generic refusal is the wrong answer to them.
WELLBEING = [
    "lonely", "alone", "isolated", "homesick", "miss home", "miss my family",
    "no friends", "sad", "stressed", "stress", "overwhelmed", "struggling",
    "been stressed", "feeling stressed", "cant cope", "can't cope",
    "panic attack", "anxiety attack", "breaking down", "hopeless",
    "was assaulted", "been assaulted", "abused", "unsafe at home",
    "kesepian", "sendiri", "rindu rumah", "sedih", "stres", "cemas",
    "\u5b64\u72ec", "\u60f3\u5bb6", "\u96be\u8fc7", "\u538b\u529b",
]

WELLBEING_RESPONSE = (
    "That sounds hard, and a lot of students feel this way, especially "
    "early on. You can talk to a qualified counsellor any time, free, on "
    "1800 887 283. It is open 24 hours and in around 160 languages. "
    "Your message was not saved."
)


def wellbeing_check(question):
    """Runs after the crisis tier and before the clinical gate.

    A coverage question that happens to name a feeling is not a wellbeing
    disclosure. "OSHC help with stress" asks what the policy pays for.
    "I have been stressed since arriving" is the student telling us
    something. First person phrasing decides which it is."""
    q = question.lower()
    first = bool(re.search(r"\b(i |i'm|im |my |me\b)", q)) or "saya " in q or "\u6211" in q
    if not first and any(c in q for c in
            ["cover", "covered", "claim", "benefit", "pay", "include",
             "oshc", "policy", "limit", "tercover", "ditanggung"]):
        return None
    for w in WELLBEING:
        if not w.isascii():
            if w in q:
                return w
        elif re.search(rf"\b{re.escape(w)}\b", q):
            return w
    return None


def safety_gate(question):
    """Returns (blocked, reason). Runs first, before anything else."""
    q = question.lower()

    # A coverage question naming a symptom is not a symptom report.
    # Deliberately narrow: first person phrasing still routes to a
    # human even when coverage words are present.
    first_person = any(p in q for p in
                       ["i have", "i feel", "my ", "saya ", "i am", "im ",
                        "\u6211", "t\u00f4i"])
    # First person disclosure always wins. A student who says they
    # have a symptom has disclosed health information, whether or not
    # they also asked about cover.
    if first_person:
        for w in CLINICAL:
            if (w in q) if not w.isascii() else re.search(rf"\b{re.escape(w)}\b", q):
                return True, w
    if any(c in q for c in COVERAGE_INTENT):
        return False, None

    for w in CLINICAL:
        # Word boundaries do not apply to scripts without spaces,
        # so CJK terms are matched as substrings.
        if not w.isascii():
            if w in q:
                return True, w
            continue
        if re.search(rf"\b{re.escape(w)}\b", q):
            return True, w
    return False, None


# ----------------------------------------------------------- 2. RETRIEVAL
class Retriever:
    """Hybrid. Dense catches meaning, so 'dentist' finds 'general
    treatment'. BM25 catches exact strings like $50 and MBS, which
    embeddings blur. Both are needed."""

    def __init__(self, chunks_path="data/chunks.json", vectors_path="data/vectors.npy"):
        self.chunks = json.loads(Path(chunks_path).read_text())
        self.texts = [c["embed_text"] for c in self.chunks]
        self.bm25 = BM25Okapi([self._tok(t) for t in self.texts])
        # Character n-grams tolerate misspelling, which matters because
        # the target users are writing in a second language. Whole-word
        # matching fails on 'psychologisss'; letter sequences do not.
        self.char_vec = TfidfVectorizer(analyzer='char_wb',
                                        ngram_range=(3, 5), sublinear_tf=True)
        self.char_M = self.char_vec.fit_transform(self.texts)

        vp = Path(vectors_path)
        self.vectors = np.load(vp) if vp.exists() else None
        # Table rows are short fragments. Dense retrieval scores them
        # highly against almost any health question, crowding out the
        # prose that actually answers it. BM25 still sees them, because
        # exact term overlap is where they are genuinely useful.
        self.dense_ok = np.array(
            [c["section"] != "Benefits table" for c in self.chunks])
        self._embedder = None

    # Without this, words like "do", "i", "have", "to" match almost
    # every chunk and BM25 returns noise that displaces correct dense
    # hits during fusion.
    STOP = {
        "a","an","and","are","as","at","be","been","before","but","by",
        "can","could","do","does","did","for","from","get","got","had",
        "has","have","how","i","if","in","into","is","it","its","me",
        "my","of","on","or","that","the","their","them","then","there",
        "they","this","to","use","using","want","was","we","what","when",
        "where","which","who","will","with","would","you","your","am",
        "much","many","any","some","need","still","back","up","out",
    }

    @classmethod
    def _tok(cls, s):
        toks = re.findall(r"[a-z0-9$%.]+", s.lower())
        return [t for t in toks if t not in cls.STOP and len(t) > 1]

    def _embed_query(self, q):
        if self._embedder is None:
            from fastembed import TextEmbedding
            self._embedder = TextEmbedding("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return np.array(list(self._embedder.embed([q]))[0])

    def search(self, question, product, k=3, pool=20, expand_query=True):
        """Product filter runs BEFORE search, not after. Without it an
        Essentials member can receive Comprehensive limits."""
        if expand_query:
            from app.synonyms import expand
            question = expand(question)
        mask = np.array([c["product"] == product for c in self.chunks])
        if not mask.any():
            mask = np.ones(len(self.chunks), dtype=bool)

        bm = np.array(self.bm25.get_scores(self._tok(question)))
        bm[~mask] = -1e9
        bm_rank = (-bm).argsort()

        if self.vectors is not None:
            qv = self._embed_query(question)
            dense = self.vectors @ qv
            dense[~mask] = -1e9
            dense[~self.dense_ok] = -1e9
            d_rank = (-dense).argsort()
        else:
            d_rank = bm_rank  # dense unavailable, fall back to lexical only

        # Weighted reciprocal rank fusion.
        # Dense is weighted higher because on this corpus it is far
        # more reliable. BM25 contributes only where its score is
        # meaningful: below the cutoff it is matching stopwords and
        # its results actively displace correct dense hits.
        W_DENSE, W_BM25, BM25_FLOOR = 1.0, 0.3, 4.0
        fused = {}
        for rank, idx in enumerate(d_rank[:pool]):
            fused[idx] = fused.get(idx, 0) + W_DENSE / (60 + rank)
        for rank, idx in enumerate(bm_rank[:pool]):
            if bm[idx] < BM25_FLOOR:
                continue
            fused[idx] = fused.get(idx, 0) + W_BM25 / (60 + rank)

        char = cosine_similarity(
            self.char_vec.transform([question]), self.char_M)[0]
        char[~mask] = -1e9
        c_rank = (-char).argsort()
        for rank, idx in enumerate(c_rank[:pool]):
            if char[idx] < 0.10:
                continue
            fused[idx] = fused.get(idx, 0) + 0.3 / (60 + rank)

        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        return [(self.chunks[i], score) for i, score in top]


# ----------------------------------------------------------- 3. VALIDATION
NUM = re.compile(r"\$?\d[\d,]*\.?\d*\s?%?")


def numbers_supported(answer, passages):
    """Every figure in the answer must appear in a retrieved passage.
    A fabricated benefit limit is the failure with the worst consequence,
    so this is checked deterministically rather than left to the prompt."""
    source = " ".join(p["text"] for p in passages)
    src_nums = {n.strip().strip("$%").replace(",", "")
                for n in NUM.findall(source)}
    bad = []
    for raw in NUM.findall(answer):
        val = raw.strip().strip("$%").replace(",", "")
        if val and val not in src_nums:
            bad.append(raw.strip())
    return (len(bad) == 0), bad


def readable(answer, target=9):
    """Year 7 is the Style Manual recommendation. We gate at 9 to leave
    room for unavoidable policy terms."""
    import textstat
    grade = textstat.flesch_kincaid_grade(answer)
    return (grade <= target), round(grade, 1)
