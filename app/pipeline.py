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
    "bleeding", "blood", "fever", "rash", "lump", "swollen", "swelling",
    "dizzy", "vomit", "nausea", "chest", "breathe", "breathing", "cough",
    "infection", "infected", "injury", "injured", "broken", "fracture",
    "sick", "ill", "unwell", "symptom", "symptoms", "diagnos", "diagnosed",
    "emergency", "urgent", "ambulance now", "bleeding heavily",
    "depressed", "depression", "anxious", "anxiety", "panic", "suicid",
    "self harm", "harm myself", "cannot sleep", "can't sleep",
    "pregnant", "pregnancy test", "std", "sti", "contracept",
    "medication", "medicine i take", "my prescription", "insulin",
    "asthma", "diabetes", "epilep", "cancer",
]

NURSE_LINE = "1800 887 283"

GATE_RESPONSE = (
    "It sounds like you may need medical advice. Please call the free "
    f"Student Health and Support Line on {NURSE_LINE}. A registered nurse "
    "answers, 24 hours, in around 160 languages. Your message was not saved."
)


def safety_gate(question):
    """Returns (blocked, reason). Runs first, before anything else."""
    q = question.lower()
    for w in CLINICAL:
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
            self._embedder = TextEmbedding("BAAI/bge-small-en-v1.5")
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
