"""
Correct typos against the policy vocabulary, before retrieval runs.

A student types "psychologisss". The model would understand that, but the
model never sees it unless retrieval finds a chunk first. So the correction
has to happen at the query.

The vocabulary is built from the indexed documents, so corrections are
always toward words that actually appear in the policy. That is safer than
a general spell checker, which might turn "oshc" into "ouch".
"""
import json
import re
from collections import Counter
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path

# Words a student uses that are not in the policy and must not be corrected.
# Common English. The policy vocabulary does not contain everyday words,
# so without this list "much" gets corrected to "such", "wait" to "with",
# and "glasses" to "assess". Correcting real words is far worse than
# leaving a typo uncorrected.
COMMON = {
    "much", "many", "most", "more", "less", "wait", "waits", "waiting",
    "glasses", "came", "come", "comes", "lately", "late", "later", "left",
    "capital", "right", "wrong", "need", "needs", "want", "wants", "know",
    "think", "feel", "feels", "felt", "tell", "told", "says", "said",
    "take", "takes", "took", "give", "gives", "gave", "make", "makes",
    "made", "find", "finds", "found", "help", "helps", "show", "shows",
    "still", "also", "just", "very", "really", "maybe", "about", "after",
    "before", "because", "since", "until", "while", "where", "when",
    "what", "which", "who", "whom", "whose", "why", "how", "been", "being",
    "have", "has", "had", "does", "did", "done", "will", "would", "could",
    "should", "must", "with", "from", "into", "over", "under", "then",
    "than", "they", "them", "their", "there", "here", "this", "that",
    "these", "those", "some", "any", "each", "every", "other", "another",
    "same", "different", "good", "better", "best", "bad", "worse", "new",
    "old", "long", "short", "high", "low", "big", "small", "next", "last",
    "first", "second", "year", "years", "month", "months", "week", "weeks",
    "day", "days", "time", "times", "today", "tomorrow", "yesterday",
    "night", "morning", "home", "work", "school", "study", "money", "cost",
    "costs", "pay", "paid", "price", "free", "cheap", "sick", "well",
    "please", "sorry", "yes", "no", "not", "and", "but", "for", "the",
    "are", "was", "were", "can", "may", "might", "something", "anything",
    "nothing", "everything", "someone", "anyone", "friend", "friends",
    "family", "student", "students", "australia", "australian",
}

KEEP = {
    "oshc", "medibank", "gp", "mbs", "i", "im", "ive", "dont", "doesnt",
    "cant", "wont", "whats", "hows", "ok", "okay", "hi", "hello", "thanks",
    "sakit", "gigi", "dokter", "obat", "biaya", "klaim", "apakah",
    "saya", "tercover", "ditanggung", "kacamata", "jiwa",
} | COMMON


@lru_cache(maxsize=1)
def _vocab():
    path = Path("data/chunks.json")
    if not path.exists():
        return frozenset(), {}
    words = Counter()
    for c in json.loads(path.read_text()):
        for w in re.findall(r"[a-z]{3,}", c["text"].lower()):
            words[w] += 1
    # Words appearing once are often extraction noise, not real vocabulary.
    vocab = frozenset(w for w, n in words.items() if n >= 2)
    return vocab, dict(words)


def correct(question, min_len=5):
    """Returns the question with unknown words replaced by their nearest
    policy word. Leaves everything else alone."""
    vocab, freq = _vocab()
    if not vocab:
        return question

    out, changed = [], []
    for token in re.findall(r"\S+", question):
        word = re.sub(r"[^a-z]", "", token.lower())
        if (len(word) < min_len or word in vocab or word in KEEP
                or not word.isascii()):
            out.append(token)
            continue
        # Cutoff of 0.75 catches a letter or two wrong, not a different word.
        near = get_close_matches(word, vocab, n=1, cutoff=0.86)
        if near:
            out.append(near[0])
            changed.append((word, near[0]))
        else:
            out.append(token)
    return " ".join(out)


def explain(question):
    """What correct() would change, for debugging and for the demo."""
    vocab, _ = _vocab()
    changes = []
    for token in re.findall(r"\S+", question):
        word = re.sub(r"[^a-z]", "", token.lower())
        if len(word) < 5 or word in vocab or word in KEEP or not word.isascii():
            continue
        near = get_close_matches(word, vocab, n=1, cutoff=0.86)
        if near:
            changes.append((word, near[0]))
    return changes
