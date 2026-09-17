"""
Answer generation.

Nothing here invents content. The model rewrites retrieved passages in
plain English and is checked twice before the student sees anything.

Set GROQ_API_KEY to enable generation. Without it the app falls back to
the human approved answer for the matched topic, which is also what
happens if generation fails on the day.
"""
import os
from dotenv import load_dotenv
load_dotenv()
import re

from app.pipeline import numbers_supported, readable, NURSE_LINE

TARGET_GRADE = 9
MAX_ATTEMPTS = 3

STYLE_EXAMPLES = """
Q: is the dentist covered
A: No, regular dental check-ups are not covered. You pay for those
   yourself. Dental surgery in hospital is different and part of that
   is covered.

Q: is an ambulance covered
A: Yes. Emergency ambulance is fully covered anywhere in Australia.
   There is no limit and no waiting period.
"""

RULES = """Write at a Year 7 reading level.
Maximum 15 words per sentence.
Answer in the first sentence, then explain.
Use "you" and "we". Active voice.
Define any policy term you have to use.
End with one thing the student can do next.
Use only the passages below. Nothing else.
Never state a number that is not written in the passages.
Do not use emoji. Be warm and direct, not casual."""


def build_prompt(question, passages, member, stricter=False):
    ctx = "\n\n".join(
        f"[{i+1}] {p['section']}, page {p['page']}: {p['text']}"
        for i, p in enumerate(passages))
    harder = ("\nThe previous attempt was too hard to read. "
              "Use shorter sentences and simpler words.\n") if stricter else ""
    mem = (f"Member: {member.get('product', 'Comprehensive')} cover"
           + (f", studies at {member['campus']}" if member.get("campus") else "")
           + (f", {member['weeks']} weeks in Australia" if member.get("weeks") else "")
           + ".")
    return f"{RULES}{harder}\n\nWrite in this style:{STYLE_EXAMPLES}\n{mem}\n\nPassages:\n{ctx}\n\nQuestion: {question}\n"


def call_model(prompt):
    """Groq. Swap the base_url and model for any OpenAI-compatible service,
    including an Australian region deployment for production."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                    base_url="https://api.groq.com/openai/v1")
    resp = client.chat.completions.create(
        model=os.getenv("MODEL", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1200,
    )
    msg = resp.choices[0].message
    text = (msg.content or "").strip()
    if not text:
        # Reasoning models can spend the whole budget on internal
        # reasoning and return empty content. Treat that as a failed
        # attempt rather than a valid empty answer.
        raise RuntimeError(
            f"empty content, finish_reason={resp.choices[0].finish_reason}")
    return text


def answer(question, passages, member, fallback=None):
    """Returns (text, trace, source). source is 'generated' or 'fallback'."""
    trace = []

    if not os.getenv("GROQ_API_KEY"):
        return (fallback or _no_answer()), trace, "fallback"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        prompt = build_prompt(question, passages, member, stricter=attempt > 1)
        try:
            text = call_model(prompt)
        except Exception as e:
            trace.append({"attempt": attempt, "error": str(e)[:120]})
            break

        ok_num, bad = numbers_supported(text, passages)
        ok_read, grade = readable(text, TARGET_GRADE)
        trace.append({"attempt": attempt, "grade": grade,
                      "unsupported_numbers": bad, "text": text})

        if ok_num and ok_read:
            return text, trace, "generated"

    return (fallback or _no_answer()), trace, "fallback"


def _no_answer():
    return ("I do not have an approved answer for that. Please call the free "
            f"Student Health and Support Line on {NURSE_LINE}.")
