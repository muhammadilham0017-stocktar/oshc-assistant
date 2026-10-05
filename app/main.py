"""
OSHC student assistant.

    streamlit run app/main.py

Three pages. The assistant sits on pages 1 and 2. Page 3 shows aggregate
patterns only and never individual students.
"""

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv()

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from app.pipeline import (Retriever, safety_gate, GATE_RESPONSE, NURSE_LINE,
                          crisis_check, CRISIS_RESPONSE, EMERGENCY_RESPONSE,
                          wellbeing_check, WELLBEING_RESPONSE)
from app.generate import answer as generate_answer

DB = Path("data/interactions.db")
APPROVED = Path("data/approved_answers.json")

st.set_page_config(page_title="Your OSHC", layout="centered")


# ------------------------------------------------------------------ logging
def init_db():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS interactions (
        ts TEXT, topic TEXT, confidence REAL, page TEXT,
        source TEXT, routed_to_nurse INTEGER)""")
    con.commit()
    return con



def md_safe(text):
    """Streamlit renders text between two dollar signs as LaTeX, which
    turns '$100 ... $200' into a formula. Escaping them keeps benefit
    amounts readable."""
    return text.replace("$", "\\$")

def log(topic, confidence, page, source, routed=0):
    """Matched topic and confidence only. Never the question text, never a
    member identity. Supports Australian Privacy Principle 2."""
    con = init_db()
    con.execute("INSERT INTO interactions VALUES (?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), topic,
                 confidence, page, source, routed))
    con.commit()
    con.close()


# ------------------------------------------------------------------ helpers
@st.cache_resource
def load_retriever():
    return Retriever()


@st.cache_data
def load_approved():
    return json.loads(APPROVED.read_text()) if APPROVED.exists() else {}


def ask(question, product, page_name):
    """The runtime path. Gate first, always."""
    # Tier 1 runs before the wellbeing and clinical checks.
    tier = crisis_check(question)
    if tier == "crisis":
        log("crisis", 1.0, page_name, "lifeline", routed=1)
        st.error(md_safe(CRISIS_RESPONSE))
        return
    if tier == "emergency":
        log("emergency", 1.0, page_name, "triple_zero", routed=1)
        st.error(md_safe(EMERGENCY_RESPONSE))
        return

    if wellbeing_check(question):
        log("wellbeing", 1.0, page_name, "support_line", routed=1)
        st.info(md_safe(WELLBEING_RESPONSE))
        return

    blocked, term = safety_gate(question)
    if blocked:
        log("clinical", 1.0, page_name, "nurse_line", routed=1)
        st.error(md_safe(GATE_RESPONSE))
        return

    r = load_retriever()
    hits = r.search(question, product=product, k=3)
    if not hits:
        st.warning("I do not have an answer for that. "
                   f"Please call {NURSE_LINE}.")
        log("unmatched", 0.0, page_name, "fallback")
        return

    passages = [h[0] for h in hits]
    confidence = float(hits[0][1])

    # Fusion scores below this mean nothing in the corpus matched.
    # Without a floor the system answers confidently from whatever
    # ranked first, which is how an accounting question returns a
    # dental answer.
    MIN_CONFIDENCE = 0.018
    if confidence < MIN_CONFIDENCE:
        st.warning("I do not have an answer for that. "
                   f"Please call {NURSE_LINE}.")
        log("unmatched", confidence, page_name, "below_threshold")
        return
    topic = passages[0]["section"]
    # Fallback answers must be product specific. Keying on topic
    # alone would give an Essentials member the Comprehensive answer
    # whenever generation fails.
    approved = load_approved().get(product, {}).get(topic)

    member = {"product": product.title(),
              "campus": st.session_state.get("campus"),
              "weeks": st.session_state.get("weeks")}

    text, trace, source = generate_answer(question, passages, member,
                                          fallback=approved)
    log(topic, confidence, page_name, source)

    st.success(md_safe(text))
    with st.expander("Where this came from"):
        for p in passages:
            st.caption(f"{p['section']}, page {p['page']} "
                       f"({p['product']}, effective {p['effective']})")
    if trace:
        with st.expander("Checks"):
            st.json(trace)


def assistant(page_name):
    st.markdown("#### Ask about your cover")
    st.caption("Please do not type personal health details here.")
    q = st.text_input("Question", key=f"q_{page_name}",
                      placeholder="Is the dentist covered?",
                      label_visibility="collapsed")
    if st.button("Ask", key=f"b_{page_name}") and q:
        ask(q, st.session_state.get("product", "comprehensive"), page_name)


# -------------------------------------------------------------------- pages
def page_module():
    st.title("How healthcare works here")
    st.write("Five things worth knowing before you need them.")
    for title, body in [
        ("See a GP first",
         "In Australia your regular doctor is called a GP, short for general "
         "practitioner. You see a GP first for anything that is not an "
         "emergency."),
        ("Specialists need a referral",
         "You cannot book a specialist directly. The GP writes you a referral."),
        ("Some clinics cost you nothing",
         "Direct Billing clinics bill us instead of you. Find one before you "
         "need it."),
        ("Emergency departments are for emergencies",
         "For anything not urgent, a GP is faster and usually cheaper."),
        ("Some things are not covered",
         "Dental check-ups, glasses and physiotherapy are not included."),
    ]:
        with st.expander(title):
            st.write(body)
    st.divider()
    assistant("module")


def page_dashboard():
    st.title("Your cover")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cover ends", "Feb 2028")
    c2.metric("Medicine left", "$412")
    c3.metric("Nearest no-cost doctor", "1.2 km")
    st.divider()
    assistant("dashboard")


def page_internal():
    st.title("Internal view")
    st.caption("Aggregate patterns only. No individual students.")
    con = init_db()
    rows = con.execute("""SELECT topic, COUNT(*) n, AVG(confidence) c
                          FROM interactions GROUP BY topic
                          ORDER BY n DESC""").fetchall()
    con.close()
    if not rows:
        st.info("No interactions logged yet.")
        return
    st.subheader("Most asked topics")
    st.dataframe([{"Topic": t, "Questions": n, "Avg confidence": round(c or 0, 2)}
                  for t, n, c in rows], use_container_width=True)
    st.caption("Topics with low average confidence are where our answers "
               "are weakest and should be rewritten first.")


# --------------------------------------------------------------------- main
with st.sidebar:
    st.subheader("Demo settings")
    st.session_state["product"] = st.selectbox(
        "Cover", ["comprehensive", "essentials"])
    st.session_state["campus"] = st.text_input("Campus", "Swinburne")
    st.session_state["weeks"] = st.number_input("Weeks in Australia", 0, 200, 3)
    page = st.radio("Page", ["1. Module", "2. Dashboard", "3. Internal"])

{"1. Module": page_module,
 "2. Dashboard": page_dashboard,
 "3. Internal": page_internal}[page]()
