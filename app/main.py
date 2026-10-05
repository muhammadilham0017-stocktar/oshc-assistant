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

STYLE = """
<style>
:root{
  --ink:#0E2033; --ink-2:#4A6076; --line:#CBD6E0;
  --paper:#FFFFFF; --floor:#EDF1F5;
  --blue:#0B5FA5; --blue-soft:#E2EDF7;
  --red:#C2151B; --red-soft:#FBE9E9;
  --green:#0F7A5A; --green-soft:#E3F2EC;
  --amber:#9A6207; --amber-soft:#FCF2DF;
}
.stApp{background:var(--floor)}
.block-container{max-width:820px;padding-top:2rem;background:var(--paper);
  border-left:1px solid var(--line);border-right:1px solid var(--line)}
h1{font-size:1.65rem!important;letter-spacing:-.015em;color:var(--ink)}
h2{font-size:1.2rem!important;color:var(--ink)}
h4{font-size:1.02rem!important;color:var(--ink);margin-bottom:.2rem}

/* answer, gate and wellbeing blocks */
div[data-testid="stAlert"]{border-radius:0;border-left-width:4px;padding:1rem 1.15rem}
div[data-testid="stAlertContentSuccess"]{font-size:1.02rem;line-height:1.6}

/* the source panel */
.srcbox{background:var(--paper);border:1px solid var(--line);
  padding:.8rem .95rem;margin-bottom:.55rem}
.srcwhere{font-size:.76rem;color:var(--ink-2);margin:0 0 .45rem;
  display:flex;flex-wrap:wrap;gap:.25rem .7rem;align-items:baseline}
.srcwhere b{color:var(--ink)}
.srcwords{margin:0;font-size:.87rem;line-height:1.5;color:var(--ink-2)}
.srcgrade{font-size:.75rem;margin:.5rem 0 0;padding-top:.45rem;
  border-top:1px dashed var(--line);color:var(--amber);font-weight:600}
.srcgrade.easy{color:var(--green)}

/* readability comparison */
.cmp{display:flex;border:1px solid var(--line);margin-bottom:.7rem;overflow:hidden}
.cmp div{flex:1;padding:.6rem .8rem}
.cmp .lab{font-size:.66rem;letter-spacing:.1em;font-weight:700;
  margin:0 0 .15rem;color:var(--ink-2)}
.cmp .val{font-size:1.2rem;font-weight:700;margin:0;font-variant-numeric:tabular-nums}
.cmp .cap{font-size:.72rem;color:var(--ink-2);margin:.1rem 0 0}
.cmp .ours{background:var(--green-soft)} .cmp .ours .val{color:var(--green)}
.cmp .theirs{background:var(--amber-soft);border-left:1px solid var(--line)}
.cmp .theirs .val{color:var(--amber)}

.privnote{font-size:.74rem;color:var(--ink-2);margin:.6rem 0 0;line-height:1.5}

/* metrics on the dashboard */
div[data-testid="stMetric"]{background:var(--floor);padding:.75rem .9rem;
  border:1px solid var(--line)}
div[data-testid="stMetricValue"]{font-size:1.45rem;color:var(--ink)}
div[data-testid="stMetricLabel"]{font-size:.78rem;color:var(--ink-2)}

/* buttons */
.stButton > button{border-radius:0;border:1.5px solid var(--line);
  font-weight:600;color:var(--ink);background:var(--paper)}
.stButton > button:hover{border-color:var(--blue);color:var(--blue)}

/* input */
.stTextInput input{border-radius:0;border:1.5px solid var(--line)}
.stTextInput input:focus{border-color:var(--blue);box-shadow:none}

/* expanders */
details{border:1px solid var(--line)!important;border-radius:0!important}
summary{font-size:.86rem!important;font-weight:600;color:var(--blue)!important}

#MainMenu,footer{visibility:hidden}
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)


def md_safe(text):
    """Streamlit treats text between two dollar signs as LaTeX, which turns
    "$100 ... $200" into a formula. Escaping keeps benefit amounts readable."""
    return str(text).replace("$", "\\$")


def _grade(text):
    try:
        import textstat
        return round(textstat.flesch_kincaid_grade(text), 1)
    except Exception:
        return None


def _reads_as(g):
    if g is None: return ""
    if g >= 18: return "Postgraduate level"
    if g >= 13: return "University level"
    if g >= 10: return "Senior school level"
    return "About a 10 year old could read it"


def show_sources(answer_text, passages):
    """Shows the real policy wording behind the answer, and how much harder
    it is to read than the plain version. A page number tells a student
    nothing. The actual sentence lets them check."""
    import html
    ours = _grade(answer_text)
    grades = [g for g in (_grade(p["text"]) for p in passages) if g is not None]
    worst = max(grades) if grades else None

    with st.expander("What this is based on"):
        if ours is not None and worst is not None:
            st.markdown(
                f'<div class="cmp">'
                f'<div class="ours"><p class="lab">THIS ANSWER</p>'
                f'<p class="val">Grade {ours}</p>'
                f'<p class="cap">{_reads_as(ours)}</p></div>'
                f'<div class="theirs"><p class="lab">THE DOCUMENT</p>'
                f'<p class="val">Grade {worst}</p>'
                f'<p class="cap">{_reads_as(worst)}</p></div></div>',
                unsafe_allow_html=True)

        for p in passages:
            g = _grade(p["text"])
            words = html.escape(p["text"][:420])
            if len(p["text"]) > 420:
                words += "..."
            cls = " easy" if (g is not None and g <= 9) else ""
            tail = ("readable" if (g is not None and g <= 9)
                    else "harder than most people can read comfortably")
            st.markdown(
                f'<div class="srcbox">'
                f'<p class="srcwhere"><b>{html.escape(p["section"])}</b>'
                f'<span>page {p["page"]}</span>'
                f'<span>{html.escape(p["product"])}</span>'
                f'<span>effective {html.escape(str(p["effective"]))}</span></p>'
                f'<p class="srcwords">&ldquo;{words}&rdquo;</p>'
                + (f'<p class="srcgrade{cls}">Reading grade {g} &middot; {tail}</p>'
                   if g is not None else "")
                + '</div>', unsafe_allow_html=True)

        st.markdown(
            '<p class="privnote">We saved the topic of this question. '
            'We did not save what you typed, and it is not linked to you.</p>',
            unsafe_allow_html=True)




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
    MIN_CONFIDENCE = 0.0231  # from tune.py, best balanced accuracy 89%
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
    show_sources(text, passages)
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
