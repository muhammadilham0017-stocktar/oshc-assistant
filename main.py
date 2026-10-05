"""
OSHC student assistant.

    streamlit run app/main.py

A chat interface over Medibank's published policy documents. Three pages.
The assistant sits on pages 1 and 2. Page 3 shows aggregate patterns only
and never individual students.

Routing order, which matters:
  1. Crisis and emergency, before anything is retrieved or stored
  2. Wellbeing
  3. Clinical gate
  4. Product filter, then retrieval
  5. Confidence floor
  6. Generation, constrained to retrieved passages
  7. Numeric and readability checks
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import html
import json
import sqlite3
from datetime import datetime, timezone

import streamlit as st

from app.pipeline import Retriever, safety_gate, GATE_RESPONSE, NURSE_LINE
from app.generate import answer as generate_answer

# These exist only after patch_crisis.py has run.
try:
    from app.pipeline import crisis_check, CRISIS_RESPONSE, EMERGENCY_RESPONSE
except ImportError:
    crisis_check = lambda q: None
    CRISIS_RESPONSE = EMERGENCY_RESPONSE = ""
try:
    from app.pipeline import wellbeing_check, WELLBEING_RESPONSE
except ImportError:
    wellbeing_check = lambda q: None
    WELLBEING_RESPONSE = ""

DB = Path("data/interactions.db")
APPROVED = Path("data/approved_answers.json")
MIN_CONFIDENCE = 0.018

st.set_page_config(page_title="Your OSHC", layout="centered",
                   initial_sidebar_state="collapsed")

st.markdown("""
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
.block-container{max-width:780px;padding:1.6rem 1.4rem 7rem;background:var(--paper);
  border-left:1px solid var(--line);border-right:1px solid var(--line);min-height:100vh}

h1{font-size:1.3rem!important;font-weight:700;letter-spacing:-.01em;
  color:var(--ink);margin-bottom:.1rem!important}
.sub{font-size:.82rem;color:var(--ink-2);margin:0 0 .2rem}
.hdr{display:flex;align-items:flex-start;gap:.7rem;padding-bottom:.9rem;
  border-bottom:1px solid var(--line);margin-bottom:1.1rem}
.hdr .who{flex:1}
.pill{font-size:.68rem;font-weight:700;letter-spacing:.05em;padding:.2rem .6rem;
  background:var(--green-soft);color:var(--green);white-space:nowrap}

/* chat bubbles */
.row{display:flex;margin-bottom:.85rem}
.row.me{justify-content:flex-end}
.b{max-width:88%;padding:.65rem .9rem;font-size:.95rem;line-height:1.55}
.b.me{background:var(--ink);color:#fff}
.b.bot{background:var(--floor);color:var(--ink)}
.b.gate{background:var(--red-soft);border-left:4px solid var(--red)}
.b.crisis{background:var(--red-soft);border:1px solid var(--red);
  border-left:6px solid var(--red);padding:.85rem 1rem}
.b.wellbeing{background:var(--green-soft);border-left:4px solid var(--green)}
.b.unsure{background:var(--amber-soft);border-left:4px solid var(--amber)}
.b .tel{font-weight:700;white-space:nowrap}

/* source panel */
.cmp{display:flex;border:1px solid var(--line);margin-bottom:.6rem;overflow:hidden}
.cmp div{flex:1;padding:.55rem .75rem}
.cmp .lab{font-size:.63rem;letter-spacing:.1em;font-weight:700;margin:0 0 .1rem;color:var(--ink-2)}
.cmp .val{font-size:1.15rem;font-weight:700;margin:0;font-variant-numeric:tabular-nums}
.cmp .cap{font-size:.7rem;color:var(--ink-2);margin:.08rem 0 0}
.cmp .ours{background:var(--green-soft)} .cmp .ours .val{color:var(--green)}
.cmp .theirs{background:var(--amber-soft);border-left:1px solid var(--line)}
.cmp .theirs .val{color:var(--amber)}
.srcbox{background:var(--paper);border:1px solid var(--line);padding:.7rem .85rem;margin-bottom:.5rem}
.srcwhere{font-size:.72rem;color:var(--ink-2);margin:0 0 .4rem;
  display:flex;flex-wrap:wrap;gap:.2rem .65rem;align-items:baseline}
.srcwhere b{color:var(--ink)}
.srcwords{margin:0;font-size:.83rem;line-height:1.5;color:var(--ink-2)}
.srcgrade{font-size:.71rem;margin:.45rem 0 0;padding-top:.4rem;
  border-top:1px dashed var(--line);color:var(--amber);font-weight:600}
.srcgrade.easy{color:var(--green)}
.privnote{font-size:.71rem;color:var(--ink-2);margin:.55rem 0 0;line-height:1.5}

details{border:1px solid var(--line)!important;border-radius:0!important;
  margin:-.4rem 0 1rem}
summary{font-size:.8rem!important;font-weight:600;color:var(--blue)!important}

.stButton > button{border-radius:0;border:1.5px solid var(--line);font-size:.82rem;
  font-weight:500;color:var(--ink-2);background:var(--paper);padding:.3rem .7rem}
.stButton > button:hover{border-color:var(--blue);color:var(--blue)}
.notice{font-size:.72rem;color:var(--ink-2);margin:.5rem 0 0}

div[data-testid="stMetric"]{background:var(--floor);padding:.7rem .85rem;
  border:1px solid var(--line)}
div[data-testid="stMetricValue"]{font-size:1.35rem;color:var(--ink)}
div[data-testid="stMetricLabel"]{font-size:.75rem;color:var(--ink-2)}
#MainMenu,footer{visibility:hidden}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------- logging
def init_db():
    DB.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS interactions (
        ts TEXT, topic TEXT, confidence REAL, page TEXT,
        source TEXT, routed_to_nurse INTEGER)""")
    con.commit()
    return con


def log(topic, confidence, page, source, routed=0):
    """Matched topic and confidence only. Never the question text, never a
    member identity. Supports Australian Privacy Principle 2."""
    con = init_db()
    con.execute("INSERT INTO interactions VALUES (?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), topic,
                 confidence, page, source, routed))
    con.commit()
    con.close()


# ----------------------------------------------------------------- helpers
@st.cache_resource
def load_retriever():
    return Retriever()


@st.cache_data
def load_approved():
    return json.loads(APPROVED.read_text()) if APPROVED.exists() else {}


def grade_of(text):
    try:
        import textstat
        return round(textstat.flesch_kincaid_grade(text), 1)
    except Exception:
        return None


def reads_as(g):
    if g is None: return ""
    if g >= 18: return "Postgraduate level"
    if g >= 13: return "University level"
    if g >= 10: return "Senior school level"
    return "About a 10 year old could read it"


def bubble(text, kind="bot"):
    """Dollar signs are escaped because Streamlit renders text between two
    of them as LaTeX, which turns a benefit amount into a formula."""
    safe = html.escape(str(text)).replace("$", "&#36;")
    side = "me" if kind == "me" else ""
    st.markdown(f'<div class="row {side}"><div class="b {kind}">{safe}</div></div>',
                unsafe_allow_html=True)


def sources_panel(answer_text, passages):
    """A page number tells a student nothing. The actual sentence lets them
    check, and shows why the plain version exists."""
    ours = grade_of(answer_text)
    grades = [g for g in (grade_of(p["text"]) for p in passages) if g is not None]
    worst = max(grades) if grades else None

    with st.expander("What this is based on"):
        if ours is not None and worst is not None:
            st.markdown(
                f'<div class="cmp">'
                f'<div class="ours"><p class="lab">THIS ANSWER</p>'
                f'<p class="val">Grade {ours}</p><p class="cap">{reads_as(ours)}</p></div>'
                f'<div class="theirs"><p class="lab">THE DOCUMENT</p>'
                f'<p class="val">Grade {worst}</p><p class="cap">{reads_as(worst)}</p></div>'
                f'</div>', unsafe_allow_html=True)

        for p in passages:
            g = grade_of(p["text"])
            words = html.escape(p["text"][:400])
            if len(p["text"]) > 400:
                words += "..."
            cls = " easy" if (g is not None and g <= 9) else ""
            tail = ("readable" if (g is not None and g <= 9)
                    else "harder than most people can read comfortably")
            st.markdown(
                f'<div class="srcbox">'
                f'<p class="srcwhere"><b>{html.escape(p["section"])}</b>'
                f'<span>page {p["page"]}</span><span>{html.escape(p["product"])}</span>'
                f'<span>effective {html.escape(str(p["effective"]))}</span></p>'
                f'<p class="srcwords">&ldquo;{words}&rdquo;</p>'
                + (f'<p class="srcgrade{cls}">Reading grade {g} &middot; {tail}</p>'
                   if g is not None else "")
                + '</div>', unsafe_allow_html=True)

        st.markdown('<p class="privnote">We saved the topic of this question. '
                    'We did not save what you typed, and it is not linked to you.</p>',
                    unsafe_allow_html=True)


# ------------------------------------------------------------- the routing
def respond(question, product, page_name):
    """Returns (kind, text, passages). Tier 1 runs first, always."""
    tier = crisis_check(question)
    if tier == "crisis":
        log("crisis", 1.0, page_name, "lifeline", routed=1)
        return "crisis", CRISIS_RESPONSE, None
    if tier == "emergency":
        log("emergency", 1.0, page_name, "triple_zero", routed=1)
        return "crisis", EMERGENCY_RESPONSE, None

    if wellbeing_check(question):
        log("wellbeing", 1.0, page_name, "support_line", routed=1)
        return "wellbeing", WELLBEING_RESPONSE, None

    blocked, _ = safety_gate(question)
    if blocked:
        log("clinical", 1.0, page_name, "nurse_line", routed=1)
        return "gate", GATE_RESPONSE, None

    hits = load_retriever().search(question, product=product, k=3)
    if not hits:
        log("unmatched", 0.0, page_name, "fallback")
        return "unsure", ("I do not have an answer for that. Please call the free "
                          f"Student Health and Support Line on {NURSE_LINE}."), None

    confidence = float(hits[0][1])
    if confidence < MIN_CONFIDENCE:
        # Retrieval always returns something. Without a floor, an unrelated
        # question gets a confident answer from whatever ranked first.
        log("unmatched", confidence, page_name, "below_threshold")
        return "unsure", ("I do not have an answer for that. Please call the free "
                          f"Student Health and Support Line on {NURSE_LINE}."), None

    passages = [h[0] for h in hits]
    topic = passages[0]["section"]
    approved = load_approved().get(product, {}).get(topic)
    member = {"product": product.title(),
              "campus": st.session_state.get("campus"),
              "weeks": st.session_state.get("weeks")}
    text, _trace, source = generate_answer(question, passages, member,
                                           fallback=approved)
    log(topic, confidence, page_name, source)
    return "bot", text, passages


# ----------------------------------------------------------------- the chat
SUGGESTIONS = ["Is the dentist covered?", "Why did the doctor charge me?",
               "Can I see a psychologist?", "Do I have to wait?",
               "How do I claim money back?"]


def chat(page_name):
    key = f"chat_{page_name}"
    if key not in st.session_state:
        st.session_state[key] = [("bot",
            "Ask me what your cover includes. Every answer comes from your own "
            "policy documents, and you can open the exact wording it came from. "
            "If you need medical advice I will put you through to a nurse instead.",
            None)]

    for kind, text, passages in st.session_state[key]:
        bubble(text, kind)
        if passages:
            sources_panel(text, passages)

    cols = st.columns(len(SUGGESTIONS))
    picked = None
    for c, s in zip(cols, SUGGESTIONS):
        if c.button(s, key=f"{page_name}_{s}"):
            picked = s

    typed = st.chat_input("Ask about your cover")
    q = picked or typed
    if q:
        st.session_state[key].append(("me", q, None))
        kind, text, passages = respond(
            q, st.session_state.get("product", "comprehensive"), page_name)
        st.session_state[key].append((kind, text, passages))
        st.rerun()

    st.markdown('<p class="notice">Please do not type personal health details. '
                'Medical questions go to a nurse, not to this assistant.</p>',
                unsafe_allow_html=True)


def header(title, sub):
    product = st.session_state.get("product", "comprehensive")
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(f'<div class="hdr"><div class="who"><h1>{title}</h1>'
                    f'<p class="sub">{sub}</p></div>'
                    f'<span class="pill">ACTIVE</span></div>',
                    unsafe_allow_html=True)
    with c2:
        other = "essentials" if product == "comprehensive" else "comprehensive"
        if st.button(f"Switch to {other.title()}"):
            st.session_state["product"] = other
            st.rerun()


# -------------------------------------------------------------------- pages
def page_module():
    product = st.session_state.get("product", "comprehensive")
    header("How healthcare works here", f"{product.title()} OSHC")
    st.caption("Five things worth knowing before you need them.")
    for t, b in [
        ("See a GP first",
         "In Australia your regular doctor is called a GP, short for general "
         "practitioner. You see a GP first for anything that is not an emergency."),
        ("Specialists need a referral",
         "You cannot book a specialist directly. The GP writes you a referral."),
        ("Some clinics cost you nothing",
         "Direct Billing clinics bill us instead of you. Find one before you need it."),
        ("Emergency departments are for emergencies",
         "For anything not urgent, a GP is faster and usually cheaper."),
        ("Some things are not covered",
         "Dental check-ups, glasses and physiotherapy are not included."),
    ]:
        with st.expander(t):
            st.write(b)
    st.divider()
    chat("module")


def page_dashboard():
    product = st.session_state.get("product", "comprehensive")
    header("Your cover", f"{product.title()} OSHC")
    a, b, c = st.columns(3)
    a.metric("Cover ends", "Feb 2028")
    b.metric("Medicine left", "$412")
    c.metric("Nearest no-cost doctor", "1.2 km")
    st.divider()
    chat("dashboard")


def page_internal():
    st.markdown('<h1>Internal view</h1>'
                '<p class="sub">Aggregate patterns only. No individual students.</p>',
                unsafe_allow_html=True)
    con = init_db()
    rows = con.execute("""SELECT topic, COUNT(*) n, AVG(confidence) c
                          FROM interactions GROUP BY topic ORDER BY n DESC""").fetchall()
    con.close()
    if not rows:
        st.info("No interactions logged yet.")
        return
    st.dataframe([{"Topic": t, "Questions": n, "Avg confidence": round(c or 0, 3)}
                  for t, n, c in rows], use_container_width=True)
    st.caption("Topics with low average confidence are where our answers are "
               "weakest and should be rewritten first.")


# --------------------------------------------------------------------- main
with st.sidebar:
    st.subheader("Demo settings")
    st.session_state["campus"] = st.text_input("Campus", "Swinburne")
    st.session_state["weeks"] = st.number_input("Weeks in Australia", 0, 200, 3)
    page = st.radio("Page", ["1. Module", "2. Dashboard", "3. Internal"])

st.session_state.setdefault("product", "comprehensive")

{"1. Module": page_module,
 "2. Dashboard": page_dashboard,
 "3. Internal": page_internal}[page]()
