"""
Restyle the Streamlit app to match the HTML design, and add the source
panel with the readability comparison.

Run from the project root:

    python patch_style.py

What it changes:
  1. CSS to match the hospital wayfinding palette
  2. A styled source panel showing the real policy wording
  3. A readability comparison, this answer against the document
  4. Dollar sign escaping so benefit amounts stop rendering as LaTeX
  5. The student support number instead of the general Medibank line
"""
import re
from pathlib import Path

MAIN = Path("app/main.py")
GEN = Path("app/generate.py")

CSS = '''
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
    return str(text).replace("$", "\\\\$")


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

'''


def patch_main():
    s = MAIN.read_text()
    if "show_sources" in s:
        print("  main.py already styled")
        return

    # insert the CSS and helpers after the page config
    anchor = 'st.set_page_config(page_title="Your OSHC", layout="centered")'
    if anchor not in s:
        m = re.search(r"st\.set_page_config\([^)]*\)", s)
        if not m:
            raise SystemExit("  could not find st.set_page_config in main.py")
        anchor = m.group(0)
    s = s.replace(anchor, anchor + "\n" + CSS, 1)

    # escape dollar signs on every rendered message
    for call in ["st.success(text)", "st.info(WELLBEING_RESPONSE)",
                 "st.error(GATE_RESPONSE)", "st.error(CRISIS_RESPONSE)",
                 "st.error(EMERGENCY_RESPONSE)"]:
        inner = call[call.index("(") + 1:-1]
        s = s.replace(call, f"{call[:call.index('(')]}(md_safe({inner}))")

    # replace the plain source expander with the styled panel
    old = re.search(
        r'    with st\.expander\("Where this came from"\):\n'
        r'(?:        .*\n)+?(?=    if trace|    with st\.expander\("Checks"\))',
        s)
    if old:
        s = s[:old.start()] + "    show_sources(text, passages)\n" + s[old.end():]
        print("  source panel replaced")
    else:
        print("  could not find the old source expander, panel added but not wired")

    MAIN.write_text(s)
    print("  main.py styled")


def patch_prompt():
    """The model was quoting 134 148, which is Medibank's general line. The
    student line is 1800 887 283 and is the one in the documents."""
    s = GEN.read_text()
    if "1800 887 283" in s and "general enquiries" in s:
        print("  generate.py already patched")
        return
    s = s.replace(
        'Do not use emoji. Be warm and direct, not casual."""',
        'Do not use emoji. Be warm and direct, not casual.\n'
        'If you give a phone number, give the Student Health and Support Line\n'
        'on 1800 887 283. Do not give the general enquiries number."""')
    GEN.write_text(s)
    print("  generate.py prompt updated")


if __name__ == "__main__":
    if not MAIN.exists():
        raise SystemExit("Run this from the project root, where app/ lives.")
    patch_main()
    patch_prompt()
    print("\nNow run:  streamlit run app/main.py")
