"""
Step 1. Turn the Medibank PDFs into searchable chunks.

Runs offline. Repeat only when Medibank republishes a document.
Nothing is trained here. We split, tag and embed.

    python ingest/build_index.py data/Medibank_OSHC_Member_Guide.pdf comprehensive

Output: data/chunks.json
"""
import json
import re
import sys
from pathlib import Path

import pdfplumber

# Section headings from the Member Guide contents pages.
# Add to this list rather than falling back to token splitting.
HEADINGS = [
    "Welcome to Medibank",
    "Your membership",
    "Activating your cover",
    "Online Member Services",
    "What your cover includes",
    "Hospital cover",
    "Medical cover",
    "Prescription medicines",
    "Ambulance services",
    "Mental health support",
    "Waiting periods",
    "Pre-existing Conditions",
    "Benefit exclusions",
    "Making a claim",
    "Direct Billing",
    "Finding a doctor",
    "Changing your cover",
    "Cancelling your cover",
    "Policy duration",
    "Glossary",
]

# These three need different handling, see split_document()
GLOSSARY = "Glossary"
EXCLUSIONS = "Benefit exclusions"


def extract_pages(pdf_path):
    """Returns [(page_number, text)].

    The Member Guide is a two column layout. Reading the page as one block
    interleaves the columns, producing text like "Dental surgery in a
    Waiting period Service hospital" where a table header has been spliced
    into the middle of a sentence. Splitting the page down the middle and
    reading each column separately fixes it.

    Contents pages are skipped. They are page number lists, and read as
    prose they produce nonsense that still scores well on retrieval."""
    pages, tables = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            mid = page.width / 2
            left = page.crop((0, 0, mid, page.height)).extract_text() or ""
            right = page.crop((mid, 0, page.width, page.height)).extract_text() or ""
            single = page.extract_text() or ""

            # Two columns only if both halves carry real text. A full width
            # page puts almost everything in one half.
            if len(left.split()) > 25 and len(right.split()) > 25:
                text = left + "\n" + right
            else:
                text = single

            if _is_contents(text):
                continue

            pages.append((i, text))
            for t in page.extract_tables():
                tables.append((i, t))
    return pages, tables


def _is_contents(text):
    """A contents page is mostly short lines ending in a page number."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 8:
        return False
    numbered = sum(1 for l in lines if re.search(r"\s\d{1,3}$", l))
    return numbered / len(lines) > 0.5


def split_document(pages, product, effective):
    """Split by heading, never by token count. A rule separated from its
    condition cannot be recovered by any later step."""
    chunks = []
    current = {"section": "Front matter", "page": 1, "lines": []}

    for page_no, text in pages:
        for line in text.split("\n"):
            stripped = line.strip()
            matched = next((h for h in HEADINGS
                            if stripped.lower().startswith(h.lower())), None)
            if matched:
                if current["lines"]:
                    chunks.append(_finish(current, product, effective))
                current = {"section": matched, "page": page_no, "lines": []}
            else:
                current["lines"].append(stripped)
    if current["lines"]:
        chunks.append(_finish(current, product, effective))

    out = []
    for c in chunks:
        if c["section"] == GLOSSARY:
            out.extend(_split_glossary(c))
        elif c["section"] == EXCLUSIONS:
            out.extend(_split_bullets(c))
        else:
            out.append(c)
    return [c for c in out if len(c["text"].split()) > 8]


def _finish(current, product, effective):
    body = " ".join(l for l in current["lines"] if l)
    body = re.sub(r"\s+", " ", body).strip()
    return {
        "section": current["section"],
        "page": current["page"],
        "product": product,
        "effective": effective,
        "text": body,
        # Breadcrumb prepended before embedding so the section travels
        # with the text even when the body never repeats it.
        "embed_text": f"{current['section']}: {body}",
    }


def _split_glossary(chunk):
    """One chunk per term. Someone asking what a waiting period is should
    hit the definition, not a paragraph that mentions it."""
    out = []
    for m in re.finditer(r"([A-Z][A-Za-z \-]{2,40})\s+means\s+(.+?)(?=[A-Z][A-Za-z \-]{2,40}\s+means|$)",
                         chunk["text"]):
        term, definition = m.group(1).strip(), m.group(2).strip()
        out.append({**chunk,
                    "section": f"Glossary: {term}",
                    "text": f"{term} means {definition}",
                    "embed_text": f"Glossary > {term}: {term} means {definition}"})
    return out or [chunk]


def _split_bullets(chunk, group=5):
    """Exclusions are independent rules, but one bullet per chunk makes
    them too short to compete. A six word fragment scores highly on
    partial matches and displaces the longer section that actually
    answers the question. Grouping restores a comparable chunk length."""
    parts = re.split(r"(?:^|\s)[\u2022\-\u2013]\s+", chunk["text"])
    parts = [p.strip() for p in parts if len(p.strip().split()) > 5]
    out = []
    for i in range(0, len(parts), group):
        body = ". ".join(parts[i:i + group])
        out.append({**chunk, "text": body,
                    "embed_text": f"{chunk['section']}: {body}"})
    return out or [chunk]


def table_chunks(tables, product, effective):
    """Keep the grid. Flattened text mixes up which hospital type gives
    which benefit."""
    out = []
    for page_no, rows in tables:
        if not rows or len(rows) < 2:
            continue
        header = [h or "" for h in rows[0]]
        for row in rows[1:]:
            cells = [c or "" for c in row]
            pairs = ", ".join(f"{h.strip()}: {c.strip()}"
                              for h, c in zip(header, cells) if c.strip())
            if len(pairs.split()) > 4:
                out.append({
                    "section": "Benefits table",
                    "page": page_no,
                    "product": product,
                    "effective": effective,
                    "text": pairs,
                    "embed_text": f"Benefits table: {pairs}",
                })
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    pdf_path, product = sys.argv[1], sys.argv[2]
    effective = sys.argv[3] if len(sys.argv) > 3 else "2026-05"

    pages, tables = extract_pages(pdf_path)
    chunks = split_document(pages, product, effective)
    chunks += table_chunks(tables, product, effective)

    out = Path("data/chunks.json")
    existing = json.loads(out.read_text()) if out.exists() else []
    # replace any chunks for this product, keep the others
    src = Path(pdf_path).name
    for c in chunks:
        c["source"] = src
    existing = [c for c in existing if c.get("source") != src]
    existing.extend(chunks)
    out.write_text(json.dumps(existing, indent=2))

    print(f"{pdf_path} -> {len(chunks)} chunks for '{product}'")
    by_section = {}
    for c in chunks:
        by_section[c["section"]] = by_section.get(c["section"], 0) + 1
    for sec, n in sorted(by_section.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:3}  {sec}")
    print(f"\ntotal in index: {len(existing)}")


if __name__ == "__main__":
    main()
