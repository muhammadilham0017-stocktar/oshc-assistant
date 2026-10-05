
import sys
sys.path.insert(0, ".")
from ingest.build_index import extract_pages, split_document

pages, _ = extract_pages("data/Medibank_OSHC_Member_Guide.pdf")
got = [t for n, t in pages if n == 28]
print("page 28 kept by extract_pages:", bool(got))
if got:
    print("  contains dental:", "dental" in got[0].lower())

chunks = split_document(pages, "comprehensive", "2026-05")
hits = [c for c in chunks if "dental" in c["text"].lower()]
print("chunks mentioning dental:", len(hits))
for c in hits[:3]:
    print("  [" + c["section"] + "]", c["text"][:120])
