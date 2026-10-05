
import pdfplumber
with pdfplumber.open("data/Medibank_OSHC_Member_Guide.pdf") as pdf:
    for i, page in enumerate(pdf.pages, 1):
        full = page.extract_text() or ""
        if "ancillary" not in full.lower():
            continue
        print("=== page", i)
        for line in full.split("\n"):
            low = line.lower()
            if "ancillary" in low or "optical" in low:
                print("   ", line)
        mid = page.width / 2
        L = page.crop((0, 0, mid, page.height)).extract_text() or ""
        R = page.crop((mid, 0, page.width, page.height)).extract_text() or ""
        print("    left words:", len(L.split()), " right words:", len(R.split()))
        print("    in left:", "ancillary" in L.lower(),
              "  in right:", "ancillary" in R.lower())
