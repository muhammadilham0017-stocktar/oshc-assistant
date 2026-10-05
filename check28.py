
import pdfplumber
with pdfplumber.open("data/Medibank_OSHC_Member_Guide.pdf") as pdf:
    page = pdf.pages[27]
    mid = page.width / 2
    L = page.crop((0, 0, mid, page.height)).extract_text() or ""
    print("LEFT COLUMN, lines containing optical or dental:")
    for line in L.split("\n"):
        low = line.lower()
        if "optical" in low or "dental" in low or "ancillary" in low:
            print("   ", line)
    print()
    print("does left column contain the word dental:", "dental" in L.lower())
