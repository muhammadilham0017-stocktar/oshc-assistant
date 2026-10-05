
import json
c = json.load(open("data/chunks.json"))

def looks_corrupt(t):
    return t.count("(.") > 4 or "((((" in t

bad = [x for x in c if looks_corrupt(x["text"])]
print("chunks with corrupted characters:", len(bad), "of", len(c))
for x in bad[:3]:
    print("  [" + x["section"] + "] p." + str(x["page"]))
    print("   ", x["text"][:120])
print()

big = [x for x in c if len(x["text"].split()) > 350]
print("oversized chunks:", len(big))
for x in sorted(big, key=lambda z: -len(z["text"].split())):
    print("  ", len(x["text"].split()), "words  [" + x["section"] + "] p." + str(x["page"]))
