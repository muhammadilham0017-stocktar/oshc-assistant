
import json
from collections import Counter
c = json.load(open("data/chunks.json"))
tiny = [x for x in c if len(x["text"].split()) < 15]
huge = [x for x in c if len(x["text"].split()) > 350]
print("SHORTEST CHUNKS, by section")
for s, n in Counter(x["section"] for x in tiny).most_common(5):
    print("  ", n, s)
print()
print("  examples:")
for x in tiny[:4]:
    print("    [" + x["section"] + "]", x["text"][:70])
print()
print("LONGEST CHUNKS")
for x in sorted(huge, key=lambda z: -len(z["text"].split()))[:3]:
    print("  ", len(x["text"].split()), "words  [" + x["section"] + "] p." + str(x["page"]))
    print("     ", x["text"][:90])
