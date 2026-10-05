
import json
import numpy as np
c = json.load(open("data/chunks.json"))
v = np.load("data/vectors.npy")
print("chunks:", len(c), " vectors:", v.shape)
print("in sync:", len(c) == v.shape[0])
hits = [x for x in c if "dental or physiotherapy" in x["text"].lower()]
print("chunks with the dental exclusion sentence:", len(hits))
for x in hits:
    print("  [" + x["section"] + "] p." + str(x["page"]))
    print("   ", x["text"][:160])
