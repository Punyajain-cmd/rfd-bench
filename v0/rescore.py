"""Strict: reply must be exactly '<A-E>, t=<int>' (whitespace-trimmed); anything else = wrong answer (-1).
Lenient: first standalone A-E letter, else label/option-text match; step from 't=N' or first integer; unparseable = abstain (0)."""
import json,re,glob,collections
from common import load,score
rows={r["id"]:r for r in load("test_public")+load("test_private")}; L="ABCDE"
def strict(r,raw):
    m=re.fullmatch(r"\s*([A-E])\s*,\s*t\s*=\s*(\d+)\s*\.?\s*",raw)
    return (r["labels"][L.index(m.group(1))],int(m.group(2))) if m else ("__unparsed__",None)
def lenient(r,raw):
    a=re.search(r"\b([A-E])\b",raw); lab=None
    if a: lab=r["labels"][L.index(a.group(1))]
    else:
        lo=raw.lower()
        for l,o in zip(r["labels"],r["options"]):
            if l.replace("_"," ") in lo or o.lower()[:40] in lo: lab=l;break
    s=re.search(r"t\s*=\s*(\d+)",raw) or re.search(r"\b(\d{1,2})\b",raw)
    return (lab or "insufficient_evidence", int(s.group(1)) if s else None)
out=[]
for fn in sorted(glob.glob("results/raw_*.jsonl")):
    R=[json.loads(l) for l in open(fn)]; tag=fn[12:-6]
    sub=[rows[x["id"]] for x in R]
    for nm,fx in [("strict",strict),("lenient",lenient)]:
        p={x["id"]:fx(rows[x["id"]],x["raw"]) for x in R}
        unp=sum(v[0]=="__unparsed__" for v in p.values())
        res={sp:score([r for r in sub if r["split"]==sp],p) for sp in ["test_public","test_private"]}; res["all"]=score(sub,p)
        out.append(f"{tag:32s} {nm:8s} n={len(R)} unparsed={unp} {res}")
print("\n".join(out))
