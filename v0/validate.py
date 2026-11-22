import json,collections,hashlib,sys,os
from common import load
rows=load(); err=[]
ids=set()
for r in rows:
    if r["id"] in ids: err.append("dup "+r["id"])
    ids.add(r["id"])
    if r["labels"][r["gold_index"]]!=r["gold_label"]: err.append("gold idx "+r["id"])
    if len(set(r["labels"]))!=5 or "insufficient_evidence" not in r["labels"]: err.append("opts "+r["id"])
    if (r["gold_label"]=="insufficient_evidence")!=(r["gold_step"] is None): err.append("step "+r["id"])
    for bad in ["diagnos","benchmark","root cause"]:
        if bad in r["episode_text"].lower(): err.append(f"term {bad} in {r['id']}")
pairs=collections.defaultdict(list)
for r in rows:
    if r["pair_id"]: pairs[r["pair_id"]].append(r)
for p,v in pairs.items():
    if len(v)!=2 or v[0]["split"]!=v[1]["split"] or v[0]["gold_label"]==v[1]["gold_label"]: err.append("pair "+p)
man=dict(l.split("  ")[::-1][::-1] for l in [])
for l in (open("data/private/MANIFEST.sha256") if os.path.exists("data/private/MANIFEST.sha256") else []):  # private split withheld
    i,h=l.split(); r=[x for x in rows if x["id"]==i][0]
    if hashlib.sha256(r["episode_text"].encode()).hexdigest()!=h or r["split"]!="test_private": err.append("manifest "+i)
pub={r["episode_text"] for r in rows if r["split"]!="test_private"}
err+=["private text leaked "+r["id"] for r in rows if r["split"]=="test_private" and r["episode_text"] in pub]
print("rows",len(rows),"pairs",len(pairs))
for s in ["train","val","test_public","test_private"]:
    sub=[r for r in rows if r["split"]==s]
    print(s,len(sub),dict(collections.Counter(r["gold_label"] for r in sub)))
print("gold_label",dict(collections.Counter(r["gold_label"] for r in rows)))
print("gold_index",dict(sorted(collections.Counter(r["gold_index"] for r in rows).items())))
print("gold_step",dict(sorted(collections.Counter(r["gold_step"] for r in rows if r["gold_step"] is not None).items())))
L=collections.defaultdict(list)
for r in rows: L[r["gold_label"]].append(len(r["episode_text"]))
print("mean text len by label",{k:round(sum(v)/len(v)) for k,v in L.items()})
print("ERRORS",len(err),err[:10]); sys.exit(1 if err else 0)
