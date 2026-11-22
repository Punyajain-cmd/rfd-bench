import json,glob,os
def load(split=None):
    rows=[]
    for f in ["data/train.jsonl","data/val.jsonl","data/test_public.jsonl","data/private/test_private.jsonl"]:
        if not os.path.exists(f): continue  # test_private is withheld (not in this repo)
        rows+= [json.loads(l) for l in open(f)]
    return [r for r in rows if split is None or r["split"]==split]
def score(rows,preds):
    """preds: id->(label,step). AA-Omniscience style: +1 correct, -1 wrong, 0 abstain."""
    s=c=st=n_ans=0
    for r in rows:
        lab,step=preds[r["id"]]
        if lab=="insufficient_evidence":
            s+= 1 if r["gold_label"]==lab else 0
            c+= r["gold_label"]==lab
        else:
            ok=lab==r["gold_label"]; s+= 1 if ok else -1; c+=ok; n_ans+=1
            st+= ok and step==r["gold_step"]
    n=len(rows); ans=[r for r in rows if r["gold_label"]!="insufficient_evidence"]
    return dict(n=n,acc=round(c/max(1,n),3),omni=round(s/max(1,n),3),cause_and_step=round(st/max(1,len(ans)),3))
