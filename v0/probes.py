"""Shortcut probes. Train on train+val, eval on test_public+test_private."""
import numpy as np,random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from common import load
tr=load("train")+load("val"); te=load("test_public")+load("test_private")
def per_option(rows):
    X=[];y=[];g=[]
    for r in rows:
        for i,o in enumerate(r["options"]): X.append(f"pos{i} "+o); y.append(int(i==r["gold_index"])); g.append(r["id"])
    return X,y
def pick(rows,scores):
    acc=0
    for j,r in enumerate(rows): acc+= int(np.argmax(scores[j*5:(j+1)*5])==r["gold_index"])
    return acc/len(rows)
Xtr,ytr=per_option(tr); Xte,_=per_option(te)
v=TfidfVectorizer(ngram_range=(1,2)).fit(Xtr)
lr=LogisticRegression(max_iter=2000).fit(v.transform(Xtr),ytr)
print("chance (5 options) .200")
print("option-only tfidf_lr", round(pick(te,lr.predict_proba(v.transform(Xte))[:,1]),3))
def feats(rows):
    F=[]
    for r in rows:
        for i,o in enumerate(r["options"]):
            ins=set(r["instruction"].lower().split()); ow=set(o.lower().split())
            F.append([i,len(o),len(ow),len(ins&ow)/len(ow), int(o==max(r["options"],key=len)), int(o==min(r["options"],key=len))])
    return np.array(F)
gb=GradientBoostingClassifier(random_state=0).fit(feats(tr),ytr)
print("option-only gboost(len,pos,overlap)", round(pick(te,gb.predict_proba(feats(te))[:,1]),3))
for nm,f in [("longest",4),("shortest",5)]:
    F=feats(te); print(f"always-{nm}-option", round(pick(te,F[:,f]),3))
# label from episode length only
Ltr=np.array([[len(r["episode_text"])] for r in tr]); Lte=np.array([[len(r["episode_text"])] for r in te])
g2=GradientBoostingClassifier(random_state=0).fit(Ltr,[r["gold_label"] for r in tr])
p=g2.predict(Lte); print("label-from-length gboost", round(np.mean([a==r["gold_label"] for a,r in zip(p,te)]),3), "(majority-class chance ~%.3f)"%(max(np.unique([r['gold_label'] for r in te],return_counts=True)[1])/len(te)))
print("  length probe on abstain-vs-not: acc", round(np.mean([(a=="insufficient_evidence")==(r["gold_label"]=="insufficient_evidence") for a,r in zip(p,te)]),3))
# instruction-only (no log)
vi=TfidfVectorizer().fit([r["instruction"] for r in tr])
li=LogisticRegression(max_iter=2000).fit(vi.transform([r["instruction"] for r in tr]),[r["gold_label"] for r in tr])
print("label-from-instruction-only tfidf_lr", round(np.mean(li.predict(vi.transform([r["instruction"] for r in te]))==np.array([r["gold_label"] for r in te])),3))
# evidence probe (should be ABOVE chance: the log carries the signal)
ve=TfidfVectorizer(token_pattern=r"[A-Za-z_]+|\d+\.\d|\-\-").fit([r["episode_text"] for r in tr])
le=LogisticRegression(max_iter=3000).fit(ve.transform([r["episode_text"] for r in tr]),[r["gold_label"] for r in tr])
print("evidence-bag-of-words tfidf_lr (reference, uses log)", round(np.mean(le.predict(ve.transform([r["episode_text"] for r in te]))==np.array([r["gold_label"] for r in te])),3))
