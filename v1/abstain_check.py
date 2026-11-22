import json,numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut,cross_val_score
R=lambda p:[json.loads(l) for l in open(p)]
E=R('data/episodes_public.jsonl')+R('data/private/episodes_test_private.jsonl');L={l['id']:l for l in R('data/private/labels.jsonl')}
A=[e for e in E if L[e['id']]['gold_label']=='insufficient_evidence']
X=[[e['observation'][-1]['block_x']-e['observation'][0]['block_x'],np.mean([o['contact_pusher_block'] for o in e['observation']])] for e in A]
y=[L[e['id']]['latent_cause']=='heavy_payload' for e in A]; g=[e['group_id'] for e in A]
s=cross_val_score(LogisticRegression(),X,y,groups=g,cv=LeaveOneGroupOut())
d=[abs(a[0]) for a in X]
r=f"abstain rows n={len(A)}: leave-group-out LR (block displacement, contact fraction) recovers hidden heavy-vs-weak cause acc={s.mean():.3f} (chance .5)\nblock displacement heavy median={np.median([x[0] for x,t in zip(X,y) if t]):.4f} m, weak median={np.median([x[0] for x,t in zip(X,y) if not t]):.4f} m"
print(r); open('report.txt','a').write('\n\n== ABSTAIN INDISTINGUISHABILITY CHECK\n'+r+'\n')
