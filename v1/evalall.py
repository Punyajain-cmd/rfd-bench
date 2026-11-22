import json,hashlib,glob,collections,random,numpy as np,re,os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
R=lambda p:[json.loads(l) for l in open(p)]
E=R('data/episodes_public.jsonl')+R('data/private/episodes_test_private.jsonl'); L={l['id']:l for l in R('data/private/labels.jsonl')}
out=[]; P=lambda *a: out.append(' '.join(map(str,a)))
# ---------- validation
P('== VALIDATION'); P('episodes',len(E),'labels',len(L),'valid labels',sum(l['valid'] for l in L.values()))
P('unique ids',len({e['id'] for e in E})==len(E))
gs=collections.defaultdict(set); fs=collections.defaultdict(set)
for e in E: gs[e['group_id']].add(e['split']); fs[L[e['id']]['family']].add(e['split'])
P('groups spanning splits',sum(len(v)>1 for v in gs.values()),'families spanning splits',sum(len(v)>1 for v in fs.values()))
bad=re.compile(r'low_friction|heavy_payload|weak_actuator|hidden_obstacle|encoder_bias|insufficient_evidence|no_failure|fault|gold|abstain|label|latent|distractor|seed|eval',re.I)
P('forbidden tokens in prompts',sum(bool(bad.search(e['prompt'].split('Options:')[0])) for e in E),'in frame paths',sum(bool(bad.search(' '.join(e['frames']))) for e in E))
P('obs length set',{len(e['observation']) for e in E},'prompt chars min/max',min(len(e['prompt']) for e in E),max(len(e['prompt']) for e in E))
na=lambda e:sum(v is None for o in e['observation'] for v in o.values())
byc=collections.defaultdict(list)
for e in E: byc[L[e['id']]['gold_label']].append(na(e))
P('NA cells per class (mean,min,max):',{k:(round(np.mean(v),1),min(v),max(v)) for k,v in byc.items()})
P('mask x abstain:',collections.Counter((L[e['id']]['mask'],L[e['id']]['gold_label']=='insufficient_evidence') for e in E))
P('label x split:',dict(collections.Counter((e['split'],L[e['id']]['gold_label']) for e in E)))
P('gold_index dist:',sorted(collections.Counter(L[e['id']]['gold_index'] for e in E).items()))
P('check pass rates:',{k:f"{sum(l['checks'].get(k,True) for l in L.values())}/{sum(k in l['checks'] for l in L.values())}" for k in ['nominal_success','control_success','faulted_fails','repair_primary_succeeds','remove_distractor_still_fails']})
P('distractor present (non-control):',sum(l['distractor'] is not None for l in L.values()))
# indistinguishability of abstain pairs: visible-evidence distance heavy_b vs weak_b in same group vs across-cause distance
def vec(e): return np.array([[0 if v is None else v for k,v in o.items() if k!='t'] for o in e['observation']],float).ravel()
ab=collections.defaultdict(dict)
for e in E:
    l=L[e['id']]
    if l['gold_label']=='insufficient_evidence': ab[e['group_id']][l['latent_cause']]=vec(e)
d_ab=[np.abs(v['heavy_payload']-v['weak_actuator']).max() for v in ab.values() if len(v)==2]
P('abstain pairs: max |visible diff| heavy vs weak per group: median',round(float(np.median(d_ab)),4),'max',round(float(max(d_ab)),4))
man={}
for f in sorted(glob.glob('data/**/*',recursive=True)):
    if os.path.isfile(f) and not f.endswith('MANIFEST.sha256'): man[f]=hashlib.sha256(open(f,'rb').read()).hexdigest()
open('data/private/MANIFEST.sha256','w').write(''.join(f"{h}  {f}\n" for f,h in man.items())); P('manifest files',len(man))
# ---------- probes
tr=[e for e in E if e['split'] in('train','val')]; te=[e for e in E if e['split']=='test_public']; tp=[e for e in E if e['split']=='test_private']
y=lambda S:[L[e['id']]['gold_index'] for e in S]; yl=lambda S:[L[e['id']]['gold_label'] for e in S]
P('\n== PROBES (fit train+val, eval test_public / test_private)')
maj=collections.Counter(y(tr)).most_common(1)[0][0]
for nm,S in(('pub',te),('priv',tp)): P(f'majority gold_index acc {nm}',round(np.mean([v==maj for v in y(S)]),3))
txt=lambda S:[' || '.join(f"{i}:{o}" for i,o in enumerate(e['options'])) for e in S]
v=TfidfVectorizer(ngram_range=(1,2)).fit(txt(tr)); m=LogisticRegression(max_iter=2000).fit(v.transform(txt(tr)),y(tr))
for nm,S in(('pub',te),('priv',tp)): P(f'options-only TFIDF-LR gold_index acc {nm}',round(m.score(v.transform(txt(S)),y(S)),3),'(chance 1/7=.143)')
def shallow(e): return [len(e['prompt']),na(e)]+[sum(o[c] is None for o in e['observation']) for c in ['enc_x','force_x','block_x','contact_unexpected']]
ab_=lambda S:[int(L[e['id']]['gold_label']=='insufficient_evidence') for e in S]
g=GradientBoostingClassifier(random_state=0).fit([shallow(e) for e in tr],ab_(tr))
for nm,S in(('pub',te),('priv',tp)): P(f'length/missingness GB abstain acc {nm}',round(g.score([shallow(e) for e in S],ab_(S)),3),'majority',round(1-np.mean(ab_(S)),3))
g2=GradientBoostingClassifier(random_state=0).fit([shallow(e) for e in tr],yl(tr))
for nm,S in(('pub',te),('priv',tp)): P(f'length/missingness GB 7-way label acc {nm}',round(g2.score([shallow(e) for e in S],yl(S)),3))
st=[e for e in tr if L[e['id']]['gold_step'] is not None]
gs_=GradientBoostingClassifier(random_state=0).fit([[L[e['id']]['gold_step']] for e in st],[L[e['id']]['gold_label'] for e in st])
sp=[e for e in te+tp if L[e['id']]['gold_step'] is not None]
P('gold-step-only GB cause acc (answerable test, oracle step leak upper bound)',round(gs_.score([[L[e['id']]['gold_step']] for e in sp],[L[e['id']]['gold_label'] for e in sp]),3),'n',len(sp))
def feats(e):
    o=e['observation'];f=[]
    for c in ['enc_x','force_x','block_x','block_y','block_yaw','contact_pusher_block','contact_unexpected']:
        xs=[r[c] for r in o if r[c] is not None]; f+= [np.mean(xs),max(xs),min(xs),xs[-1],1] if xs else [0,0,0,0,0]
    f+=[o[-1]['cmd_x']]; return f
g3=GradientBoostingClassifier(random_state=0).fit([feats(e) for e in tr],yl(tr))
for nm,S in(('pub',te),('priv',tp)): P(f'telemetry-feature GB (full evidence) label acc {nm}',round(g3.score([feats(e) for e in S],yl(S)),3))
# ---------- rule baseline
def rule(e):
    o=e['observation']; c=e['contract']; gx=float(re.search(r'goal at x=([\d.]+)',c).group(1)); tol=float(re.search(r'within ([\d.]+) m',c).group(1))
    g=lambda k:[(r['t'],r[k]) for r in o if r[k] is not None]
    if all(v is None for r in o for k,v in r.items() if k!='t'): return 'insufficient_evidence',None
    un=g('contact_unexpected'); hit=[t for t,v in un if v]
    if hit: return 'hidden_obstacle',hit[0]
    enc=g('enc_x'); bxs=g('block_x'); force=g('force_x')
    if enc and o[0]['cmd_x'] is not None and abs(enc[0][1]-o[0]['cmd_x'])>.02: return 'encoder_bias',enc[0][0]
    fin=bxs[-1][1] if bxs and bxs[-1][0]==39 else None
    if fin is not None and fin>gx+tol: return 'low_friction',None
    if fin is not None and abs(fin-gx)<tol: return 'no_failure',None
    if force:
        fm=max(abs(v) for _,v in force)
        if fm<3: return 'weak_actuator',None
        if fm>5: 
            return ('heavy_payload',None) if fin is not None and fin<gx-0.1 else ('hidden_obstacle',None)
    if fin is not None and fin<gx-0.1: return 'insufficient_evidence',None
    if fin is not None and fin<gx-tol: return 'encoder_bias',None
    return 'insufficient_evidence',None
def score(name,S,pred):
    n=len(S); acc=np.mean([p[0]==L[e['id']]['gold_label'] for e,p in zip(S,pred)])
    ans=[(e,p) for e,p in zip(S,pred) if L[e['id']]['gold_label']!='insufficient_evidence']
    ab=[(e,p) for e,p in zip(S,pred) if L[e['id']]['gold_label']=='insufficient_evidence']
    omni=np.mean([1 if p[0]==L[e['id']]['gold_label'] else (0 if p[0]=='insufficient_evidence' else -1) for e,p in zip(S,pred)])
    cs=[p[0]==L[e['id']]['gold_label'] and (L[e['id']]['gold_step'] is None or (p[1] is not None and abs(p[1]-L[e['id']]['gold_step'])<=1)) for e,p in ans]
    P(f"{name}: n={n} acc={acc:.3f} answerable_acc={np.mean([p[0]==L[e['id']]['gold_label'] for e,p in ans]):.3f} abstain_recall={np.mean([p[0]=='insufficient_evidence' for e,p in ab]):.3f} omni={omni:.3f} cause+step(±1)={np.mean(cs):.3f}")
P('\n== BASELINES (scored on test_public+test_private; omni=+1 right, -1 wrong, 0 abstain)')
S=te+tp; rnd=random.Random(0); OPTS=sorted(set(yl(E)))
score('always-abstain',S,[('insufficient_evidence',None)]*len(S))
score('uniform-random',S,[(rnd.choice(OPTS),rnd.randint(0,39)) for _ in S])
score('rule (full evidence)',S,[rule(e) for e in S])
score('rule (blank evidence)',S,[rule(dict(e,observation=[{k:(v if k=='t' else None) for k,v in o.items()} for o in e['observation']])) for e in S])
sw=S[1:]+S[:1]
score('rule (evidence swapped from another row)',S,[rule(dict(e,observation=f['observation'])) for e,f in zip(S,sw)])
P('rule per-gold confusion:',dict(collections.Counter((L[e['id']]['gold_label'],rule(e)[0]) for e in S)))
open('report.txt','w').write('\n'.join(out)); print('\n'.join(out))
