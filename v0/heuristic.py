"""Rule-based reader of the structured trace: an evidence-using reference, not a learned model."""
import random,json
from common import load,score
BINMAP=None
def diagnose(r):
    S=r["steps"]; txt=r["episode_text"]
    plan=[l for l in txt.splitlines() if l.startswith("planner")][0]
    if plan.split("goal_location=")[1].strip() not in r["instruction"].replace(" ","_").replace("bin_","bin_"):
        if plan.split("goal_location=")[1].strip().replace("_"," ") not in r["instruction"]: return "instruction_misread",0
    vis=[s for s in S if not s.get("redacted")]
    red=[s["t"] for s in S if s.get("redacted")]
    tgt=r["instruction"].split("Put the ")[1].split(" in ")[0].replace(" ","_")
    for s in vis:
        if s["det"] and s["det"][0]!=tgt: return "wrong_object",s["t"]
        if s["det"] and s["det"][1]<.45: return "perception_miss",s["t"]
        if s["contact_N"]>40: return "collision",s["t"]
        if s["q_max_ratio"]>=.999: return "joint_limit",s["t"]
        if s["obj_vz"]>1.6: return "sim_contact_artifact",s["t"]
        if s["phase"] in("lift","transport","place_approach") and s["grip_F"]<15 and s["t"]>5 and S[5].get("grip_F",0)>0: return "grasp_slip",s["t"]
    for a,b in zip(vis,vis[1:]):
        if b["t"]==a["t"]+1 and b["phase"]!=a["phase"] and abs(b["ee"][2]-a["ee"][2])<.01 and b["phase"] not in("transport","place_approach","retreat") and a["phase"] not in ("grasp",): return "timeout",a["t"]
        if b["t"]==a["t"]+1 and b["phase"] in("transport",) and a["phase"]=="transport" and b["ee"]==a["ee"]: pass
    if red and min(red)<10: return "insufficient_evidence",None
    # stalled in transport/place
    for a,b in zip(vis,vis[1:]):
        if b["phase"]!=a["phase"] and max(abs(x-y) for x,y in zip(a["ee"],b["ee"]))<.01 and b["phase"]!="release": return "timeout",a["t"]
    return "insufficient_evidence",None
te=load("test_public")+load("test_private"); rng=random.Random(0)
out={}
out["random"]={r["id"]:(rng.choice(r["labels"]),rng.randrange(16)) for r in te}
out["always_abstain"]={r["id"]:("insufficient_evidence",None) for r in te}
out["heuristic_rules"]={r["id"]:diagnose(r) for r in te}
for k,p in out.items():
    print(k,{sp:score([r for r in te if r["split"]==sp],p) for sp in ["test_public","test_private"]},"all",score(te,p))
# counterfactual pair consistency for heuristic
pairs={}
for r in te:
    if r["pair_id"]: pairs.setdefault(r["pair_id"],[]).append(r)
p=out["heuristic_rules"]; both=sum(all(p[r["id"]][0]==r["gold_label"] for r in v) for v in pairs.values())
print("heuristic: pairs with both twins right",both,"/",len(pairs))
