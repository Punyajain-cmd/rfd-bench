import sys,json,re,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from common import load,score
torch.set_num_threads(4)
name=sys.argv[1]; mode=sys.argv[2] if len(sys.argv)>2 else "full"
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float32).eval()
rows=load("test_public")+load("test_private"); L="ABCDE"; preds={}
for r in rows:
    ev=r["episode_text"] if mode=="full" else "Instruction: "+r["instruction"]+"\n(log not provided)"
    opts="\n".join(f"{L[i]}. {o}" for i,o in enumerate(r["options"]))
    msg=f"{ev}\n\nWhich option best explains the outcome, and at which step index t did things first go wrong?\n{opts}\nReply exactly as: <letter>, t=<number>"
    ids=tok.apply_chat_template([{"role":"user","content":msg}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"]
    with torch.no_grad(): out=m.generate(ids,max_new_tokens=12,do_sample=False)
    txt=tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)
    a=re.search(r"[A-E]",txt); s=re.search(r"t\s*=\s*(\d+)",txt)
    lab=r["labels"][L.index(a.group())] if a else "insufficient_evidence"
    preds[r["id"]]=(lab,int(s.group(1)) if s else None)
res={sp:score([r for r in rows if r["split"]==sp],preds) for sp in ["test_public","test_private"]}
res["all_test"]=score(rows,preds)
tag=name.split("/")[-1]+"_"+mode
json.dump(dict(res=res,preds=preds),open(f"results/{tag}.json","w")); print(tag,res)
