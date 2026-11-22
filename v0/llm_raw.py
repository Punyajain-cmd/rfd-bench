import sys,json,os,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from common import load
torch.set_num_threads(int(os.environ.get("NT","3")))
name=sys.argv[1]; mode=sys.argv[2]; tag=name.split("/")[-1]+"_"+mode; fn=f"results/raw_{tag}.jsonl"
done={json.loads(l)["id"] for l in open(fn)} if os.path.exists(fn) else set()
tok=AutoTokenizer.from_pretrained(name); m=AutoModelForCausalLM.from_pretrained(name,torch_dtype=torch.float32).eval()
L="ABCDE"; f=open(fn,"a")
for r in load("test_public")+load("test_private"):
    if r["id"] in done: continue
    ev=r["episode_text"] if mode=="full" else "Instruction: "+r["instruction"]+"\n(log not provided)"
    opts="\n".join(f"{L[i]}. {o}" for i,o in enumerate(r["options"]))
    msg=f"{ev}\n\nWhich option best explains the outcome, and at which step index t did things first go wrong?\n{opts}\nReply exactly as: <letter>, t=<number>"
    ids=tok.apply_chat_template([{"role":"user","content":msg}],add_generation_prompt=True,return_tensors="pt",return_dict=True)["input_ids"]
    with torch.no_grad(): out=m.generate(ids,max_new_tokens=24,do_sample=False)
    f.write(json.dumps(dict(id=r["id"],raw=tok.decode(out[0][ids.shape[1]:],skip_special_tokens=True)))+"\n"); f.flush()
