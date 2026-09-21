#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InternVL3.5 LoRA SFT smoke test (no monkey-patch, source code already patched)"""
import json, os, sys, time, traceback
import torch

MODELS = [
    ("InternVL3_5-2B", "/root/models/models/OpenGVLab--InternVL3_5-2B/snapshots/master"),
    ("InternVL3_5-8B", "/root/models/models/OpenGVLab--InternVL3_5-8B/snapshots/master"),
]

results = []
for model_id, model_path in MODELS:
    result = {"id": model_id, "path": model_path, "status": "unknown"}
    print(f"\n{'='*60}\nTesting: {model_id}\n{'='*60}")
    model = None
    try:
        from transformers import AutoModel, AutoTokenizer
        print("[1/4] Load tokenizer ...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print("[2/4] Load model ...")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                                          device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
        load_time = time.time() - t0
        mem = torch.cuda.memory_allocated() / 1024**3
        print(f"  Loaded in {load_time:.1f}s, GPU: {mem:.1f} GB")
        result["load_time_s"] = round(load_time, 1)
        result["load_mem_gb"] = round(mem, 1)

        print("[3/4] Inject LoRA ...")
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], bias="none")
        model = get_peft_model(model, lora_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        pct = trainable/total*100 if total > 0 else 0
        print(f"  Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
        result["trainable_params"] = trainable
        result["total_params"] = total
        result["trainable_pct"] = round(pct, 4)

        print("[4/4] Forward + backward ...")
        model.train()
        text = "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        labels = inputs["input_ids"].clone()
        inputs["labels"] = labels
        batch = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"  Loss: {loss.item():.4f}, GPU peak: {peak:.1f} GB")
        result["loss"] = round(loss.item(), 4)
        result["gpu_peak_gb"] = round(peak, 1)
        result["status"] = "success"
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()
        result["status"] = "failed"
        result["error"] = str(e)[:500]
    finally:
        try: del model
        except: pass
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    results.append(result)

print("\n" + "="*60 + "\nSummary\n" + "="*60)
for r in results:
    s = r["status"]
    ls = str(r.get("load_time_s","-"))
    m = str(r.get("load_mem_gb","-"))
    tp = f"{r.get('trainable_pct',0):.2f}%" if "trainable_pct" in r else "-"
    l = str(r.get("loss","-"))
    p = str(r.get("gpu_peak_gb","-"))
    print(f"  {r['id']:<20} {s:<8} load={ls}s mem={m}GB train={tp} loss={l} peak={p}GB")
n = sum(1 for r in results if r["status"]=="success")
print(f"\n{n}/{len(results)} InternVL models passed.")
with open("runs/sft_smoke_internvl_final.json","w",encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
