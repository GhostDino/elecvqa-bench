#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""InternVL3.5 LoRA SFT smoke test - with dummy image input"""
import json, os, sys, time, traceback
import torch
import numpy as np
from PIL import Image

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
        from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor
        print("[1/4] Load tokenizer & image processor ...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        image_processor = CLIPImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        
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

        print("[4/4] Forward + backward (with dummy image) ...")
        model.train()
        
        # Create a dummy image
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        pixel_values = image_processor(images=dummy_img, return_tensors="pt")["pixel_values"]
        
        # Create input_ids with image token
        # InternVL uses <image> as image placeholder
        img_token = "<image>"
        question = f"{img_token}\nWhat is shown in the image?"
        answer = "A gray square."
        
        # Format as chat
        text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        
        # Get image_token_id
        try:
            image_token_id = tokenizer.convert_tokens_to_ids(img_token)
        except:
            image_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
        
        # Replace first padding token with image token
        if image_token_id and image_token_id != tokenizer.unk_token_id:
            # Find first occurrence of a token we can replace
            inputs["input_ids"][0, 0] = image_token_id
        
        labels = inputs["input_ids"].clone()
        inputs["labels"] = labels
        inputs["pixel_values"] = pixel_values
        
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
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
with open("runs/sft_smoke_internvl_v4.json","w",encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
