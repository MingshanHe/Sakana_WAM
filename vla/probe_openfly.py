"""Probe OpenFly-Agent prompt / image-order variants on one test frame (env sakana_vla, PYTHONNOUSERSITE=1)."""
import os, sys, json, numpy as np, torch
from PIL import Image
sys.argv = [sys.argv[0], "--root", "/dev/null"]
import openfly_server as S
from transformers import AutoModelForVision2Seq, AutoProcessor

proc = AutoProcessor.from_pretrained("IPEC-COMMUNITY/openfly-agent-7b")
pol = AutoModelForVision2Seq.from_pretrained("IPEC-COMMUNITY/openfly-agent-7b", attn_implementation="sdpa",
                                             torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0")
key = next(iter(pol.norm_stats))
E = os.path.join(os.path.dirname(__file__), "..", "diffwam", "data", "samples_eval")

chat = S.chat_prompt

texts = {"nav": "Fly straight forward across the living room toward the far wall with the window and pictures.",
         "openfly_style": "fly forward to the window and then stop"}
for sid in ["s01_room_forward_seed0", "s05_ascend_seed0", "s07_turn_right_seed0"]:
    img = Image.open(os.path.join(E, sid, "I0.png")).convert("RGB")
    for name, t in texts.items():
        for fmt, prompt in (("raw", t), ("chat", chat(t))):
            inputs = proc(prompt, [img, img, img]).to("cuda:0", dtype=torch.bfloat16)
            raw = pol.predict_action(**inputs, unnorm_key=key, do_sample=False)
            print(sid, name, fmt, np.round(raw, 2).tolist(), "->", S.to_id(raw), flush=True)
print("pixel_values shape", inputs["pixel_values"].shape)
