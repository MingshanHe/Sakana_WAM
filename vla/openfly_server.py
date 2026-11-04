"""OpenFly-Agent (v1, 7B OpenVLA) action server for zero-shot tests (env: conda `sakana_vla`, run with
PYTHONNOUSERSITE=1 because ~/.local has a newer huggingface_hub that breaks transformers 4.47).

Needs the OpenFly-Platform code (its HF classes are not in the model repo):
    git clone https://github.com/SHAILAB-IPEC/OpenFly-Platform ~/vla/OpenFly-Platform

Protocol (directory requests, like dreamer/wan_server.py):
  <root>/<req>/request.json  {"text": instruction, "frames": [current, previous, the one before], "prompt": "chat"|"raw"}
  <root>/<req>/action.json   {"raw": [8 floats], "id": action id 0-9, "seconds": model time}   written here
Inference follows OpenFly's training format (train/datasets/dataset.py), which differs from its released
train/eval.py in three ways: the Llama-2 chat prompt "What action should the robot take to ...?" (eval.py passes
the raw text), the current frame first (eval.py passes the oldest first), and the 8-dim output mapped to the
nearest discrete action (eval.py's exact match turns e.g. "forward 5 m" into "stop"). Greedy decoding.

    python openfly_server.py --root ../dreamer/data/vla_requests
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

OPENFLY = os.path.expanduser("~/vla/OpenFly-Platform/train")
sys.path.insert(0, OPENFLY)
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor  # noqa: E402

from extern.hf.configuration_prismatic import OpenFlyConfig  # noqa: E402
from extern.hf.modeling_prismatic import OpenVLAForActionPrediction  # noqa: E402
from extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor  # noqa: E402

AutoConfig.register("openvla", OpenFlyConfig)
AutoImageProcessor.register(OpenFlyConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenFlyConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenFlyConfig, OpenVLAForActionPrediction)

ACTIONS = [  # OpenFly train/eval.py convert_to_action_id
    [1, 0, 0, 0, 0, 0, 0, 0],   # 0 stop
    [0, 3, 0, 0, 0, 0, 0, 0],   # 1 forward 3 m
    [0, 0, 15, 0, 0, 0, 0, 0],  # 2 turn left 30 deg
    [0, 0, 0, 15, 0, 0, 0, 0],  # 3 turn right 30 deg
    [0, 0, 0, 0, 2, 0, 0, 0],   # 4 up
    [0, 0, 0, 0, 0, 2, 0, 0],   # 5 down
    [0, 0, 0, 0, 0, 0, 5, 0],   # 6 left
    [0, 0, 0, 0, 0, 0, 0, 5],   # 7 right
    [0, 6, 0, 0, 0, 0, 0, 0],   # 8 forward 6 m
    [0, 9, 0, 0, 0, 0, 0, 0],   # 9 forward 9 m
]


SYSTEM = ("You are a helpful language and vision assistant. You are able to understand the visual content that the user "
          "provides, and assist the user with a variety of tasks using natural language.")


def chat_prompt(text):
    """The training prompt (train/datasets/dataset.py: LLaMa2ChatPromptBuilder("prismatic"), including its "<<SYS>"
    typo); OpenFly's released train/eval.py passes the raw instruction instead."""
    return f"[INST] <<SYS>\n{SYSTEM}\n<</SYS>>\n\nWhat action should the robot take to {text.lower().strip()}? [/INST]"


UNIT = np.array([1, 3, 15, 15, 2, 2, 5, 5], float)  # magnitude of each dimension's canonical action


def to_id(raw):
    """Nearest of the 10 discrete actions (each dimension in units of its canonical value). OpenFly's eval.py needs
    an exact match after rounding and calls everything else "stop"; the model often outputs e.g. forward 5 m, which
    that rule turns into "stop"."""
    x = np.asarray(raw, float) / UNIT
    return int(np.argmin([np.linalg.norm(x - np.asarray(v, float) / UNIT) for v in ACTIONS]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="IPEC-COMMUNITY/openfly-agent-7b")
    ap.add_argument("--idle-exit", type=float, default=1800)
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model)
    policy = AutoModelForVision2Seq.from_pretrained(args.model, attn_implementation="sdpa", torch_dtype=torch.bfloat16,
                                                    low_cpu_mem_usage=True).to("cuda:0")
    unnorm = next(iter(policy.norm_stats))  # the checkpoint ships one key ("vln_norm"); eval.py's "vlnv1" is stale
    open(os.path.join(args.root, "READY"), "w").write(str(os.getpid()))
    print("openfly server ready, unnorm key", unnorm, flush=True)
    last = time.time()
    while time.time() - last < args.idle_exit:
        todo = sorted((os.path.getmtime(p), os.path.dirname(p)) for p in glob.glob(os.path.join(args.root, "*", "request.json"))
                      if not os.path.exists(os.path.join(os.path.dirname(p), "action.json")))
        if not todo:
            time.sleep(0.02)
            continue
        d = todo[0][1]
        req = json.load(open(os.path.join(d, "request.json")))
        images = [Image.open(os.path.join(d, f)).convert("RGB") for f in req["frames"]]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prompt = chat_prompt(req["text"]) if req.get("prompt", "chat") == "chat" else req["text"]
        inputs = processor(prompt, images).to("cuda:0", dtype=torch.bfloat16)
        with torch.no_grad():
            raw = policy.predict_action(**inputs, unnorm_key=unnorm, do_sample=False)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        out = {"raw": [float(x) for x in raw], "id": to_id(raw), "seconds": sec}
        json.dump(out, open(os.path.join(d, "action.tmp"), "w"))
        os.replace(os.path.join(d, "action.tmp"), os.path.join(d, "action.json"))
        print(json.dumps({"req": os.path.basename(d), **out}), flush=True)
        last = time.time()


if __name__ == "__main__":
    main()
