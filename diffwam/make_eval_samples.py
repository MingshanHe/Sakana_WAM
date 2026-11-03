"""Test samples from the 10 evaluation scenarios (houses never used for training): seeds 0 and 1 each.

Copies I0 + depth from ../follow/scenarios and writes data/samples_eval/<id>/sample.json + index.json, in the same
format as make_starts.py, so wan_worker.py / label_teacher.py run on it unchanged.
"""

import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "follow", "scenarios")
OUT = os.path.join(HERE, "data", "samples_eval")


def main():
    index = []
    for sc in json.load(open(os.path.join(SRC, "scenarios.json")))["scenarios"]:
        for seed in (0, 1):
            sid = f"{sc['id']}_seed{seed}"
            d = os.path.join(OUT, sid)
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(SRC, sc["id"], "I0.png"), d)
            shutil.copy(os.path.join(SRC, sc["id"], "I0_depth.npy"), d)
            sample = {"id": sid, "house": sc["house"], "pose": sc["start"]["pose"], "motion": sc["motion"],
                      "instruction": sc["instruction"], "seed": seed, "split": "test", "hfov_deg": 90.0,
                      "width": 832, "height": 480}
            json.dump(sample, open(os.path.join(d, "sample.json"), "w"), indent=2)
            index.append(sample)
    json.dump(index, open(os.path.join(OUT, "index.json"), "w"), indent=1)
    print(len(index), "test samples")


if __name__ == "__main__":
    main()
