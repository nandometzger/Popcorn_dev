"""
Spike tool (popmap Phase 0): time Popcorn training epochs without the imagery.

Runs run_train.py's own Trainer (model, transforms, normalisation, loss, optimiser,
DataLoader, collate, sparse head) but swaps Population_Dataset for a stand-in that
returns random S2 (4 bands) and S1 (2 bands) arrays with the exact window size of each
census unit in a Popcorn census CSV (bbox + 32 px admin overlap, as __getadminitem__).
The same max_pix / max_pix_box filters are applied, so steps per epoch match the real run.
What it leaves out: reading and NaN-filling the GeoTIFF windows (done in DataLoader workers).

Run from the Popcorn root; everything after "--" goes to run_train's parser:
    POPCORN_DATA=<dir with PopMapData> WANDB_MODE=disabled PYTORCH_ENABLE_MPS_FALLBACK=1 \
    python tools/synthetic_epoch.py --census <census_coarse.csv> [--max-steps N] -- \
        -S2 -NIR -S1 -treg rwa -tregtrain rwa -occmodel -wd 0.00001 -senbuilds -pret \
        --biasinit 0.9407 -e 1 --save_dir <outdir>
"""

import argparse
import json
import os
import random
import resource
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

POPCORN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if POPCORN_ROOT not in sys.path:
    sys.path.insert(0, POPCORN_ROOT)

ADMIN_OVERLAP = 32  # as in Population_Dataset.__getadminitem__


class MaxStepsReached(Exception):
    pass


class SyntheticPopulationDataset(Dataset):
    """Stand-in for Population_Dataset: same constructor, random inputs of the real unit sizes."""

    census_csv = None  # set by main() before the Trainer is built

    def __init__(self, region, S1=False, S2=True, VIIRS=True, NIR=False, patchsize=1024, overlap=32,
                 fourseasons=False, mode="test", max_samples=None, transform=None, sentinelbuildings=True,
                 ascfill=False, ascAug=False, train_level="fine", split="all", max_pix=5e6,
                 max_pix_box=12000000):
        super().__init__()
        self.region, self.mode, self.S1, self.S2, self.NIR = region, mode, S1, S2, NIR
        if mode != "weaksup":
            self.census = pd.DataFrame()  # test loader is built but not iterated in epoch 0
            return
        c = pd.read_csv(self.census_csv)
        bb = c["bbox"].str.strip("()[]").str.split(",", expand=True).astype(int)
        c["xmin"], c["xmax"], c["ymin"], c["ymax"] = bb[0], bb[1], bb[2], bb[3]
        if split != "all":
            raise ValueError("only split='all' is mirrored here")
        c = c[c["count"] < max_pix]
        c = c[(c.xmax - c.xmin) * (c.ymax - c.ymin) < max_pix_box].reset_index(drop=True)
        self.census = c
        stats = json.load(open(os.path.join(POPCORN_ROOT, "data", "config", "dataset_stats.json")))
        s2key = "sen2springNIR" if NIR else "sen2spring"
        self.s2_mean = np.array(stats[s2key]["mean"], dtype=np.float32)[:, None, None]
        self.s2_std = np.array(stats[s2key]["std"], dtype=np.float32)[:, None, None]
        self.s1_mean = np.array(stats["sen1"]["mean"], dtype=np.float32)[:, None, None]
        self.s1_std = np.array(stats["sen1"]["std"], dtype=np.float32)[:, None, None]
        print("Synthetic dataset:", region, "units", len(c))

    def __len__(self):
        return len(self.census)

    def __getitem__(self, index):
        r = self.census.loc[index]
        x0 = max(r.xmin - ADMIN_OVERLAP, 0)
        y0 = max(r.ymin - ADMIN_OVERLAP, 0)
        h = r.xmax + ADMIN_OVERLAP - x0
        w = r.ymax + ADMIN_OVERLAP - y0
        rng = np.random.default_rng(int(index) + 1000 * random.randint(0, 10**6))
        out = {}
        if self.S2:
            nb = 4 if self.NIR else 3
            s2 = rng.standard_normal((nb, h, w), dtype=np.float32) * self.s2_std[:nb] + self.s2_mean[:nb]
            out["S2"] = torch.from_numpy(np.clip(s2, 0, 10000))
        if self.S1:
            s1 = rng.standard_normal((2, h, w), dtype=np.float32) * self.s1_std + self.s1_mean
            out["S1"] = torch.from_numpy(s1)
        # admin mask: unit id on `count` pixels inside the bbox, 0 elsewhere
        admin = np.zeros((h, w), dtype=np.float32)
        inner = admin[r.xmin - x0: r.xmax - x0, r.ymin - y0: r.ymax - y0].reshape(-1)
        inner[: int(r["count"])] = float(r["idx"])
        admin[r.xmin - x0: r.xmax - x0, r.ymin - y0: r.ymax - y0] = inner.reshape(r.xmax - r.xmin, r.ymax - r.ymin)
        return {
            **out,
            "y": torch.tensor(float(r["POP20"]), dtype=torch.float32),
            "admin_mask": torch.from_numpy(admin),
            "img_coords": (int(r.xmin), int(r.ymin)),
            "valid_coords": (int(r.xmin), int(r.xmax), int(r.ymin), int(r.ymax)),
            "season": random.randint(0, 3),
            "census_idx": torch.tensor([int(r["idx"])]),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", required=True, help="Popcorn census CSV (idx, POP20, bbox, count)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop the epoch after N steps")
    ap.add_argument("--report", default=None, help="write timing JSON here")
    own, rest = ap.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    sys.argv = ["run_train.py"] + rest  # arguments/train.py parses sys.argv at import

    # collect every distinct warning (MPS CPU fallbacks show up as UserWarnings)
    seen = {}
    _show = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        key = "{}: {}".format(category.__name__, message)
        if key not in seen:
            seen[key] = "{}:{}".format(filename, lineno)
            _show(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning

    SyntheticPopulationDataset.census_csv = own.census

    import run_train
    from utils.device import get_device, synchronize

    run_train.Population_Dataset = SyntheticPopulationDataset
    run_train.args = run_train.train_parser.parse_args()  # train_epoch reads the module-level args

    device = get_device()
    print("device:", device, "| torch", torch.__version__)

    # per-step timestamps and memory, taken where run_train calls empty_cache() once per step
    step_t, mem_drv, mem_cur = [], [], []
    orig_empty_cache = run_train.empty_cache

    def empty_cache_probe():
        step_t.append(time.perf_counter())
        if device.type == "mps":
            mem_drv.append(torch.mps.driver_allocated_memory())
            mem_cur.append(torch.mps.current_allocated_memory())
        elif device.type == "cuda":
            mem_drv.append(torch.cuda.max_memory_reserved())
            mem_cur.append(torch.cuda.max_memory_allocated())
        if own.max_steps is not None and len(step_t) >= own.max_steps:
            raise MaxStepsReached()
        orig_empty_cache()

    run_train.empty_cache = empty_cache_probe

    t0 = time.perf_counter()
    trainer = run_train.Trainer(run_train.args)
    t_init = time.perf_counter() - t0
    steps_per_epoch = len(trainer.dataloaders["train"])
    print("steps per epoch:", steps_per_epoch)

    t1 = time.perf_counter()
    stopped_early = False
    try:
        trainer.train()
    except MaxStepsReached:
        stopped_early = True
    synchronize()
    t_train = time.perf_counter() - t1

    n = len(step_t) if stopped_early else len(step_t) - 1  # last probe call is the end-of-epoch one
    dts = np.diff([t1] + step_t[:n])
    rep = {
        "device": str(device),
        "torch": torch.__version__,
        "steps_per_epoch": steps_per_epoch,
        "steps_done": int(n),
        "stopped_early": stopped_early,
        "init_s": round(t_init, 1),
        "train_wall_s": round(t_train, 1),
        "first_step_s": round(float(dts[0]), 2) if n else None,
        "median_step_s": round(float(np.median(dts)), 3) if n else None,
        "mean_step_s_excl_first": round(float(np.mean(dts[1:])), 3) if n > 1 else None,
        "it_per_s_excl_first": round(float(1 / np.mean(dts[1:])), 3) if n > 1 else None,
        "est_epoch_s": round(float(dts[0] + np.mean(dts[1:]) * (steps_per_epoch - 1)), 1) if n > 1 else None,
        "peak_driver_mem_gb": round(max(mem_drv) / 1e9, 2) if mem_drv else None,
        "peak_current_alloc_gb": round(max(mem_cur) / 1e9, 2) if mem_cur else None,
        "max_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2),
        "warnings": seen,
    }
    print("TIMING " + json.dumps(rep, indent=1))
    if own.report:
        with open(own.report, "w") as f:
            json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
