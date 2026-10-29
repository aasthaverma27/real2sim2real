#!/usr/bin/env python3
"""Split a LeRobot v2.1 dataset into train / val datasets by holding out N episodes per task.

The held-out episodes are spread evenly over each task's episodes (first..last), so val is not one
recording session. Both outputs are standalone datasets renumbered 0..N-1; state, action, videos and
every other column are copied unchanged. meta/split_from.jsonl in each records the source episode.
The source is never modified; the output dir must not exist.

    python split_dataset.py <dataset> --out <dir> --val-per-task 5
    # -> <dir>/train, <dir>/val
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.is_file() else []


def write_jsonl(p: Path, rows: list[dict]) -> None:
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def stats_of(x) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "count": [len(x)]}


def write_subset(src: Path, out: Path, episodes: list[int]) -> tuple[int, int]:
    info = json.loads((src / "meta/info.json").read_text())
    video_keys = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    src_eps = {e["episode_index"]: e for e in jsonl(src / "meta/episodes.jsonl")}
    src_stats = {s["episode_index"]: s for s in jsonl(src / "meta/episodes_stats.jsonl")}
    src_times = {t["episode_index"]: t for t in jsonl(src / "meta/episode_times.jsonl")}
    tasks = jsonl(src / "meta/tasks.jsonl")

    (out / "meta").mkdir(parents=True)
    for name in ("modality.json", "tasks.jsonl"):   # task indices keep their meaning in both splits
        if (src / "meta" / name).is_file():
            shutil.copy(src / "meta" / name, out / "meta" / name)

    eps, stats, times, origin = [], [], [], []
    gidx = 0
    for new, old in enumerate(episodes):
        cs, cn = old // info["chunks_size"], new // info["chunks_size"]
        df = pd.read_parquet(src / info["data_path"].format(episode_chunk=cs, episode_index=old))
        df["episode_index"] = np.int64(new)
        df["index"] = np.arange(gidx, gidx + len(df), dtype=np.int64)
        dst = out / info["data_path"].format(episode_chunk=cn, episode_index=new)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(dst, index=False)
        for vk in video_keys:
            sv = src / info["video_path"].format(episode_chunk=cs, video_key=vk, episode_index=old)
            dv = out / info["video_path"].format(episode_chunk=cn, video_key=vk, episode_index=new)
            dv.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sv, dv)
        eps.append(dict(src_eps[old], episode_index=new))
        s = {"episode_index": new, "stats": dict(src_stats[old]["stats"])}
        for col in ("episode_index", "index"):
            s["stats"][col] = stats_of(df[col].values)
        stats.append(s)
        if old in src_times:
            times.append(dict(src_times[old], episode_index=new))
        origin.append({"episode_index": new, "from_episode": old, "task": src_eps[old]["tasks"][0]})
        gidx += len(df)

    write_jsonl(out / "meta/episodes.jsonl", eps)
    write_jsonl(out / "meta/episodes_stats.jsonl", stats)
    if times:
        write_jsonl(out / "meta/episode_times.jsonl", times)
    write_jsonl(out / "meta/split_from.jsonl", origin)
    used = {e["tasks"][0] for e in eps}
    info.update(total_episodes=len(eps), total_frames=gidx, total_videos=len(eps) * len(video_keys),
                total_tasks=sum(t["task"] in used for t in tasks),
                total_chunks=(len(eps) - 1) // info["chunks_size"] + 1, splits={"train": f"0:{len(eps)}"})
    (out / "meta/info.json").write_text(json.dumps(info, indent=4))
    return len(eps), gidx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-per-task", type=int, default=5)
    ap.add_argument("--val-episodes", default=None,
                    help="explicit comma-separated val episode indices; overrides --val-per-task "
                         "(e.g. to keep relit replays and their originals on the same side)")
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f"{a.out} already exists")

    by_task: dict[str, list[int]] = {}
    for e in jsonl(a.dataset / "meta/episodes.jsonl"):
        by_task.setdefault(e["tasks"][0], []).append(e["episode_index"])
    if a.val_episodes:
        val = sorted(int(x) for x in a.val_episodes.split(","))
        known = {i for idx in by_task.values() for i in idx}
        if not set(val) <= known:
            raise SystemExit(f"unknown val episodes: {sorted(set(val) - known)}")
    else:
        val = []
        for task, idx in by_task.items():
            idx = sorted(idx)
            if len(idx) <= a.val_per_task:
                raise SystemExit(f"task '{task}' has only {len(idx)} episodes")
            pick = np.linspace(0, len(idx) - 1, a.val_per_task + 2)[1:-1].round().astype(int)  # skip the ends
            val += [idx[i] for i in pick]
        val = sorted(val)
    train = sorted(set(i for idx in by_task.values() for i in idx) - set(val))

    for name, eps in (("train", train), ("val", val)):
        n, f = write_subset(a.dataset, a.out / name, eps)
        print(f"{a.out / name}: {n} episodes, {f} frames")
    print("val source episodes:", val)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
