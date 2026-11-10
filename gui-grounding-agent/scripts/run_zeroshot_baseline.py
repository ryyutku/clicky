#!/usr/bin/env python
"""
Week 1 deliverable: "Inference script and baseline table".

Runs Florence-2 zero-shot over a ScreenSpot subset (normalized JSONL from
src/data/screenspot_loader.py) and reports click accuracy + mean IoU overall
and per-category - the baseline every later ablation compares against.

Usage:
    python scripts/run_zeroshot_baseline.py --n 50
    python scripts/run_zeroshot_baseline.py --n 50 --out results/week1_baseline.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.screenspot_loader import load_config, load_examples_jsonl
from src.inference.florence2_inference import Florence2Grounder


def click_in_box(pred_bbox, gt_bbox) -> bool:
    """Standard ScreenSpot metric: does the predicted point/box center fall
    inside the ground-truth box?"""
    if pred_bbox is None:
        return False
    px = (pred_bbox[0] + pred_bbox[2]) / 2
    py = (pred_bbox[1] + pred_bbox[3]) / 2
    x1, y1, x2, y2 = gt_bbox
    return x1 <= px <= x2 and y1 <= py <= y2


def iou(pred_bbox, gt_bbox) -> float:
    if pred_bbox is None:
        return 0.0
    px1, py1, px2, py2 = pred_bbox
    gx1, gy1, gx2, gy2 = gt_bbox

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    pred_area = max(0, px2 - px1) * max(0, py2 - py1)
    gt_area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    union = pred_area + gt_area - inter
    return inter / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=None, help="Defaults to configs/config.yaml's screenspot cache path.")
    parser.add_argument("--n", type=int, default=50, help="Number of examples to evaluate (keep small for a quick Week 1 check).")
    parser.add_argument("--out", default="results/week1_zeroshot_baseline.csv")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    jsonl_path = args.jsonl or str(Path(cfg["data"]["screenspot"]["cache_dir"]) / "screenspot_normalized.jsonl")

    if not Path(jsonl_path).exists():
        print(f"ERROR: {jsonl_path} not found.")
        print("Run `python -m src.data.screenspot_loader` first to download and normalize ScreenSpot.")
        sys.exit(1)

    examples = load_examples_jsonl(jsonl_path)[: args.n]
    print(f"Evaluating zero-shot baseline on {len(examples)} examples ...")

    grounder = Florence2Grounder(
        model_id=cfg["model"]["base_model_id"],
        task_prompt=cfg["model"].get("task_prompt", "<CAPTION_TO_PHRASE_GROUNDING>"),
    )

    from PIL import Image

    rows = []
    for ex in examples:
        image = Image.open(ex.image_path).convert("RGB")
        pred = grounder.ground(image, ex.instruction)

        rows.append(
            {
                "example_id": ex.example_id,
                "category": ex.category,
                "platform": ex.platform,
                "instruction": ex.instruction,
                "gt_bbox": ex.bbox_xyxy,
                "pred_bbox": pred.bbox_xyxy,
                "click_correct": click_in_box(pred.bbox_xyxy, ex.bbox_xyxy),
                "iou": iou(pred.bbox_xyxy, ex.bbox_xyxy),
                "latency_ms": pred.latency_ms,
            }
        )

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"\nSaved per-example results to {args.out}")
    print("\n=== Overall ===")
    print(f"Click accuracy: {df['click_correct'].mean():.3f}")
    print(f"Mean IoU:       {df['iou'].mean():.3f}")
    print(f"Mean latency:   {df['latency_ms'].mean():.1f} ms")

    if df["category"].notna().any():
        print("\n=== Per category ===")
        print(df.groupby("category")[["click_correct", "iou"]].mean())


if __name__ == "__main__":
    main()
