"""
ScreenSpot dataset loader.

ScreenSpot provides (image, instruction, bounding_box) triples across web,
mobile and desktop UI screenshots - the public base dataset referenced in
Section 7 of the project brief.

This module:
  1. Downloads/caches the dataset from the Hugging Face Hub.
  2. Normalizes it into a flat list of GroundingExample records (one schema
     shared with the self-collected annotations, so downstream code - the
     preprocessing module, the training loop, the eval scripts - never has
     to know which source an example came from).
  3. Optionally writes train/val/test split CSVs.

NOTE (Week 1 action item): the exact HF repo id for ScreenSpot has moved /
been re-uploaded by different groups over time (e.g. ScreenSpot-v2,
ScreenSpot-Pro variants exist). Confirm the current canonical repo on
huggingface.co/datasets before relying on the default in configs/config.yaml,
and update `hf_repo` there once confirmed. This loader is written against the
common schema (image, instruction/prompt, bbox) used by most ScreenSpot
releases, with light normalization to absorb small field-name differences.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class GroundingExample:
    """Unified schema for every grounding example, regardless of source."""

    example_id: str
    image_path: str
    instruction: str
    bbox_xyxy: tuple[float, float, float, float]  # absolute pixel coords (x1, y1, x2, y2)
    image_width: int
    image_height: int
    category: Optional[str] = None      # button / text_field / icon / menu / checkbox
    platform: Optional[str] = None      # web / mobile / desktop
    source: str = "screenspot"          # "screenspot" | "self_collected" | "ood_test"

    def to_dict(self):
        d = asdict(self)
        d["bbox_xyxy"] = list(d["bbox_xyxy"])
        return d


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _normalize_bbox(raw_bbox, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """
    ScreenSpot releases store bbox as either [x, y, w, h] or [x1, y1, x2, y2],
    and sometimes normalized to [0,1]. Normalize everything to absolute xyxy.
    """
    x1, y1, a, b = raw_bbox
    # Heuristic: if a < x1 or b < y1, it's almost certainly (w, h) not (x2, y2).
    if a < x1 or b < y1:
        x2, y2 = x1 + a, y1 + b
    else:
        x2, y2 = a, b

    # Heuristic: values in [0, 1] are normalized fractions of image size.
    if max(x1, y1, x2, y2) <= 1.0:
        x1, x2 = x1 * img_w, x2 * img_w
        y1, y2 = y1 * img_h, y2 * img_h

    return (float(x1), float(y1), float(x2), float(y2))


def download_screenspot(cache_dir: str, hf_repo: str):
    """
    Pulls ScreenSpot via the `datasets` library. Returns the raw HF dataset
    object (dict-like, usually keyed by split or just one split called
    'train'/'test' depending on the release).
    """
    from datasets import load_dataset

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"Loading '{hf_repo}' into cache_dir='{cache_dir}' ...")
    ds = load_dataset(hf_repo, cache_dir=cache_dir)
    print(f"Loaded. Splits/keys available: {list(ds.keys())}")
    return ds


def normalize_screenspot_split(hf_split, image_out_dir: str, limit: Optional[int] = None) -> list[GroundingExample]:
    """
    Converts one HF dataset split into a list of GroundingExample, saving
    each image to disk under image_out_dir (HF datasets keep images in an
    arrow cache, not as loose files, so we materialize them once for the
    rest of the pipeline - OpenCV preprocessing, the data loader - to read
    directly as files).
    """
    Path(image_out_dir).mkdir(parents=True, exist_ok=True)
    examples: list[GroundingExample] = []

    n = len(hf_split) if limit is None else min(limit, len(hf_split))
    for i in range(n):
        row = hf_split[i]

        # Field names vary a little by release; check the common aliases.
        image = row.get("image") or row.get("img")
        instruction = (
            row.get("instruction")
            or row.get("prompt")
            or row.get("query")
            or ""
        )
        raw_bbox = row.get("bbox") or row.get("bounding_box") or row.get("box")
        category = row.get("category") or row.get("element_type")
        platform = row.get("platform") or row.get("data_type")

        if image is None or raw_bbox is None:
            continue  # skip malformed rows rather than crash a long download

        img_w, img_h = image.size
        image_path = str(Path(image_out_dir) / f"screenspot_{i:06d}.png")
        if not Path(image_path).exists():
            image.save(image_path)

        bbox = _normalize_bbox(raw_bbox, img_w, img_h)

        examples.append(
            GroundingExample(
                example_id=f"screenspot_{i:06d}",
                image_path=image_path,
                instruction=instruction,
                bbox_xyxy=bbox,
                image_width=img_w,
                image_height=img_h,
                category=category,
                platform=platform,
                source="screenspot",
            )
        )

    return examples


def save_examples_jsonl(examples: list[GroundingExample], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_dict()) + "\n")
    print(f"Wrote {len(examples)} examples to {out_path}")


def load_examples_jsonl(path: str) -> list[GroundingExample]:
    examples = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            d["bbox_xyxy"] = tuple(d["bbox_xyxy"])
            examples.append(GroundingExample(**d))
    return examples


def main():
    """
    Week 1 entry point: download a ScreenSpot subset and write it out as a
    normalized JSONL ready for the baseline zero-shot eval.

    Run: python -m src.data.screenspot_loader
    """
    cfg = load_config()
    ss_cfg = cfg["data"]["screenspot"]

    ds = download_screenspot(ss_cfg["cache_dir"], ss_cfg["hf_repo"])
    split_name = "test" if "test" in ds else list(ds.keys())[0]
    print(f"Using split: '{split_name}'")

    fraction = ss_cfg.get("subset_fraction", 1.0)
    limit = None if fraction >= 1.0 else int(len(ds[split_name]) * fraction)

    examples = normalize_screenspot_split(
        ds[split_name],
        image_out_dir=str(Path(ss_cfg["cache_dir"]) / "images"),
        limit=limit,
    )
    save_examples_jsonl(examples, str(Path(ss_cfg["cache_dir"]) / "screenspot_normalized.jsonl"))

    if examples:
        print("\nSample example:")
        print(json.dumps(examples[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
