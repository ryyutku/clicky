"""
Florence-2 zero-shot GUI grounding inference.

Week 1 deliverable: given a screenshot + natural language instruction,
get Florence-2-base's zero-shot prediction for the target element's
bounding box, with NO fine-tuning yet. This produces the baseline table
that every later result (QLoRA fine-tuned, +classical CV aid) gets
compared against.

Florence-2 supports several task prompts out of the box; grounding-relevant
ones include:
    <CAPTION_TO_PHRASE_GROUNDING>  - needs a text phrase, returns boxes per phrase
    <OPEN_VOCABULARY_DETECTION>    - open-ended "find this thing" detection
    <REGION_TO_DESCRIPTION>        - inverse direction, not used here

For GUI grounding ("click the Downloads folder") the natural fit is
<CAPTION_TO_PHRASE_GROUNDING> with the instruction (or the noun phrase
extracted from it) as the input text. Confirm against the Florence-2 model
card in Week 1 and adjust `task_prompt` in configs/config.yaml if needed -
this is flagged as an open item, not a settled design decision.

Usage:
    python -m src.inference.florence2_inference \\
        --image path/to/screenshot.png \\
        --instruction "click the Downloads folder"
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image


@dataclass
class GroundingPrediction:
    bbox_xyxy: Optional[tuple[float, float, float, float]]  # None if no box returned
    raw_output: dict
    latency_ms: float


class Florence2Grounder:
    """Thin wrapper around Florence-2 for the grounding task, kept separate
    from training code so the same class works for zero-shot baseline,
    fine-tuned checkpoint eval, and the later action-layer inference loop."""

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        task_prompt: str = "<CAPTION_TO_PHRASE_GROUNDING>",
    ):
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.task_prompt = task_prompt
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        print(f"Loading {model_id} on {self.device} ({self.dtype}) ...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)

        if checkpoint_path:
            # Week 4+: load a QLoRA adapter on top of the base model.
            from peft import PeftModel

            print(f"Loading LoRA adapter from {checkpoint_path} ...")
            self.model = PeftModel.from_pretrained(self.model, checkpoint_path)

        self.model.eval()

    @torch.inference_mode()
    def ground(self, image: Image.Image, instruction: str) -> GroundingPrediction:
        """Run zero-shot (or fine-tuned) grounding for one (image, instruction) pair."""
        prompt = self.task_prompt + instruction

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(
            self.device, self.dtype
        )

        start = time.perf_counter()
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            generated_text, task=self.task_prompt, image_size=(image.width, image.height)
        )

        bbox = self._extract_first_bbox(parsed)
        return GroundingPrediction(bbox_xyxy=bbox, raw_output=parsed, latency_ms=latency_ms)

    def _extract_first_bbox(self, parsed: dict) -> Optional[tuple[float, float, float, float]]:
        """Florence-2's post-processed output nests boxes under the task key;
        pull the first one out. Returns None if the model found nothing -
        a real, expected outcome worth logging rather than crashing on."""
        task_output = parsed.get(self.task_prompt, {})
        bboxes = task_output.get("bboxes")
        if not bboxes:
            return None
        x1, y1, x2, y2 = bboxes[0]
        return (float(x1), float(y1), float(x2), float(y2))


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def draw_prediction(image: Image.Image, bbox: Optional[tuple], out_path: str):
    """Save a copy of the image with the predicted box drawn on it, for quick
    visual sanity-checking - useful in the notebook and for report figures."""
    from PIL import ImageDraw

    img = image.copy()
    draw = ImageDraw.Draw(img)
    if bbox:
        draw.rectangle(bbox, outline="red", width=4)
    else:
        draw.text((10, 10), "NO BOX PREDICTED", fill="red")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    print(f"Saved visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Florence-2 zero-shot GUI grounding (Week 1 baseline).")
    parser.add_argument("--image", required=True, help="Path to a screenshot image.")
    parser.add_argument("--instruction", required=True, help="Natural language instruction, e.g. 'click the Downloads folder'.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional LoRA checkpoint dir (Week 4+).")
    parser.add_argument("--save-viz", default=None, help="Optional path to save a visualized prediction image.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]

    grounder = Florence2Grounder(
        model_id=model_cfg["base_model_id"],
        checkpoint_path=args.checkpoint,
        task_prompt=model_cfg.get("task_prompt", "<CAPTION_TO_PHRASE_GROUNDING>"),
    )

    image = Image.open(args.image).convert("RGB")
    result = grounder.ground(image, args.instruction)

    print(f"\nInstruction: {args.instruction}")
    print(f"Predicted bbox (x1, y1, x2, y2): {result.bbox_xyxy}")
    print(f"Latency: {result.latency_ms:.1f} ms")
    print(f"Raw model output: {result.raw_output}")

    if args.save_viz:
        draw_prediction(image, result.bbox_xyxy, args.save_viz)


if __name__ == "__main__":
    main()
