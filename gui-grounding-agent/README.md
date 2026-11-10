# VLM-Based GUI Grounding Agent — CM4610 Part B

Local, open-weight VLM agent that takes a screenshot + natural-language
instruction and localizes the target UI element (GUI grounding). Tests
whether classical CV preprocessing (edge/contour candidate regions) improves
a fine-tuned small VLM's grounding accuracy, including on OOD
adversarial/CAPTCHA-style elements. See `docs/project_brief.txt` for the
full brief (hypotheses, scope, roadmap, viva prep).

## Project structure

```
configs/config.yaml          Single source of truth for paths, model id, QLoRA params, splits
src/data/                    Dataset loading + normalization (ScreenSpot, self-collected, OOD)
src/preprocessing/           Classical CV module (Canny/Sobel, contours, candidate boxes) - Week 3
src/models/                  QLoRA fine-tuning code - Week 4
src/inference/                Florence-2 grounding wrapper (zero-shot + fine-tuned)
src/eval/                    Metrics: click accuracy, IoU, latency, per-category breakdown
src/action/                  Screenshot capture, coordinate mapping, click/type, Bezier movement - Week 5
scripts/                     Standalone entry points (env check, baseline run, etc.)
notebooks/                   The .ipynb deliverable - ties everything together with narrative
data/                        raw/, processed/, screenspot/, annotations/, ood_test/ (gitignored except ood_test)
checkpoints/                 Saved LoRA adapters (gitignored)
results/                     CSV/plot outputs from eval runs (gitignored)
```

## Week 1 setup

1. **Create and activate a virtual environment** (Python 3.10-3.12 recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   Note: `bitsandbytes` (QLoRA) has the most friction on native Windows. If
   `scripts/check_env.py` reports a `[FAIL]` on the bitsandbytes CUDA check,
   use WSL2, or fall back to Qwen2-VL-2B / Moondream2 per the brief's risk
   plan (Section 11) — both still demonstrate the same CV+VLM pipeline.

3. **Run the environment check:**
   ```bash
   python scripts/check_env.py
   ```
   This verifies GPU/VRAM, package versions, the bitsandbytes 4-bit CUDA
   path, and that Florence-2-base actually loads with `trust_remote_code`.
   Fix any `[FAIL]` before going further — this is the exact failure mode
   flagged as the #1 risk in the brief.

4. **Download + normalize a ScreenSpot subset:**
   ```bash
   python -m src.data.screenspot_loader
   ```
   **Action item before running this:** confirm the current canonical
   ScreenSpot Hugging Face repo id and set it in `configs/config.yaml`
   (`data.screenspot.hf_repo`) — see the note at the top of
   `src/data/screenspot_loader.py`.

5. **Run Florence-2 zero-shot on a single image** (sanity check the task
   prompt before running the full baseline):
   ```bash
   python -m src.inference.florence2_inference \
       --image data/screenspot/images/screenspot_000000.png \
       --instruction "click the search bar" \
       --save-viz results/sanity_check.png
   ```
   **Action item:** confirm `<CAPTION_TO_PHRASE_GROUNDING>` is the right
   Florence-2 task prompt for this use case against the model card, and
   update `model.task_prompt` in `configs/config.yaml` if not.

6. **Run the Week 1 zero-shot baseline table:**
   ```bash
   python scripts/run_zeroshot_baseline.py --n 50
   ```
   Produces `results/week1_zeroshot_baseline.csv` with per-example and
   per-category click accuracy / IoU / latency — the "Inference script and
   baseline table" deliverable for Week 1.

## Open decisions / action items carried from the brief

- Confirm the real Part B deadline on Moodle (brief's date looks stale).
- Confirm exact ScreenSpot HF repo id (naming has forked into v2/Pro variants).
- Confirm Florence-2 grounding task prompt token against its model card.
- Decide classical-CV integration method (crop hint vs. overlay vs. snap-and-rerank) — due Week 3, justify in report.
- Confirm LoRA `target_modules` once Florence-2's module names are inspected (Week 1/2).

## GenAI usage note

Per the module's GenAI policy, keep a running log of how AI assistance was
used (scaffolding vs. logic I wrote/understood myself) for the required
report section — see `docs/genai_usage_log.md`.
