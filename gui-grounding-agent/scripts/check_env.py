#!/usr/bin/env python
"""
Week 1 environment check for CM4610 Part B.

Run this first, before anything else. It verifies:
  - Python version
  - GPU visibility + VRAM (must fit an 8GB budget - RTX 5050)
  - Required packages are importable and report their versions
  - bitsandbytes can actually see CUDA (common QLoRA setup failure point)
  - Florence-2 can be loaded with trust_remote_code (common version-pin failure point)

Usage:
    python scripts/check_env.py
    python scripts/check_env.py --skip-model-load   # fast check, skips the ~1GB download
"""
import argparse
import importlib
import platform
import sys


def ok(msg):
    print(f"  [OK]   {msg}")


def warn(msg):
    print(f"  [WARN] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")


def section(title):
    print(f"\n=== {title} ===")


def check_python():
    section("Python")
    v = sys.version_info
    print(f"  {platform.python_version()} ({platform.platform()})")
    if v < (3, 10):
        warn("Python < 3.10: some packages (labelImg, newer transformers) may not install cleanly.")
    elif v >= (3, 13):
        warn("Python >= 3.13: bitsandbytes/labelImg wheels may lag behind this version.")
    else:
        ok("Python version is in a safe range (3.10-3.12).")


def check_package(name, import_name=None, min_version=None):
    import_name = import_name or name
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        ok(f"{name} {version}")
        return version
    except ImportError as e:
        fail(f"{name} not importable: {e}")
        return None


def check_packages():
    section("Core packages")
    results = {}
    for name, import_name in [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("timm", "timm"),
        ("einops", "einops"),
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("pillow", "PIL"),
        ("datasets", "datasets"),
        ("mss", "mss"),
        ("pyautogui", "pyautogui"),
    ]:
        results[name] = check_package(name, import_name)
    return results


def check_gpu():
    section("GPU / CUDA")
    try:
        import torch
    except ImportError:
        fail("torch not installed - cannot check GPU. Install requirements.txt first.")
        return None

    if not torch.cuda.is_available():
        fail(
            "CUDA not available to torch. Training/inference will fall back to CPU, "
            "which will be far too slow for a VLM. Check your PyTorch CUDA build "
            "(https://pytorch.org/get-started/locally/) and NVIDIA driver."
        )
        return False

    name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  Device: {name}")
    print(f"  Total VRAM: {total_vram_gb:.1f} GB")
    print(f"  CUDA (torch build): {torch.version.cuda}")

    if total_vram_gb < 7.0:
        warn(
            f"Only {total_vram_gb:.1f} GB VRAM detected. QLoRA fine-tuning of even a "
            "small VLM may need careful batch size / gradient accumulation tuning."
        )
    else:
        ok(f"{total_vram_gb:.1f} GB VRAM should fit Florence-2-base QLoRA with batch size 2-4.")
    return True


def check_bitsandbytes_cuda():
    section("bitsandbytes CUDA backend")
    try:
        import torch
        import bitsandbytes as bnb  # noqa: F401
    except ImportError as e:
        fail(f"Cannot check - import failed: {e}")
        return
    if not torch.cuda.is_available():
        warn("Skipped (no CUDA available).")
        return
    try:
        # A small 4-bit quantization smoke test - this is the step that most often breaks
        # on Windows / mismatched CUDA toolkit versions.
        import torch.nn as nn
        from bitsandbytes.nn import Linear4bit

        layer = Linear4bit(16, 16).cuda()
        x = torch.randn(2, 16, dtype=torch.float16).cuda()
        _ = layer(x)
        ok("4-bit Linear layer ran successfully on GPU - QLoRA quantization path works.")
    except Exception as e:
        fail(
            f"bitsandbytes 4-bit op failed on GPU: {e}\n"
            "         This is the #1 QLoRA setup-friction risk flagged in the brief. "
            "On native Windows, consider WSL2 or the Qwen2-VL-2B / Moondream2 fallback."
        )


def check_florence2_loads(skip):
    section("Florence-2-base load test")
    if skip:
        warn("Skipped (--skip-model-load). Run without this flag before Week 1 ends.")
        return
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        import torch

        model_id = "microsoft/Florence-2-base"
        print(f"  Downloading/loading {model_id} (trust_remote_code=True)... this may take a minute.")
        AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        ok("Florence-2-base loaded successfully with trust_remote_code=True.")
    except Exception as e:
        fail(
            f"Florence-2 failed to load: {e}\n"
            "         Pin transformers per requirements.txt. If this persists, fall back "
            "to Qwen2-VL-2B or Moondream2 per the brief's risk mitigation (Section 11)."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip the Florence-2 download/load smoke test (faster, but leaves the riskiest check untested).",
    )
    args = parser.parse_args()

    print("CM4610 Part B - Environment Check")
    print("=" * 40)

    check_python()
    check_packages()
    has_gpu = check_gpu()
    if has_gpu:
        check_bitsandbytes_cuda()
    check_florence2_loads(args.skip_model_load)

    print("\n" + "=" * 40)
    print("Done. Fix any [FAIL] lines before starting Week 1 baseline work.")
    print("[WARN] lines are worth noting in your report's 'environment setup' notes.")


if __name__ == "__main__":
    main()
