#!/usr/bin/env python3
"""
FLUX Art Generator — Standalone text-to-image on Apple Silicon
==============================================================
Uses mflux with FLUX.2-klein-4B (4-bit quantized, 4B params) to generate
images from text prompts. Runs entirely on Apple Silicon Metal GPU.

Usage:
    python flux_art.py "a cat astronaut floating in space"
    python flux_art.py "sunset" --width 1024 --height 768 --seed 42
    python flux_art.py   # interactive mode

    # As a library (for Beast integration)
    from flux_art import generate_image
    path = generate_image("a fox in a forest")

Output: Images saved to ./generated_art/<timestamp>_<prompt_slug>.png
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "generated_art"
# Auto-detect: check common local paths first, fall back to HuggingFace download
MODEL_LOCAL_PATHS = [
    Path.home() / "FLUX.2-klein-4B-mflux-4bit",
    Path.home() / "MLX_Models" / "FLUX.2-klein-4B-mflux-4bit",
    Path(__file__).parent / "FLUX.2-klein-4B-mflux-4bit",
]
MODEL_HF = "RunPod/FLUX.2-klein-4B-mflux-4bit"  # auto-download fallback
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 4
DEFAULT_SEED = None

_flux = None


def _slugify(text: str, max_len: int = 40) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:max_len]


def load_model():
    global _flux
    if _flux is not None:
        return _flux

    # Find model: local path first, then HuggingFace
    model_path = None
    for p in MODEL_LOCAL_PATHS:
        if p.exists() and any(p.rglob("*.safetensors")):
            model_path = str(p)
            break
    if model_path is None:
        model_path = MODEL_HF  # mflux auto-downloads from HuggingFace

    print(f"Loading FLUX.2-klein-4B from {model_path}...")
    start = time.time()

    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
    from mflux.models.common.config.model_config import ModelConfig

    _flux = Flux2Klein(
        quantize=4,
        model_path=model_path,
        model_config=ModelConfig.flux2_klein_4b(),
    )

    print(f"  Loaded in {time.time() - start:.1f}s")
    return _flux


def generate_image(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    seed: int | None = DEFAULT_SEED,
    output_dir: str | Path | None = None,
    output_filename: str | None = None,
) -> str:
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    if seed is None:
        import random
        seed = random.randint(0, 2**32 - 1)

    if output_filename:
        filepath = out / output_filename
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out / f"{ts}_{_slugify(prompt)}.png"

    print(f"\nGenerating image:")
    print(f"  Prompt: {prompt}")
    print(f"  Size:   {width}x{height}")
    print(f"  Steps:  {steps}")
    print(f"  Seed:   {seed}")

    flux = load_model()

    start = time.time()
    image = flux.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
    )
    elapsed = time.time() - start

    image.save(str(filepath))
    print(f"  Done in {elapsed:.1f}s — {filepath}")

    return str(filepath.resolve())


def main():
    parser = argparse.ArgumentParser(description="FLUX Art Generator — Apple Silicon")
    parser.add_argument("prompt", nargs="?", help="Text prompt (omit for interactive)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("  FLUX Art Generator — FLUX.2-klein-4B (Apple Silicon)")
    print("=" * 60)

    if args.prompt:
        path = generate_image(args.prompt, args.width, args.height, args.steps, args.seed, output_filename=args.output)
        print(f"\nImage: {path}")
    else:
        print("\nInteractive mode. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input("Prompt: ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ("quit", "exit", "q"):
                    break
                generate_image(prompt, args.width, args.height, args.steps, args.seed)
                print()
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
