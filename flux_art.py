#!/usr/bin/env python3
"""
FLUX Art Generator — Standalone text-to-image on Apple Silicon
==============================================================
Uses MFLUX (pure MLX FLUX.1-schnell implementation) to generate images
from text prompts. Runs entirely on Apple Silicon Metal GPU — no PyTorch,
no CUDA, no cloud API needed.

Model: FLUX.1-schnell (4-bit quantized, ~5GB download on first run)
       Optimized for 4 inference steps — fast generation.

Usage:
    # Interactive mode — prompts you for text
    python flux_art.py

    # One-shot from command line
    python flux_art.py "a cat astronaut floating in space"

    # With options
    python flux_art.py "sunset over mountains" --width 1024 --height 768 --steps 4 --seed 42

    # As a library (for Beast integration)
    from flux_art import generate_image
    path = generate_image("a fox in a forest", width=512, height=512)

Output: Images saved to ./generated_art/<timestamp>_<prompt_slug>.png

First run downloads the 4-bit quantized model (~5GB). Subsequent runs
load from cache (~15s on M-series Mac Studio).

Requirements: pip install mflux  (already installed in .venv)
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "generated_art"
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_STEPS = 4        # FLUX.1-schnell is optimized for 4 steps
DEFAULT_QUANTIZE = 4     # 4-bit quantization (fast, ~5GB model)
DEFAULT_SEED = None       # None = random

# Global model reference — loaded once, reused across calls
_model = None


def _slugify(text: str, max_len: int = 40) -> str:
    """Turn a prompt into a safe filename slug."""
    import re
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
    return slug[:max_len]


def load_model(quantize: int = DEFAULT_QUANTIZE):
    """
    Load the FLUX.1-schnell model (downloads on first run).
    Returns the Flux1 model instance. Cached globally for reuse.
    """
    global _model
    if _model is not None:
        return _model

    print("Loading FLUX.1-schnell model...")
    print(f"  Quantization: {quantize}-bit")
    print("  (First run downloads ~5GB — subsequent runs load from cache)")
    start = time.time()

    from mflux.models.flux.variants.txt2img.flux import Flux1
    from mflux.models.common.config.model_config import ModelConfig

    _model = Flux1(
        quantize=quantize,
        model_config=ModelConfig.schnell(),
    )

    elapsed = time.time() - start
    print(f"  Model loaded in {elapsed:.1f}s")
    return _model


def generate_image(
    prompt: str,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    seed: int | None = DEFAULT_SEED,
    quantize: int = DEFAULT_QUANTIZE,
    output_dir: str | Path | None = None,
    output_filename: str | None = None,
) -> str:
    """
    Generate an image from a text prompt using FLUX.1-schnell.

    Args:
        prompt:     Text description of the image to generate
        width:      Image width in pixels (default 512)
        height:     Image height in pixels (default 512)
        steps:      Inference steps (default 4, schnell is optimized for 4)
        seed:       Random seed (None = random)
        quantize:   Model quantization bits (4 or 8, default 4)
        output_dir: Where to save (default: ./generated_art/)
        output_filename: Custom filename (default: timestamp_prompt.png)

    Returns:
        Absolute path to the saved PNG image.
    """
    # Resolve output path
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    if seed is None:
        import random
        seed = random.randint(0, 2**32 - 1)

    if output_filename:
        filepath = out / output_filename
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _slugify(prompt)
        filepath = out / f"{ts}_{slug}.png"

    print(f"\nGenerating image:")
    print(f"  Prompt: {prompt}")
    print(f"  Size:   {width}x{height}")
    print(f"  Steps:  {steps}")
    print(f"  Seed:   {seed}")
    print(f"  Output: {filepath}")

    model = load_model(quantize=quantize)

    start = time.time()
    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
    )
    elapsed = time.time() - start

    # Save the image
    image.image.save(str(filepath))
    print(f"\n  Done in {elapsed:.1f}s — saved to {filepath}")

    return str(filepath.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="FLUX Art Generator — text-to-image on Apple Silicon",
        epilog="Examples:\n"
               "  python flux_art.py \"a cat in space\"\n"
               "  python flux_art.py \"sunset\" --width 1024 --height 768\n"
               "  python flux_art.py  (interactive mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prompt", nargs="?", help="Text prompt (omit for interactive mode)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help=f"Image width (default {DEFAULT_WIDTH})")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help=f"Image height (default {DEFAULT_HEIGHT})")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help=f"Inference steps (default {DEFAULT_STEPS})")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--quantize", type=int, default=DEFAULT_QUANTIZE, choices=[4, 8], help="Quantization bits (default 4)")
    parser.add_argument("--output", type=str, default=None, help="Output filename (default: auto)")

    args = parser.parse_args()

    print("=" * 60)
    print("  FLUX Art Generator — Apple Silicon (MLX)")
    print("  Model: FLUX.1-schnell (4-bit quantized)")
    print("=" * 60)

    if args.prompt:
        # One-shot mode
        path = generate_image(
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            steps=args.steps,
            seed=args.seed,
            quantize=args.quantize,
            output_filename=args.output,
        )
        print(f"\nImage: {path}")
    else:
        # Interactive mode
        print("\nInteractive mode — type a prompt, press Enter to generate.")
        print("Type 'quit' to exit.\n")

        while True:
            try:
                prompt = input("Prompt: ").strip()
                if not prompt:
                    continue
                if prompt.lower() in ("quit", "exit", "q"):
                    print("Goodbye!")
                    break

                path = generate_image(
                    prompt=prompt,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    seed=args.seed,
                    quantize=args.quantize,
                )
                print(f"Image: {path}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
