#!/usr/bin/env python3
"""
FLUX Art Generator — Linux (MLX on CUDA or CPU)
===============================================
Same workflow as flux_art.py, but intended for **Linux** with **MLX**
(`pip install mlx[cuda12]`, `mlx[cuda13]`, or `mlx[cpu]` — see MLX docs).
Uses mflux with FLUX.2-klein-4B (pre-quantized 4-bit) and local weights.

On **macOS Apple Silicon**, use `flux_art.py` instead (Metal).

Usage:
    python flux_art_linux.py "a cat astronaut floating in space"
    python flux_art_linux.py --width 1024 --height 768 --seed 42 "sunset"
    python flux_art_linux.py --model ~/FLUX.2-klein-4B-mflux-4bit "a dog"
    export FLUX_ART_MODEL=~/path/to/FLUX.2-klein-4B-mflux-4bit
    python flux_art_linux.py   # interactive mode

    # As a library (optional; agent/Beast does not wire this yet on Linux)
    from flux_art_linux import generate_image
    path = generate_image("a fox in a forest", model_path="~/FLUX.2-klein-4B-mflux-4bit")
"""

import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path(__file__).parent / "generated_art"
# Local mflux 4-bit weights (same layout as Hugging Face "mflux-4bit" bundle).
DEFAULT_MODEL_PATH = Path.home() / "FLUX.2-klein-4B-mflux-4bit"
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 4
DEFAULT_SEED = None

_flux = None
_loaded_model_path: Path | None = None


def _slugify(text: str, max_len: int = 40) -> str:
    import re
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')[:max_len]


def resolve_model_path(explicit: str | Path | None = None) -> Path:
    """CLI --model > env FLUX_ART_MODEL > default under home."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("FLUX_ART_MODEL")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_MODEL_PATH.resolve()


def load_model(model_path: str | Path | None = None) -> object:
    global _flux, _loaded_model_path
    path = resolve_model_path(model_path)
    if _flux is not None and _loaded_model_path == path:
        return _flux

    if not path.is_dir():
        raise FileNotFoundError(
            f"Model folder not found: {path}\n"
            "  Point --model or FLUX_ART_MODEL at your FLUX.2-klein mflux-4bit directory "
            f"(default: {DEFAULT_MODEL_PATH})."
        )

    print(f"Loading FLUX.2-klein from {path} ...")
    start = time.time()

    from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

    _flux = Flux2Klein(model_path=str(path))
    _loaded_model_path = path

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
    model_path: str | Path | None = None,
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

    flux = load_model(model_path)

    start = time.time()
    image = flux.generate_image(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        seed=seed,
    )
    elapsed = time.time() - start

    image.save(str(filepath))
    print(f"  Done in {elapsed:.1f}s — {filepath}")

    return str(filepath.resolve())


def main():
    # CLI entry: prefer flux_art.py on macOS (Metal); this file targets Linux MLX.
    if sys.platform == "darwin":
        print(
            "flux_art_linux.py is for Linux (MLX CUDA/CPU). On macOS use flux_art.py instead.",
            file=sys.stderr,
        )
        sys.exit(2)

    _epilog = f"""\
Local text-to-image on Linux via MLX (CUDA or CPU): load FLUX.2-klein mflux-4bit weights — no cloud image API.

  Install: pip install mflux
           pip install 'mlx[cuda12]'   # or mlx[cuda13] / mlx[cpu] — see https://ml-explore.github.io/mlx/

  Default weights folder: {DEFAULT_MODEL_PATH}
  Override: --model PATH  or  export FLUX_ART_MODEL=PATH

  Output: {OUTPUT_DIR}/  (use --output for a fixed filename)

  Use at least 2 --steps (default {DEFAULT_STEPS}); step count 1 can fail in the scheduler.
"""
    parser = argparse.ArgumentParser(
        description="Draw images with local FLUX.2-klein (mflux + MLX, Linux).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog,
    )
    parser.add_argument("prompt", nargs="?", help="Text prompt (omit for interactive mode)")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Image width in pixels")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Image height in pixels")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Inference steps (>=2 recommended)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: random)")
    parser.add_argument("--output", type=str, default=None, help="Output filename inside generated_art/")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Path to FLUX.2-klein mflux-4bit folder (default: {DEFAULT_MODEL_PATH}, or env FLUX_ART_MODEL)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  FLUX Art Generator — FLUX.2-klein-4B (Linux / MLX)")
    print("=" * 60)

    if args.prompt:
        path = generate_image(
            args.prompt,
            args.width,
            args.height,
            args.steps,
            args.seed,
            output_filename=args.output,
            model_path=args.model,
        )
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
                generate_image(prompt, args.width, args.height, args.steps, args.seed, model_path=args.model)
                print()
            except KeyboardInterrupt:
                break


if __name__ == "__main__":
    main()
