#!/usr/bin/env python3
"""
Z-Image-Turbo text-to-image for Linux (and non-macOS) — PyTorch + diffusers.

Vendored from Agent_learning assets.py pattern; used by Obedient Beast
generate_art on non-darwin. macOS art stays on flux_art.py (MLX + mflux).

Install: see README and requirements-zimage-linux.txt (torch CUDA nightly +
diffusers from git until ZImagePipeline is on PyPI).

Weights: DIFFUSION_MODELS_DIR, ~/Models_Diffusers/Z-Image-Turbo/, or HF
Tongyi-MAI/Z-Image-Turbo. Override with env ZIMAGE_ART_MODEL (path to dir or HF id).
"""

from __future__ import annotations

import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).parent / "generated_art"
_HF_FALLBACK_MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
_NATIVE_SIZE = 768
_DEFAULT_STEPS = 9  # turbo: 8 DiT forwards + scheduler tail
_GUIDANCE = 0.0


def _slugify(text: str, max_len: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:max_len]


def _default_model_search_dirs() -> list[str]:
    home = os.path.expanduser("~")
    return [
        os.path.join(home, "Models_Diffusers"),
        os.path.join(home, "Diffusion_Models"),
        "/home/jonathan/Models_Diffusers",
        "./models_diffusers",
    ]


_MODEL_SEARCH_DIRS = _default_model_search_dirs()


def _resolve_zimage_path(explicit: str | Path | None = None) -> str:
    """Directory path or HuggingFace model id for from_pretrained."""
    if explicit is not None:
        p = Path(explicit).expanduser()
        if p.is_dir():
            return str(p.resolve())
        return str(p)  # HF id string

    env_z = (os.environ.get("ZIMAGE_ART_MODEL") or "").strip()
    if env_z:
        p = Path(env_z).expanduser()
        if p.is_dir():
            return str(p.resolve())
        return env_z

    env_dir = (os.environ.get("DIFFUSION_MODELS_DIR") or "").strip()
    candidates: list[str] = []
    if env_dir:
        candidates.extend([
            os.path.join(env_dir, "Z-Image-Turbo"),
            os.path.join(env_dir, "Tongyi-MAI_Z-Image-Turbo"),
        ])
    for base in _MODEL_SEARCH_DIRS:
        candidates.extend([
            os.path.join(base, "Z-Image-Turbo"),
            os.path.join(base, "Tongyi-MAI_Z-Image-Turbo"),
        ])
    for c in candidates:
        if os.path.isdir(c):
            return c
    return _HF_FALLBACK_MODEL_ID


class _ZImageTurboGenerator:
    """Lazy ZImagePipeline; generate at NATIVE_SIZE then caller resizes."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._pipeline: Any = None
        self._device: str | None = None

    def _lazy_init(self) -> bool:
        if self._pipeline is not None:
            return True
        try:
            import torch
            from diffusers import ZImagePipeline
        except Exception:
            return False

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.bfloat16
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = "mps"
            dtype = torch.float16
        else:
            return False

        try:
            self._pipeline = ZImagePipeline.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            self._pipeline.to(device)
            self._device = device
            return True
        except Exception:
            self._pipeline = None
            self._device = None
            return False

    def generate_native(
        self, prompt: str, seed: int, num_inference_steps: int = _DEFAULT_STEPS
    ) -> Any | None:
        """Returns PIL Image or None."""
        if not self._lazy_init():
            return None
        try:
            import torch
            gen = torch.Generator(self._device or "cpu").manual_seed(int(seed) & 0x7FFFFFFF)
            out = self._pipeline(
                prompt=prompt,
                height=_NATIVE_SIZE,
                width=_NATIVE_SIZE,
                num_inference_steps=num_inference_steps,
                guidance_scale=_GUIDANCE,
                generator=gen,
            )
            return out.images[0]
        except Exception:
            return None


_gen_singleton: _ZImageTurboGenerator | None = None
_singleton_path: str | None = None


def _get_generator(model_path: str) -> _ZImageTurboGenerator:
    global _gen_singleton, _singleton_path
    if _gen_singleton is None or _singleton_path != model_path:
        _gen_singleton = _ZImageTurboGenerator(model_path)
        _singleton_path = model_path
    return _gen_singleton


def generate_image(
    prompt: str,
    width: int = 1024,
    height: int = 1024,
    steps: int = _DEFAULT_STEPS,
    seed: int | None = None,
    output_dir: str | Path | None = None,
    output_filename: str | None = None,
    model_path: str | Path | None = None,
) -> str:
    """
    Same signature as flux_art.generate_image for Beast compatibility.
    Renders at 768x768 then resizes to width x height (PIL LANCZOS).
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    if seed is None:
        seed = random.randint(0, 2**31 - 1)

    if output_filename:
        filepath = out / output_filename
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = out / f"{ts}_{_slugify(prompt)}.png"

    resolved = _resolve_zimage_path(
        str(model_path) if model_path is not None else None
    )

    print(f"\nGenerating image (Z-Image-Turbo):")
    print(f"  Prompt: {prompt}")
    print(f"  Target: {width}x{height} (native {_NATIVE_SIZE}, then resize)")
    print(f"  Steps:  {steps}")
    print(f"  Seed:   {seed}")
    print(f"  Model:  {resolved}")

    gen = _get_generator(resolved)
    t0 = time.time()
    pil = gen.generate_native(prompt, seed=seed, num_inference_steps=steps)
    if pil is None:
        raise RuntimeError(
            "Z-Image-Turbo unavailable: need CUDA or MPS, diffusers with "
            "ZImagePipeline, and valid weights. See README / requirements-zimage-linux.txt."
        )

    try:
        from PIL import Image
        if (pil.width, pil.height) != (width, height):
            pil = pil.resize((int(width), int(height)), Image.LANCZOS)
        pil.save(str(filepath), format="PNG")
    except Exception as e:
        raise RuntimeError(f"Failed to save image: {e}") from e

    print(f"  Done in {time.time() - t0:.1f}s — {filepath}")
    return str(filepath.resolve())
