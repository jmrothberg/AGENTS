---
library_name: mlx
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.6-27B
tags:
- mlx
---

# mlx-community/Qwen3.6-27B-mxfp8

This model was converted to MLX format from [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B)
using mlx-vlm version **0.4.4**.
Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.6-27B) for more details on the model.

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model mlx-community/Qwen3.6-27B-mxfp8 --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
