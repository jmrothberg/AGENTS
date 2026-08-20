# lfm_thinking.py — Local Model Server

**lfm = local model** (legacy name). This is the macOS/MLX OpenAI-compatible server Beast uses when `LLM_BACKEND=lfm`.

Human setup, Qwen3.8 default, and how to start Beast: **[README.md](README.md)**.

```bash
python lfm_thinking.py --model Qwen3.8-27B-mxfp8 --server
# or via Beast:
cd obedient_beast && ./start.sh cli
```

Hot-swap: `POST /v1/models/switch` or Beast `/model`. Verbose parse logs: `LFM_VERBOSE=1`.

Linux twin: `linux_thinking.py`. pm2: see [PM2_SETUP.md](PM2_SETUP.md) or `obedient_beast/start.sh pm2`.
