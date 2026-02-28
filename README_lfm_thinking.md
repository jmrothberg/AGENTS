# lfm_thinking.py — Local Model Inference Server

Written by Jonathan M Rothberg

Dynamically scans model directories and serves any compatible model locally.
- **macOS**: Uses MLX (`mlx-lm` for text, `mlx-vlm` for vision)
- **Linux**: Falls back to `transformers` / PyTorch

Model type (text vs vision) is auto-detected from `config.json`.

> This is the **Local brain** for [Obedient Beast](README.md). Beast connects to this server when you use `/lfm` mode. See [README.md](README.md) for the full project overview.

## Usage

```bash
# Interactive model selection
python lfm_thinking.py

# Serve the most recently modified model
python lfm_thinking.py --model latest --server

# Serve a specific model by substring match
python lfm_thinking.py --model Qwen3 --server

# Custom port
python lfm_thinking.py --model latest --server --port 9000

# List available models
python lfm_thinking.py --list
```

## PM2 — Run as a Managed Background Service

Copy/paste to start:

```bash
pm2 start /Users/jonathanrothberg/Agents/lfm_thinking.py \
  --name lfm-thinking --interpreter python3 \
  --max-restarts 3 --restart-delay 10000 \
  -- --model Qwen3.5-122B --server
```

This serves the **Qwen3.5-122B-A10B-MLX-9bit** model from `/Users/jonathanrothberg/MLX_Models/`.

> **Why `--max-restarts 3` and `--restart-delay 10000`?** Large models (e.g. 122B params)
> can take a long time to load into memory. Without these flags, pm2 will kill the process
> during loading and restart it in an infinite crash loop.

### PM2 Management Commands

```bash
pm2 status              # check running processes
pm2 logs lfm-thinking   # view stdout/stderr
pm2 restart lfm-thinking
pm2 stop lfm-thinking
pm2 delete lfm-thinking # remove from pm2 entirely before re-adding
pm2 save                # persist across reboots (pair with: pm2 startup)
```

## Hot-Swap Models (no restart needed)

While the server is running, switch models via the API or from Beast:

```bash
# From Beast CLI or WhatsApp
/model Qwen3         # switch by name (fuzzy match)
/model latest        # switch to most recently downloaded
/model               # list all available models

# Via curl
curl -X POST http://localhost:8000/v1/models/switch \
  -H "Content-Type: application/json" \
  -d '{"model": "GLM"}'

# List all models
curl http://localhost:8000/v1/models/available
```
