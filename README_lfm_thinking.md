# lfm_thinking.py — Local Model Server

**lfm = local model** (legacy name). This is the macOS/MLX OpenAI-compatible server Beast uses when `LLM_BACKEND=lfm`. Linux twin: `linux_thinking.py` (same `:8000` API).

Human setup, default `LFM_MODEL` (today Qwen3.8-27B-class), and how to start Beast: **[README.md](README.md)**.

```bash
python lfm_thinking.py --model Qwen3.8-27B-mxfp8 --server
# or via Beast on macOS:
cd obedient_beast && ./start.sh cli
```

`--model` is a folder substring under `/Users/jonathanrothberg/MLX_Models/` (or `--model latest`). Interactive picker if you omit it. Hot-swap: `POST /v1/models/switch` or Beast `/model`. Verbose parse logs: `LFM_VERBOSE=1`.

Beast sends tools in OpenAI format plus `chat_template_kwargs`. The server applies family presets from `local_harness.py` and passes native message roles into `apply_chat_template` (flatten only if there is no template). Unknown kwargs are dropped if the tokenizer rejects them.

Eval (no model name in the tests — uses whatever is loaded): `python test_client.py --eval-parse` / `python test_client.py --eval`.

pm2: see [PM2_SETUP.md](PM2_SETUP.md) or `obedient_beast/start.sh pm2` (pm2 does **not** run this brain; it stays in a Terminal window).
