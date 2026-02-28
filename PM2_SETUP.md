# PM2 Setup — Copy-Paste Commands

All commands below are ready to copy-paste. Run from any directory.

**IMPORTANT: pm2 runs background services only. To chat with Beast, run beast.py in a terminal:**
```bash
cd /Users/jonathanrothberg/Agents/obedient_beast && /Users/jonathanrothberg/Agents/.venv/bin/python3 beast.py
```

## First Time: Install pm2

```bash
sudo npm install -g pm2
```

## Nuke Old Config (if starting fresh)

```bash
pm2 delete all
```

## Start All Services

Copy-paste this whole block to register all 4 background services.
Then open a terminal and run beast.py for your interactive CLI (see below).

```bash
# 1) Local LLM server (Qwen3.5-122B — change model name as needed)
pm2 start /Users/jonathanrothberg/Agents/lfm_thinking.py \
    --name lfm-thinking \
    --interpreter /Users/jonathanrothberg/Agents/.venv/bin/python3 \
    --cwd /Users/jonathanrothberg/Agents \
    --max-restarts 3 --restart-delay 10000 \
    -- --model Qwen3.5-122B --server

# 2) Beast HTTP server (receives WhatsApp messages)
pm2 start /Users/jonathanrothberg/Agents/obedient_beast/server.py \
    --name beast-server \
    --interpreter /Users/jonathanrothberg/Agents/.venv/bin/python3 \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast

# 3) WhatsApp bridge
pm2 start /Users/jonathanrothberg/Agents/obedient_beast/whatsapp/bridge.js \
    --name whatsapp-bridge \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast/whatsapp

# 4) Heartbeat (autonomous task processor)
pm2 start /Users/jonathanrothberg/Agents/obedient_beast/heartbeat.py \
    --name beast-heartbeat \
    --interpreter /Users/jonathanrothberg/Agents/.venv/bin/python3 \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast
```

## Save & Enable Auto-Start on Reboot

```bash
pm2 save
pm2 startup    # Run the sudo command it prints
```

## Watching Logs (pm2 runs headless — no individual terminal windows)

pm2 runs everything in the background. To see what's happening, use `pm2 logs`.
Log files are stored at `~/.pm2/logs/`:

```
~/.pm2/logs/lfm-thinking-out.log        # lfm-thinking stdout
~/.pm2/logs/lfm-thinking-error.log      # lfm-thinking stderr
~/.pm2/logs/beast-server-out.log
~/.pm2/logs/beast-server-error.log
~/.pm2/logs/whatsapp-bridge-out.log
~/.pm2/logs/whatsapp-bridge-error.log
~/.pm2/logs/beast-heartbeat-out.log
~/.pm2/logs/beast-heartbeat-error.log
```

Quick commands:

```bash
# Live tail ALL services at once (Ctrl-C to stop watching)
pm2 logs

# Live tail just one service
pm2 logs lfm-thinking
pm2 logs beast-server
pm2 logs whatsapp-bridge
pm2 logs beast-heartbeat

# Show last 30 lines of errors (no live tail)
pm2 logs --err --lines 30 --nostream

# Show last 50 lines of one service (no live tail)
pm2 logs lfm-thinking --lines 50 --nostream
```

Each log line is prefixed with the service name so you can tell them apart.

## Beast CLI — Your Interactive Terminal

beast.py is interactive (you type, it responds), so it can NOT run under pm2.
Run it yourself in a separate terminal. It works alongside WhatsApp — both
talk to the same agent, same memory, same task queue.

```bash
cd /Users/jonathanrothberg/Agents/obedient_beast
/Users/jonathanrothberg/Agents/.venv/bin/python3 beast.py
```

So your typical workflow is:
1. `pm2 restart all` — starts the 4 background services
2. Open a terminal, run `beast.py` — your interactive CLI
3. Now you can chat via CLI AND WhatsApp at the same time

## Day-to-Day Commands

```bash
# Check status of everything
pm2 status

# Restart everything
pm2 restart all

# Restart one service
pm2 restart lfm-thinking
pm2 restart beast-server
pm2 restart whatsapp-bridge
pm2 restart beast-heartbeat

# Stop one service
pm2 stop lfm-thinking

# Switch local model without restarting
curl -X POST http://localhost:8000/v1/models/switch \
    -H "Content-Type: application/json" \
    -d '{"model": "GLM"}'

# See available models
curl http://localhost:8000/v1/models/available
```

## Troubleshooting

**lfm-thinking crash loop (61 restarts)?**
The old config used system `python3` which doesn't have cv2/mlx installed.
Fix: delete and re-register with the venv interpreter (commands above).

**Large models take a long time to load:**
`--max-restarts 3 --restart-delay 10000` prevents pm2 from kill-restarting
during the long model load. If you see restarts climbing, check logs.

**WhatsApp "Try again later":**
WhatsApp rate-limits device linking. Wait 10-15 minutes and retry.

**Check what's actually wrong:**
```bash
pm2 logs --err --lines 30 --nostream
```
