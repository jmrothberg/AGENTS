# PM2 Setup — Copy-Paste Commands

All commands below are ready to copy-paste. Run from any directory.

**IMPORTANT: pm2 runs background services only. Two processes must stay as terminal windows:**
- **beast.py** — interactive CLI (you type, it responds)
- **lfm_thinking.py** — interactive model picker (choose your model, watch it load)

`start.sh` handles both: it starts the 3 pm2 services and opens terminal windows for lfm_thinking + CLI.
```bash
cd /Users/jonathanrothberg/Agents/obedient_beast && ./start.sh
```
Or run beast.py directly for CLI only:
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

`start.sh` handles everything — run this instead of copy-pasting:
```bash
cd /Users/jonathanrothberg/Agents/obedient_beast && ./start.sh
```

Or register the 3 background services manually with pm2:

```bash
# 1) Beast HTTP server (receives WhatsApp messages)
pm2 start /Users/jonathanrothberg/Agents/obedient_beast/server.py \
    --name beast-server \
    --interpreter /Users/jonathanrothberg/Agents/.venv/bin/python3 \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast

# 2) WhatsApp bridge
pm2 start /Users/jonathanrothberg/Agents/obedient_beast/whatsapp/bridge.js \
    --name whatsapp-bridge \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast/whatsapp

# 3) Heartbeat (autonomous task processor)
pm2 start /Users/jonathanrothberg/Agents/obedient_beast/heartbeat.py \
    --name beast-heartbeat \
    --interpreter /Users/jonathanrothberg/Agents/.venv/bin/python3 \
    --cwd /Users/jonathanrothberg/Agents/obedient_beast
```

**lfm_thinking.py is NOT managed by pm2** — run it in a terminal so you can pick the model interactively:
```bash
cd /Users/jonathanrothberg/Agents && /Users/jonathanrothberg/Agents/.venv/bin/python3 lfm_thinking.py
# or via start.sh:
./start.sh lfm
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
~/.pm2/logs/beast-server-out.log
~/.pm2/logs/beast-server-error.log
~/.pm2/logs/whatsapp-bridge-out.log
~/.pm2/logs/whatsapp-bridge-error.log
~/.pm2/logs/beast-heartbeat-out.log
~/.pm2/logs/beast-heartbeat-error.log
```

lfm_thinking.py logs are visible directly in its terminal window (not pm2).

Quick commands:

```bash
# Live tail ALL pm2 services at once (Ctrl-C to stop watching)
pm2 logs

# Live tail just one service
pm2 logs beast-server
pm2 logs whatsapp-bridge
pm2 logs beast-heartbeat

# Show last 30 lines of errors (no live tail)
pm2 logs --err --lines 30 --nostream

# Show last 50 lines of one service (no live tail)
pm2 logs beast-server --lines 50 --nostream
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
1. `./start.sh` — starts the 3 pm2 background services + opens terminal windows for lfm_thinking and beast CLI
2. In the lfm_thinking terminal, pick your model and wait for it to load
3. Chat via the Beast CLI terminal AND WhatsApp at the same time

## Day-to-Day Commands

```bash
# Check status of everything (pm2 + terminal processes)
./start.sh status

# pm2 status only
pm2 status

# Restart all pm2 services
pm2 restart all

# Restart one pm2 service
pm2 restart beast-server
pm2 restart whatsapp-bridge
pm2 restart beast-heartbeat

# Stop one pm2 service
pm2 stop beast-server

# Switch local model without restarting
curl -X POST http://localhost:8000/v1/models/switch \
    -H "Content-Type: application/json" \
    -d '{"model": "GLM"}'

# See available models
curl http://localhost:8000/v1/models/available
```

## Troubleshooting

**WhatsApp "Try again later":**
WhatsApp rate-limits device linking. Wait 10-15 minutes and retry.

**Check what's actually wrong:**
```bash
pm2 logs --err --lines 30 --nostream
```
