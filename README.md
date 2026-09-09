# Agents — Local LLM + Obedient Beast

**Author:** Jonathan M Rothberg

Two projects in one repo: a local model server (brain), and Obedient Beast (CLI + WhatsApp). **lfm** means local model (legacy name). Canonical `.env` is the **repo root**.

## Four pieces (only one is chat)

| Piece | Window title | Role |
|-------|--------------|------|
| **Brain** | LFM Thinking | Model on `:8000`. Wait until ready. **Do not type chat here.** |
| **Mailbox** | Beast Server | WhatsApp → `:5001` → Beast → brain |
| **Phone link** | WhatsApp Bridge | Links your phone (QR only if asked) |
| **Local client** | Local client — You: | Type here (`You:`). Same agent as WhatsApp |

```
You:        →  beast.py  →  brain :8000
WhatsApp    →  bridge → mailbox :5001 → beast.py → brain :8000
```

Local client does **not** need the mailbox or bridge. WhatsApp does **not** use the `You:` window. Both need the brain.

`local_harness.py` is a library (not a window). `test_client.py` is a raw `:8000` ping with no Beast tools.

## Start / stop

First time: `./obedient_beast/setup.sh`

| Want | Command | Windows |
|------|---------|---------|
| Terminal only | `./obedient_beast/start.sh cli` | brain + `You:` |
| Phone | `./obedient_beast/start.sh phone` | brain + mailbox + WhatsApp |
| Extra `You:` (phone already up) | `./obedient_beast/start.sh you` | local client only |
| Everything | `./obedient_beast/start.sh` | brain + mailbox + WhatsApp + heartbeat + `You:` |
| Stop | `./obedient_beast/start.sh stop` | kills the stack |

Order that works: start → wait until **LFM Thinking** says the API is ready → then type at `You:` or text WhatsApp.

`phone` / `you` force `LFM_URL=http://localhost:8000` and `MCP_ENABLED=false`.

If a restart leaves junk on ports: `./obedient_beast/start.sh stop`, close leftover Terminal tabs if needed, then `./obedient_beast/start.sh` again.

Also: `status` · `pm2` · `lfm` · `server` · `whatsapp` · `heartbeat` · `clear-history`.

## Brain by hand

```bash
# from repo root, with venv on
source .venv/bin/activate
python lfm_thinking.py --model Qwen3.8-27B-mxfp8 --server   # default (LFM_MODEL)
# optional Flash-Next (does not change start.sh default):
# python lfm_thinking.py --model Qwen3.8-Flash-Next-MLX-4bit --server
```

Linux twin: `python linux_thinking.py --model latest --server`. Default folder is `LFM_MODEL` under `~/MLX_Models/` (today `Qwen3.8-27B-mxfp8`). Missing folder → interactive picker.

Mailbox by hand (must use the venv — system Python has no Flask):

```bash
source .venv/bin/activate
cd obedient_beast && python3 server.py
```

## Layout

```
Agents/
├── README.md / CLAUDE.md
├── lfm_thinking.py / linux_thinking.py   ← brain
├── local_harness.py                      ← shared parse / templates
├── test_client.py
├── .env.example
└── obedient_beast/
    ├── beast.py / llm.py / server.py / heartbeat.py
    ├── start.sh / setup.sh
    ├── whatsapp/bridge.js
    └── workspace/SOUL.md                 ← system prompt (+ AGENTS.md, skills/)
```

## Beast (short)

`beast.run()` → LLM with tools → execute tools → loop up to `DEPTH` (cloud 10, local 8, `/depth N`). Local default tool groups: `core,browser,art` (`/tools all` for everything). Switch backends in chat: `/lfm` `/claude` `/openai`.

WhatsApp allowlist (repo-root `.env`): `ALLOWED_NUMBERS`, `ALLOWED_GROUPS`, `RESPOND_TO_OTHERS`. OWNER group commands: `!openbeast` / `!closebeast` / `!listbeast`. Details: [obedient_beast/README.md](obedient_beast/README.md).

Skills: drop `obedient_beast/workspace/skills/<name>/SKILL.md` (`list_skills` / `use_skill`). `/skills` is the MCP catalog, not those files.

## Env (repo-root `.env`)

| Variable | Purpose |
|----------|---------|
| `LLM_BACKEND` | `lfm` / `openai` / `claude` |
| `LFM_URL` | Brain URL (default `http://localhost:8000`, no `/v1`) |
| `LFM_MODEL` | Folder substring for `start.sh` (default `Qwen3.8-27B-mxfp8`) |
| `QWEN_REASONING_EFFORT` | `low` / `medium` / `xhigh` (default `medium`) |
| `BEAST_TOOL_GROUPS` | Local default `core,browser,art` |
| `MCP_ENABLED` | Default `false` if unset |
| `ALLOWED_NUMBERS` / `ALLOWED_GROUPS` / `RESPOND_TO_OTHERS` | WhatsApp |
| `BEAST_PORT` | Mailbox (default `5001`) |

## Troubleshooting

- **WhatsApp silent** → look at **Beast Server** for `[In]` / `[Allow]` / `[Blocked]` / `[Run]` / `[Out]`. Bridge shows `[Blocked]` / `[Sent]`.
- **Mailbox crashed `No module named 'flask'`** → activate `.venv`, then `python3 server.py` (or `./start.sh server`).
- **No `You:`** → `phone` does not open it; run `./obedient_beast/start.sh you`.
- **Brain busy / wrong model still on :8000** → `./start.sh stop`, free leftovers, start again.
- **CLI hangs at startup** → set `MCP_ENABLED=false` (forced by `phone`/`you`).
- **`.env` ignored** → must live at repo root.

Optional art: macOS `flux_art.py` (mflux); Linux Beast art `zimage_art_linux.py`. LLMs: [CLAUDE.md](CLAUDE.md).
