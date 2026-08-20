# PM2 — background services only

Setup and day-to-day start: **[README.md](README.md)** and `obedient_beast/start.sh`.

pm2 runs **server, WhatsApp bridge, heartbeat**. The local model server and Beast CLI stay in terminal windows (`./start.sh pm2` opens those for you).

```bash
cd obedient_beast && ./start.sh pm2
pm2 status
pm2 logs beast-server
./start.sh stop
```

First-time: `sudo npm install -g pm2` then `./start.sh pm2`.
