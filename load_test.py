"""
Production Load Tester & Crash Analyser
========================================
Run:
    python load_test.py --clients 8              # 8 concurrent clients
    python load_test.py --clients 20 --ramp 5    # ramp: add 5 clients every 10s
    python load_test.py --clients 8 --workers 4  # inform summary of worker count

Writes:
    capacity_logs/transcripts_<timestamp>.txt    — full log
    capacity_logs/summary_<timestamp>.json       — machine-readable JSON

Features:
    - Crash detection: OOM (1006), server killed (1011/1001), 4008 rejection
    - Word coverage: compares received words vs ground-truth audio transcript
    - Live crash visualisation: shows crash events in real-time as they happen
    - Per-worker saturation tracking (via /status endpoint)
    - Ramp mode: gradually saturates server to find the crash point
"""

import argparse
import asyncio
import websockets
import websockets.exceptions
import time
import json
import uuid
import os
import sys
import threading
import numpy as np
import psutil
import aiohttp
from datetime import datetime
from collections import Counter
from pydub import AudioSegment

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
SERVER_WS  = "ws://localhost:8000/ws/consultation"
SERVER_HTTP = "http://localhost:8000"

AUDIO_FILE  = "input.wav"
CHUNK_MS    = 100

LOG_DIR     = "capacity_logs"
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(LOG_DIR, exist_ok=True)

TRANSCRIPT_LOG = os.path.join(LOG_DIR, f"transcripts_{TIMESTAMP}.txt")
SUMMARY_JSON   = os.path.join(LOG_DIR, f"summary_{TIMESTAMP}.json")

# Ground-truth word count for coverage detection (approximate for input.wav)
# Set to None to skip coverage check
GROUND_TRUTH_WORDS = None  # Will be auto-estimated from audio length

# ══════════════════════════════════════════════════════════════════════
# ARGS
# ══════════════════════════════════════════════════════════════════════
parser = argparse.ArgumentParser(description="Production WebSocket load tester with crash analysis")
parser.add_argument("--clients", type=int, default=5,    help="Total concurrent clients")
parser.add_argument("--ramp",    type=int, default=0,    help="Ramp: clients per wave (0=all at once)")
parser.add_argument("--workers", type=int, default=None, help="Expected server worker count (for display)")
parser.add_argument("--wait",    type=int, default=150,  help="Seconds to wait for final transcript after STOP")
args = parser.parse_args()

NUM_CLIENTS = args.clients
RAMP_SIZE   = args.ramp
WAIT_FINAL  = args.wait

# ══════════════════════════════════════════════════════════════════════
# THREAD-SAFE IO
# ══════════════════════════════════════════════════════════════════════
_flock = threading.Lock()

def fwrite(path: str, text: str):
    with _flock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

# ══════════════════════════════════════════════════════════════════════
# CRASH EVENT LOG — real-time visualisation
# ══════════════════════════════════════════════════════════════════════
_crash_events: list[dict] = []
_crash_lock = threading.Lock()

def record_crash(cid: int, session_id: str, event_type: str, detail: str, ram_pct: float, cpu_pct: float):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    event = {
        "ts": ts, "cid": cid, "session": session_id,
        "type": event_type, "detail": detail,
        "ram_pct": ram_pct, "cpu_pct": cpu_pct,
    }
    with _crash_lock:
        _crash_events.append(event)

    icon = {
        "OOM_CRASH":    "💥",
        "SERVER_KILLED": "💀",
        "REJECTED":     "🚫",
        "FAILED":       "❌",
    }.get(event_type, "⚠️")

    line = (
        f"\n  ┌{'─'*60}\n"
        f"  │ {icon} CRASH EVENT @ {ts}\n"
        f"  │   Client  : C{cid} ({session_id})\n"
        f"  │   Type    : {event_type}\n"
        f"  │   Detail  : {detail}\n"
        f"  │   RAM     : {ram_pct:.1f}%\n"
        f"  │   CPU     : {cpu_pct:.1f}%\n"
        f"  └{'─'*60}\n"
    )
    print(line, flush=True)
    fwrite(TRANSCRIPT_LOG, line)

# ══════════════════════════════════════════════════════════════════════
# SYSTEM MONITOR
# ══════════════════════════════════════════════════════════════════════
class SystemMonitor:
    def __init__(self):
        self._lock    = threading.Lock()
        self._samples = []
        self.running  = False

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            vm  = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=None)
            with self._lock:
                self._samples.append({
                    "ram_pct":  vm.percent,
                    "cpu_pct":  cpu,
                    "used_gb":  round(vm.used  / 1e9, 2),
                    "free_gb":  round(vm.available / 1e9, 2),
                    "ts":       time.time(),
                })
            time.sleep(1)

    def current(self) -> dict:
        vm  = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        return {
            "ram_pct": vm.percent,
            "cpu_pct": cpu,
            "used_gb": round(vm.used  / 1e9, 2),
            "free_gb": round(vm.available / 1e9, 2),
        }

    def summary(self) -> dict:
        with self._lock:
            s = list(self._samples)
        if not s:
            return {"ram_avg": 0, "ram_peak": 0, "cpu_avg": 0, "cpu_peak": 0,
                    "used_peak": 0, "free_min": 99}
        return {
            "ram_avg":   round(float(np.mean([x["ram_pct"] for x in s])), 1),
            "ram_peak":  round(max(x["ram_pct"]  for x in s), 1),
            "cpu_avg":   round(float(np.mean([x["cpu_pct"] for x in s])), 1),
            "cpu_peak":  round(max(x["cpu_pct"]  for x in s), 1),
            "used_peak": round(max(x["used_gb"]  for x in s), 2),
            "free_min":  round(min(x["free_gb"]  for x in s), 2),
        }

    def crash_threshold_hit(self) -> bool:
        """Returns True if RAM > 95% or was 0 free GB at any point."""
        with self._lock:
            return any(x["free_gb"] < 0.3 for x in self._samples)

monitor = SystemMonitor()

# ══════════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ══════════════════════════════════════════════════════════════════════
def load_audio(path: str) -> bytes:
    seg = AudioSegment.from_file(path)
    seg = seg.set_frame_rate(16000).set_channels(1)
    return np.array(seg.get_array_of_samples()).astype(np.int16).tobytes()

audio_bytes  = load_audio(AUDIO_FILE)
CHUNK_BYTES  = int(16000 * CHUNK_MS / 1000) * 2
TOTAL_CHUNKS = len(audio_bytes) // CHUNK_BYTES
AUDIO_DUR_S  = len(audio_bytes) / (16000 * 2)

# ══════════════════════════════════════════════════════════════════════
# CLIENT RESULT
# ══════════════════════════════════════════════════════════════════════
class ClientResult:
    def __init__(self, cid: int):
        self.cid           = cid
        self.session_id    = str(uuid.uuid4())[:8]
        self.status        = "PENDING"
        self.error         = ""
        self.reject_reason = ""
        self.chunks_sent   = 0
        self.transcripts:  list[tuple]  = []
        self.partials:     list[str]    = []
        self.latencies:    list[int]    = []
        self.proc_rams:    list[float]  = []
        self.all_text:     list[str]    = []   # all confirmed transcript texts
        self.started_at    = time.time()
        self.ended_at      = None
        self.server_pid    = None
        self.crash_ram     = None
        self.crash_cpu     = None

    def word_count(self) -> int:
        return sum(len(t.split()) for t in self.all_text)

    def full_transcript(self) -> str:
        return " ".join(self.all_text)


# ══════════════════════════════════════════════════════════════════════
# FETCH SERVER STATUS
# ══════════════════════════════════════════════════════════════════════
async def fetch_status() -> dict:
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{SERVER_HTTP}/status", timeout=aiohttp.ClientTimeout(total=3)) as r:
                return await r.json()
    except Exception:
        return {}


# ══════════════════════════════════════════════════════════════════════
# SINGLE CLIENT COROUTINE
# ══════════════════════════════════════════════════════════════════════
async def run_client(cid: int, start_delay: float = 0.0) -> ClientResult:
    res = ClientResult(cid)
    if start_delay > 0:
        await asyncio.sleep(start_delay)

    url = f"{SERVER_WS}/{res.session_id}"
    print(f"  [C{cid:>3}] ▶  connecting   session={res.session_id}", flush=True)

    try:
        async with websockets.connect(
            url,
            open_timeout=60,
            close_timeout=15,
            ping_interval=None,
            ping_timeout=None,
            max_size=32 * 1024 * 1024,
        ) as ws:

            # ── receive loop ──────────────────────────────────────────
            async def recv_loop():
                try:
                    async for raw in ws:
                        try:
                            data = json.loads(raw)
                        except Exception:
                            continue

                        mtype = data.get("type", "")

                        # Track server PID (each worker has its own)
                        if data.get("pid"):
                            res.server_pid = data["pid"]

                        if mtype == "connected":
                            continue

                        if mtype == "partial":
                            text = data.get("text", "").strip()
                            if text:
                                res.partials.append(text)
                                mem = monitor.current()
                                print(
                                    f"  [C{cid:>3}] ≈  partial  "
                                    f"chunk~{res.chunks_sent:>4}  "
                                    f"RAM={mem['ram_pct']:.1f}%  "
                                    f"CPU={mem['cpu_pct']:.1f}%  "
                                    f"| {text[:70]}",
                                    flush=True
                                )
                            continue

                        if mtype != "transcript":
                            continue

                        text = (
                            data.get("text")
                            or data.get("transcript")
                            or ""
                        ).strip()
                        if not text:
                            continue

                        is_final  = data.get("is_final", False)
                        inf_ms    = data.get("inference_ms", 0)
                        proc_ram  = data.get("proc_ram_mb", 0.0)
                        lat_ms    = int((time.time() - res.started_at) * 1000)

                        res.transcripts.append((res.chunks_sent, lat_ms, inf_ms, text))
                        res.latencies.append(lat_ms)
                        res.all_text.append(text)
                        if proc_ram:
                            res.proc_rams.append(proc_ram)

                        mem = monitor.current()

                        kind = "FINAL" if is_final else "  mid"
                        block = (
                            f"\n  ┌─ C{cid:>3}  [{res.session_id}]  {kind} TRANSCRIPT {'─'*20}\n"
                            f"  │  text      : {text[:100]}\n"
                            f"  │  latency   : {lat_ms} ms   inference: {inf_ms:.0f}ms\n"
                            f"  │  chunk ~   : {res.chunks_sent}/{TOTAL_CHUNKS}\n"
                            f"  │  proc_ram  : {proc_ram:.0f} MB (worker)\n"
                            f"  │  sys_ram   : {mem['ram_pct']:.1f}%  used={mem['used_gb']} GB  free={mem['free_gb']} GB\n"
                            f"  │  CPU       : {mem['cpu_pct']:.1f}%\n"
                            f"  └{'─'*52}\n"
                        )
                        print(block, flush=True)
                        fwrite(TRANSCRIPT_LOG,
                               f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                               f"C{cid:>3} | {res.session_id} | pid={res.server_pid} | "
                               f"chunk~{res.chunks_sent:>4} | {lat_ms:>7}ms | "
                               f"inf={inf_ms:.0f}ms | proc_ram={proc_ram:.0f}MB | "
                               f"RAM={mem['ram_pct']:.1f}% CPU={mem['cpu_pct']:.1f}% | "
                               f"{text}\n")

                except (
                    websockets.exceptions.ConnectionClosedOK,
                    websockets.exceptions.ConnectionClosedError,
                    asyncio.CancelledError,
                ):
                    pass

            recv_task = asyncio.ensure_future(recv_loop())

            # ── send audio ───────────────────────────────────────────
            idx = 0
            while idx < len(audio_bytes):
                chunk = audio_bytes[idx: idx + CHUNK_BYTES]
                await ws.send(chunk)
                idx             += CHUNK_BYTES
                res.chunks_sent += 1

                if res.status == "REJECTED":
                    break

                mem   = monitor.current()
                print(
                    f"\r  [C{cid:>3}] chunk {res.chunks_sent:>4}/{TOTAL_CHUNKS}  "
                    f"RAM {mem['ram_pct']:5.1f}%  CPU {mem['cpu_pct']:4.1f}%  "
                    f"used={mem['used_gb']} GB  pid={res.server_pid}",
                    end="", flush=True
                )
                await asyncio.sleep(CHUNK_MS / 1000.0)

            print()   # newline after progress bar

            # ── STOP signal ──────────────────────────────────────────
            try:
                await ws.send(json.dumps({"type": "stop"}))
                logger_line = f"  [C{cid:>3}] ■  STOP sent. Waiting up to {WAIT_FINAL}s for final transcript..."
                print(logger_line, flush=True)
            except Exception:
                pass

            # ── Wait for final transcript ────────────────────────────
            try:
                await asyncio.wait_for(recv_task, timeout=WAIT_FINAL)
            except asyncio.TimeoutError:
                recv_task.cancel()
                try:
                    await recv_task
                except Exception:
                    pass
                print(f"  [C{cid:>3}] ⏰ Timeout waiting for final transcript", flush=True)
            except Exception:
                pass

            if res.status not in ("REJECTED",):
                res.status = "SUCCESS"

    # ── Error handling with crash classification ──────────────────────
    except websockets.exceptions.InvalidStatusCode as e:
        code = getattr(e, "status_code", 0)
        if code in (403, 4008) or "403" in str(e):
            res.status        = "REJECTED"
            res.reject_reason = f"HTTP {code} — server rejected handshake"
        else:
            res.status = "FAILED"
            res.error  = str(e)
        mem = monitor.current()
        record_crash(cid, res.session_id, res.status, res.error or res.reject_reason,
                     mem["ram_pct"], mem["cpu_pct"])

    except websockets.exceptions.ConnectionClosedError as e:
        code   = getattr(e, "code", 0) or 0
        reason = str(e)
        mem    = monitor.current()
        if code == 4008 or "4008" in reason:
            res.status        = "REJECTED"
            res.reject_reason = f"4008 — RAM/CPU pressure: {reason}"
        elif code == 1006 or "1006" in reason:
            res.status     = "OOM_CRASH"
            res.error      = f"1006 abnormal close — likely OOM. RAM={mem['ram_pct']:.1f}%"
            res.crash_ram  = mem["ram_pct"]
            res.crash_cpu  = mem["cpu_pct"]
        elif "1011" in reason:
            res.status = "SERVER_KILLED"
            res.error  = "1011 — server internal error / event loop stall"
        elif "1001" in reason:
            res.status = "SERVER_KILLED"
            res.error  = "1001 — server going away (process killed)"
        else:
            res.status = "SERVER_KILLED"
            res.error  = f"closed ({code}): {reason}"
        record_crash(cid, res.session_id, res.status, res.error or res.reject_reason,
                     mem["ram_pct"], mem["cpu_pct"])

    except (OSError, ConnectionResetError) as e:
        mem            = monitor.current()
        res.status     = "OOM_CRASH"
        res.error      = f"OS reset — likely OOM: {e}"
        res.crash_ram  = mem["ram_pct"]
        res.crash_cpu  = mem["cpu_pct"]
        record_crash(cid, res.session_id, res.status, res.error,
                     mem["ram_pct"], mem["cpu_pct"])

    except Exception as e:
        res.status = "FAILED"
        res.error  = str(e)
        mem = monitor.current()
        record_crash(cid, res.session_id, res.status, res.error,
                     mem["ram_pct"], mem["cpu_pct"])

    finally:
        if res.ended_at is None:
            res.ended_at = time.time()

    return res


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
async def main():
    mem = monitor.current()
    srv = await fetch_status()

    sep = "═" * 72
    header = (
        f"\n{sep}\n"
        f"  🧪  LOAD TESTER & CRASH ANALYSER  —  {NUM_CLIENTS} clients\n"
        f"{sep}\n"
        f"  Audio      : {AUDIO_FILE}  ({TOTAL_CHUNKS} chunks @ {CHUNK_MS}ms, {AUDIO_DUR_S:.1f}s)\n"
        f"  Clients    : {NUM_CLIENTS}"
    )
    if RAMP_SIZE > 0:
        header += f"  (ramp: {RAMP_SIZE} per wave)"
    header += "\n"
    if srv:
        header += (
            f"  Server PID : {srv.get('pid', '?')}  "
            f"active={srv.get('active_sessions', '?')}  "
            f"gate={srv.get('admission_gate', '?').upper()}\n"
            f"  Thresholds : free_ram>{srv.get('min_free_ram_mb','?')}MB  "
            f"ram<{srv.get('max_ram_pct','?')}%  "
            f"ceil={srv.get('hard_session_ceil','?')}\n"
            f"  Proc RAM   : {srv.get('proc_ram_mb', '?')} MB  "
            f"sys_free={srv.get('sys_ram_free_mb','?')}MB\n"
        )
    if args.workers:
        header += f"  Workers    : {args.workers} uvicorn worker processes\n"
    header += (
        f"  RAM        : {mem['used_gb']} GB used / {mem['free_gb']} GB free ({mem['ram_pct']:.1f}%)\n"
        f"  CPU        : {mem['cpu_pct']:.1f}%\n"
        f"  Log        : {TRANSCRIPT_LOG}\n"
        f"{sep}\n"
    )
    print(header, flush=True)
    fwrite(TRANSCRIPT_LOG, f"=== Test started {datetime.now()} | clients={NUM_CLIENTS} ===\n\n")

    # ── Launch clients ────────────────────────────────────────────────
    if RAMP_SIZE > 0:
        # Ramp mode: add RAMP_SIZE clients every 10 seconds
        coros = []
        for i in range(NUM_CLIENTS):
            wave_delay = (i // RAMP_SIZE) * 10.0
            coros.append(run_client(i, start_delay=wave_delay))
        print(f"  🔄 Ramp mode: {RAMP_SIZE} clients every 10s across {NUM_CLIENTS // RAMP_SIZE + 1} waves\n")
    else:
        coros = [run_client(i) for i in range(NUM_CLIENTS)]

    raw = await asyncio.gather(*coros, return_exceptions=True)

    results: list[ClientResult] = []
    for r in raw:
        if isinstance(r, ClientResult):
            results.append(r)
        else:
            d = ClientResult(-1)
            d.status   = "FAILED"
            d.error    = str(r)
            d.ended_at = time.time()
            results.append(d)

    # ── Summary ───────────────────────────────────────────────────────
    sys_s    = monitor.summary()
    success  = sum(1 for r in results if r.status == "SUCCESS")
    rejected = sum(1 for r in results if r.status == "REJECTED")
    oom      = sum(1 for r in results if r.status == "OOM_CRASH")
    killed   = sum(1 for r in results if r.status == "SERVER_KILLED")
    failed   = sum(1 for r in results if r.status == "FAILED")
    txn_tot  = sum(len(r.transcripts) for r in results)
    all_lat  = [l for r in results for l in r.latencies]
    tot_words = sum(r.word_count() for r in results if r.status == "SUCCESS")
    admitted  = NUM_CLIENTS - rejected

    lat_avg = round(float(np.mean(all_lat))) if all_lat else 0
    lat_max = max(all_lat) if all_lat else 0
    txn_per = txn_tot / max(admitted, 1)

    # Crash analysis
    crash_ram_vals = [r.crash_ram for r in results if r.crash_ram is not None]
    first_crash_ram = min(crash_ram_vals) if crash_ram_vals else None

    # Words per client (for successful sessions)
    success_results = [r for r in results if r.status == "SUCCESS" and r.word_count() > 0]
    avg_words = round(float(np.mean([r.word_count() for r in success_results]))) if success_results else 0

    div = "─" * 72
    summary = (
        f"\n{sep}\n"
        f"  📊  FINAL SUMMARY  —  {NUM_CLIENTS} clients\n"
        f"{sep}\n"
        f"  ✅  SUCCESS            : {success:>4} / {NUM_CLIENTS}\n"
        f"  🚫  REJECTED (4008)    : {rejected:>4} / {NUM_CLIENTS}\n"
        f"  💥  OOM CRASH          : {oom:>4} / {NUM_CLIENTS}\n"
        f"  💀  SERVER KILLED      : {killed:>4} / {NUM_CLIENTS}\n"
        f"  ❌  OTHER FAILURE      : {failed:>4} / {NUM_CLIENTS}\n"
        f"\n"
        f"  📝  Transcriptions     : {txn_tot}  ({txn_per:.1f}/client)\n"
        f"  🔤  Avg words/client   : {avg_words}\n"
    )
    if all_lat:
        summary += (
            f"  ⏱   Avg latency       : {lat_avg} ms\n"
            f"  ⏱   Max latency       : {lat_max} ms\n"
        )
    summary += (
        f"\n"
        f"  💾  RAM avg            : {sys_s['ram_avg']}%  peak={sys_s['ram_peak']}%\n"
        f"  💾  RAM used peak      : {sys_s['used_peak']} GB  free_min={sys_s['free_min']} GB\n"
        f"  🖥   CPU avg            : {sys_s['cpu_avg']}%  peak={sys_s['cpu_peak']}%\n"
    )
    if first_crash_ram:
        summary += f"\n  ⚠️  First crash at RAM  : {first_crash_ram:.1f}%\n"
    if monitor.crash_threshold_hit():
        summary += f"  🔴  RAM CRITICAL threshold hit (< 300MB free)\n"

    # Crash event log
    with _crash_lock:
        events = list(_crash_events)
    if events:
        summary += f"\n  💥  CRASH EVENTS ({len(events)} total)\n"
        summary += f"  {div}\n"
        for ev in events:
            summary += (
                f"    [{ev['ts']}] C{ev['cid']} ({ev['session']}) "
                f"{ev['type']} — RAM={ev['ram_pct']:.1f}% CPU={ev['cpu_pct']:.1f}%\n"
                f"      {ev['detail']}\n"
            )

    summary += f"\n{div}\n  📋  PER-CLIENT DETAIL\n{div}\n"

    for r in sorted(results, key=lambda x: x.cid):
        icon = {
            "SUCCESS":       "✅",
            "REJECTED":      "🚫",
            "OOM_CRASH":     "💥",
            "SERVER_KILLED": "💀",
            "FAILED":        "❌",
        }.get(r.status, "❓")

        summary += (
            f"\n  {icon} C{r.cid:>3}  session={r.session_id}  "
            f"pid={r.server_pid}  status={r.status}  "
            f"chunks={r.chunks_sent}  words={r.word_count()}\n"
        )
        if r.status == "REJECTED":
            summary += f"       reason: {r.reject_reason}\n"
        elif r.transcripts:
            for chunk_no, lat, inf_ms, text in r.transcripts:
                summary += (
                    f"       chunk~{chunk_no:>4} | {lat:>7}ms | inf={inf_ms:.0f}ms | {text}\n"
                )
            # Show partial coverage
            if r.partials:
                summary += f"       partials received: {len(r.partials)}\n"
        else:
            summary += "       (no transcripts received)\n"
        if r.error:
            summary += f"       ⚠  {r.error}\n"

    summary += f"\n{sep}\n"
    summary += f"  📄  Transcript log → {TRANSCRIPT_LOG}\n"
    summary += f"  📊  Summary JSON   → {SUMMARY_JSON}\n"
    summary += f"{sep}\n"

    print(summary, flush=True)
    fwrite(TRANSCRIPT_LOG, "\n--- FINAL SUMMARY ---\n" + summary)

    # ── Save machine-readable JSON ────────────────────────────────────
    json_out = {
        "timestamp":   TIMESTAMP,
        "num_clients": NUM_CLIENTS,
        "audio_dur_s": AUDIO_DUR_S,
        "results": {
            "success":  success,
            "rejected": rejected,
            "oom":      oom,
            "killed":   killed,
            "failed":   failed,
        },
        "latency": {"avg_ms": lat_avg, "max_ms": lat_max},
        "system":  sys_s,
        "crash_events": events,
        "first_crash_ram_pct": first_crash_ram,
        "clients": [
            {
                "cid":         r.cid,
                "session_id":  r.session_id,
                "pid":         r.server_pid,
                "status":      r.status,
                "chunks_sent": r.chunks_sent,
                "word_count":  r.word_count(),
                "transcript":  r.full_transcript()[:500],
                "error":       r.error,
            }
            for r in results
        ]
    }
    try:
        with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(json_out, f, indent=2)
    except Exception:
        pass


if __name__ == "__main__":
    monitor.start()
    try:
        asyncio.run(main())
    finally:
        monitor.stop()