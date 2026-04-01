import asyncio
import websockets
import time
from pydub import AudioSegment
import numpy as np
import uuid
import json
import websockets.exceptions
from datetime import datetime
from collections import defaultdict
import csv
import os
import psutil
import threading

# ==============================
# CONFIGURATION
# ==============================
NUM_CLIENTS   = 25
CHUNK_MS      = 100
AUDIO_FILE    = "input.wav"
LOG_DIR       = "test_logs"
TIMESTAMP     = datetime.now().strftime("%Y%m%d_%H%M%S")
MAX_RETRIES   = 3
PING_TIMEOUT  = 5000   # ms  (must match websockets.connect ping_timeout)
PING_INTERVAL = 10     # sec (must match websockets.connect ping_interval)

os.makedirs(LOG_DIR, exist_ok=True)

# ==============================
# SYSTEM MONITOR  (CPU + RAM)
# ==============================
class SystemMonitor:
    def __init__(self):
        self.samples   = []          # list of dicts
        self.running   = False
        self.proc      = psutil.Process()

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                mem  = self.proc.memory_info()
                self.samples.append({
                    "ts"         : datetime.now(),
                    "cpu_pct"    : self.proc.cpu_percent(),
                    "rss_mb"     : mem.rss   / 1024 / 1024,
                    "vms_mb"     : mem.vms   / 1024 / 1024,
                    "sys_ram_pct": psutil.virtual_memory().percent,
                    "sys_ram_mb" : psutil.virtual_memory().used / 1024 / 1024,
                })
                time.sleep(0.5)
            except Exception:
                pass

    def summary(self):
        if not self.samples:
            return {}
        cpus      = [s["cpu_pct"]     for s in self.samples]
        rss       = [s["rss_mb"]      for s in self.samples]
        sys_ram   = [s["sys_ram_mb"]  for s in self.samples]
        sys_pct   = [s["sys_ram_pct"] for s in self.samples]
        return {
            "proc_cpu_avg"    : np.mean(cpus),
            "proc_cpu_peak"   : max(cpus),
            "proc_rss_avg_mb" : np.mean(rss),
            "proc_rss_peak_mb": max(rss),
            "sys_ram_avg_mb"  : np.mean(sys_ram),
            "sys_ram_peak_mb" : max(sys_ram),
            "sys_ram_avg_pct" : np.mean(sys_pct),
            "sys_ram_peak_pct": max(sys_pct),
            "samples"         : len(self.samples),
        }

system_monitor = SystemMonitor()

# ==============================
# CLIENT METRICS
# ==============================
class ClientMetrics:
    def __init__(self, client_id):
        self.client_id        = client_id
        self.session_id       = None
        self.start_time       = None
        self.end_time         = None
        self.status           = "PENDING"
        self.error_reason     = None

        # bytes / chunks
        self.total_bytes_sent  = 0
        self.total_chunks_sent = 0

        # reconnect / failure tracking
        self.reconnects        = 0
        self.failures          = []      # list of dicts

        # transcript + latency
        self.transcripts       = []      # list of dicts  {text, latency_ms, wait_ms}
        self.latencies_ms      = []      # end-to-end latency per transcript
        self.chunk_wait_ms     = []      # per-chunk RTT: time client waited for each ack/chunk send

        # connection
        self.connection_times  = []      # seconds to establish each WS connection

    # ---- helpers ----
    def add_failure(self, reason, bytes_sent, retry_count):
        self.failures.append({
            "reason"   : reason,
            "bytes"    : bytes_sent,
            "retry"    : retry_count,
            "ts"       : datetime.now(),
        })

    def add_transcript(self, text, latency_ms, wait_ms=0):
        self.transcripts.append({"text": text, "latency_ms": latency_ms, "wait_ms": wait_ms})
        self.latencies_ms.append(latency_ms)

    def add_chunk_wait(self, wait_ms):
        self.chunk_wait_ms.append(wait_ms)

    def summary(self):
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        return {
            "client_id"         : self.client_id,
            "session_id"        : self.session_id,
            "status"            : self.status,
            "duration_sec"      : round(duration, 2),
            "bytes_sent"        : self.total_bytes_sent,
            "chunks_sent"       : self.total_chunks_sent,
            "reconnects"        : self.reconnects,
            "failures"          : len(self.failures),
            "transcripts"       : len(self.transcripts),
            "avg_latency_ms"    : round(np.mean(self.latencies_ms),   1) if self.latencies_ms   else 0,
            "max_latency_ms"    : round(max(self.latencies_ms),        1) if self.latencies_ms   else 0,
            "avg_chunk_wait_ms" : round(np.mean(self.chunk_wait_ms),   1) if self.chunk_wait_ms  else 0,
            "max_chunk_wait_ms" : round(max(self.chunk_wait_ms),       1) if self.chunk_wait_ms  else 0,
            "error_reason"      : self.error_reason,
        }

metrics_dict = {}          # client_id → ClientMetrics

# ==============================
# LOAD AUDIO
# ==============================
def load_audio(file):
    audio   = AudioSegment.from_file(file)
    audio   = audio.set_frame_rate(16000).set_channels(1)
    samples = np.array(audio.get_array_of_samples()).astype(np.int16)
    return samples.tobytes()

audio_bytes        = load_audio(AUDIO_FILE)
AUDIO_DURATION_SEC = len(audio_bytes) / (16000 * 2)
BYTES_PER_SAMPLE   = 2
SAMPLES_PER_CHUNK  = int(16000 * (CHUNK_MS / 1000))
CHUNK_SIZE         = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE
TOTAL_CHUNKS       = len(audio_bytes) // CHUNK_SIZE

# ==============================
# CLIENT SIMULATION
# ==============================
async def simulate_client(client_id, max_retries=MAX_RETRIES):
    m             = ClientMetrics(client_id)
    metrics_dict[client_id] = m

    session_id    = str(uuid.uuid4())[:6]
    m.session_id  = session_id
    m.start_time  = datetime.now()

    url           = f"ws://localhost:8000/ws/consultation/{session_id}"
    retry_count   = 0
    bytes_sent    = 0
    chunks_sent   = 0
    done          = False

    print(f"[CONNECT] Client {client_id} | Session: {session_id}")

    while retry_count < max_retries and not done:
        try:
            t_connect = datetime.now()
            async with websockets.connect(
                url,
                ping_interval=PING_INTERVAL,
                ping_timeout=PING_TIMEOUT / 1000,
            ) as ws:
                m.connection_times.append((datetime.now() - t_connect).total_seconds())

                # Send resume context
                await ws.send(json.dumps({
                    "type"           : "resume",
                    "bytes_processed": bytes_sent,
                    "session_id"     : session_id,
                }))

                transcript_received = asyncio.Event()

                # ---- receive loop ----
                async def receive():
                    try:
                        async for message in ws:
                            data = json.loads(message)
                            if data.get("type") == "transcript":
                                text = data.get("text", "")
                                ts   = data.get("timestamps", {})
                                lat  = ts.get("audio_processed_ts", 0) - ts.get("audio_sent_ts", 0)
                                m.add_transcript(text, lat)
                                transcript_received.set()
                    except asyncio.CancelledError:
                        pass
                    except Exception:
                        pass

                recv_task = asyncio.create_task(receive())

                # ---- send loop (resume from where we stopped) ----
                index = bytes_sent
                try:
                    while index < len(audio_bytes):
                        chunk = audio_bytes[index : index + CHUNK_SIZE]
                        if not chunk:
                            break

                        t_before = time.monotonic()
                        await ws.send(chunk)
                        wait_ms  = (time.monotonic() - t_before) * 1000
                        m.add_chunk_wait(wait_ms)

                        await asyncio.sleep(CHUNK_MS / 1000)
                        index       += CHUNK_SIZE
                        bytes_sent  += len(chunk)
                        chunks_sent += 1

                    m.total_bytes_sent  = bytes_sent
                    m.total_chunks_sent = chunks_sent

                    await asyncio.sleep(1)
                    await ws.send(json.dumps({"type": "stop"}))

                    try:
                        await asyncio.wait_for(transcript_received.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        pass

                    await ws.close()
                    recv_task.cancel()

                    if transcript_received.is_set():
                        m.status = "SUCCESS"
                        print(f"[SUCCESS]    Client {client_id} | chunks={chunks_sent}/{TOTAL_CHUNKS} bytes={bytes_sent}")
                    else:
                        m.status       = "INCOMPLETE"
                        m.error_reason = "No transcript received"
                        print(f"[INCOMPLETE] Client {client_id} | chunks={chunks_sent}/{TOTAL_CHUNKS} bytes={bytes_sent}")

                    done = True

                except Exception as e:
                    recv_task.cancel()
                    raise

        except websockets.exceptions.ConnectionClosedOK:
            m.total_bytes_sent  = bytes_sent
            m.total_chunks_sent = chunks_sent
            m.status            = "SUCCESS"
            done                = True
            print(f"[CLOSED-OK]  Client {client_id} | chunks={chunks_sent}/{TOTAL_CHUNKS}")

        except (websockets.exceptions.WebSocketException, Exception) as e:
            retry_count += 1
            err_str      = str(e)
            m.add_failure(err_str, bytes_sent, retry_count)
            m.reconnects += 1
            pct = (bytes_sent / len(audio_bytes)) * 100 if audio_bytes else 0
            print(f"[RETRY {retry_count}/{max_retries}] Client {client_id} | progress={pct:.1f}% | {err_str[:60]}")
            if retry_count < max_retries:
                await asyncio.sleep(2 ** retry_count)
            else:
                m.status           = "FAILED"
                m.error_reason     = err_str
                m.total_bytes_sent  = bytes_sent
                m.total_chunks_sent = chunks_sent
                print(f"[FAILED]     Client {client_id}")

    m.end_time = datetime.now()

# ==============================
# SINGLE UNIFIED REPORT
# ==============================
def write_unified_report():
    report_path = os.path.join(LOG_DIR, f"load_test_report_{TIMESTAMP}.txt")
    W = 130        # line width

    summaries   = [m.summary() for m in metrics_dict.values()]
    sys_stat    = system_monitor.summary()

    # ── Aggregate metrics ─────────────────────────────────────────
    all_lat          = []
    all_wait         = []
    for m in metrics_dict.values():
        all_lat.extend(m.latencies_ms)
        all_wait.extend(m.chunk_wait_ms)

    total_failures   = sum(len(m.failures)     for m in metrics_dict.values())
    total_reconnects = sum(m.reconnects         for m in metrics_dict.values())
    total_transcripts= sum(len(m.transcripts)   for m in metrics_dict.values())
    total_bytes_sent = sum(m.total_bytes_sent   for m in metrics_dict.values())
    total_chunks_sent= sum(m.total_chunks_sent  for m in metrics_dict.values())
    successful       = sum(1 for s in summaries if s["status"] == "SUCCESS")
    failed           = sum(1 for s in summaries if s["status"] == "FAILED")
    incomplete       = sum(1 for s in summaries if s["status"] == "INCOMPLETE")

    avg_lat_ms       = np.mean(all_lat)  if all_lat  else 9000
    avg_wait_ms      = np.mean(all_wait) if all_wait else CHUNK_MS

    total_conn_attempts = NUM_CLIENTS * MAX_RETRIES
    failure_rate        = (total_failures / total_conn_attempts) * 100

    chunks_per_sec      = (NUM_CLIENTS * 1000) / CHUNK_MS
    processing_rate     = 1 / (avg_lat_ms / 1000) if avg_lat_ms else 0
    overload_factor     = chunks_per_sec / processing_rate if processing_rate else 0
    rtf                 = avg_lat_ms / CHUNK_MS

    backlog_per_sec     = max(0, chunks_per_sec - processing_rate)
    peak_backlog        = backlog_per_sec * (AUDIO_DURATION_SEC / 2)

    ping_violations     = sum(1 for l in all_lat if l > PING_TIMEOUT)

    test_durations      = [(m.end_time - m.start_time).total_seconds()
                           for m in metrics_dict.values() if m.end_time]
    test_wall_sec       = max(test_durations) if test_durations else 0

    # ── crash scoring ──────────────────────────────────────────────
    crash_indicators = []
    crash_score      = 0

    def indicator(label, pts, condition):
        nonlocal crash_score
        if condition:
            crash_indicators.append((label, pts))
            crash_score += pts

    indicator("[CRITICAL] Failure Rate > 90%",          3, failure_rate >= 90)
    indicator("[HIGH]     Failure Rate > 50%",          2, 50 <= failure_rate < 90)
    indicator("[MEDIUM]   Failure Rate > 30%",          1, 30 <= failure_rate < 50)
    indicator("[CRITICAL] ZERO Successful Requests",    3, total_transcripts == 0)
    indicator("[HIGH]     < 1 Success per Client",      2, 0 < total_transcripts < NUM_CLIENTS)
    indicator("[HIGH]     Excessive Reconnects",        2, total_reconnects >= total_failures * 0.8)
    indicator("[CRITICAL] Overload Factor > 100×",      3, overload_factor > 100)
    indicator("[HIGH]     Overload Factor > 10×",       2, 10 < overload_factor <= 100)
    indicator("[HIGH]     CPU Peak > 90%",              2, sys_stat.get("proc_cpu_peak", 0) > 90)
    indicator("[HIGH]     RAM Peak > 500 MB",           2, sys_stat.get("proc_rss_peak_mb", 0) > 500)

    if   crash_score >= 12: verdict = "COMPLETELY CRASHED (CRITICAL)"
    elif crash_score >= 8:  verdict = "SEVERELY DEGRADED  (MAJOR)"
    elif crash_score >= 4:  verdict = "DEGRADED           (MODERATE)"
    else:                   verdict = "OPERATIONAL        (STABLE)"

    # ══════════════════════════════════════════════════════════════
    def sep(char="="):  return char * W + "\n"
    def h(title, char="="): return sep(char) + f"{title}\n" + sep(char)

    with open(report_path, "w", encoding="utf-8") as f:

        # ─── HEADER ────────────────────────────────────────────────
        f.write(sep())
        f.write(f"  LOAD TEST REPORT  |  {TIMESTAMP}\n")
        f.write(sep())

        f.write("\nTEST CONFIGURATION\n" + sep("-"))
        f.write(f"  Audio File              : {AUDIO_FILE}\n")
        f.write(f"  Audio Duration          : {AUDIO_DURATION_SEC:.2f} s\n")
        f.write(f"  Audio Size              : {len(audio_bytes):,} bytes  ({len(audio_bytes)/1024:.1f} KB)\n")
        f.write(f"  Chunk Duration          : {CHUNK_MS} ms\n")
        f.write(f"  Chunk Size              : {CHUNK_SIZE:,} bytes  ({SAMPLES_PER_CHUNK} samples)\n")
        f.write(f"  Total Chunks per Client : {TOTAL_CHUNKS}\n")
        f.write(f"  Concurrent Clients      : {NUM_CLIENTS}\n")
        f.write(f"  Max Retries             : {MAX_RETRIES}\n")
        f.write(f"  WS Ping Interval        : {PING_INTERVAL} s\n")
        f.write(f"  WS Ping Timeout         : {PING_TIMEOUT} ms\n")
        f.write(f"  Wall-clock Test Time    : {test_wall_sec:.1f} s\n\n")

        # ─── OVERALL RESULTS ───────────────────────────────────────
        f.write("OVERALL RESULTS\n" + sep("-"))
        f.write(f"  Successful Clients      : {successful}/{NUM_CLIENTS}\n")
        f.write(f"  Failed Clients          : {failed}/{NUM_CLIENTS}\n")
        f.write(f"  Incomplete Clients      : {incomplete}/{NUM_CLIENTS}\n")
        f.write(f"  Total Connection Attempts: {total_conn_attempts}\n")
        f.write(f"  Total Failures          : {total_failures}\n")
        f.write(f"  Failure Rate            : {failure_rate:.1f}%\n")
        f.write(f"  Total Reconnects        : {total_reconnects}  (avg {total_reconnects/NUM_CLIENTS:.1f} per client)\n")
        f.write(f"  Total Transcripts       : {total_transcripts}  (expected {NUM_CLIENTS * 3})\n")
        f.write(f"  Transcript Success Rate : {(total_transcripts/(NUM_CLIENTS*3))*100:.1f}%\n\n")

        # ─── DATA SENT ─────────────────────────────────────────────
        f.write("DATA SENT TO SERVER\n" + sep("-"))
        f.write(f"  Total Bytes Sent        : {total_bytes_sent:,} bytes  ({total_bytes_sent/1024:.1f} KB)\n")
        f.write(f"  Expected Total Bytes    : {len(audio_bytes)*NUM_CLIENTS:,} bytes\n")
        f.write(f"  Data Delivery Rate      : {(total_bytes_sent/(len(audio_bytes)*NUM_CLIENTS))*100:.1f}%\n")
        f.write(f"  Total Chunks Sent       : {total_chunks_sent:,}\n")
        f.write(f"  Expected Total Chunks   : {TOTAL_CHUNKS * NUM_CLIENTS:,}\n")
        f.write(f"  Chunk Delivery Rate     : {(total_chunks_sent/(TOTAL_CHUNKS*NUM_CLIENTS))*100:.1f}%\n")
        f.write(f"  Avg Chunk Send Latency  : {avg_wait_ms:.1f} ms  (time client waited per ws.send())\n")
        if all_wait:
            f.write(f"  Max Chunk Send Latency  : {max(all_wait):.1f} ms\n")
        f.write("\n")

        

        # ─── SYSTEM RESOURCES ──────────────────────────────────────
        f.write("SYSTEM RESOURCES  (process + host)\n" + sep("-"))
        if sys_stat:
            f.write(f"  Process CPU  avg / peak : {sys_stat['proc_cpu_avg']:.1f}%  /  {sys_stat['proc_cpu_peak']:.1f}%\n")
            f.write(f"  Process RAM  avg / peak : {sys_stat['proc_rss_avg_mb']:.1f} MB  /  {sys_stat['proc_rss_peak_mb']:.1f} MB  (RSS)\n")
            f.write(f"  Host RAM     avg / peak : {sys_stat['sys_ram_avg_mb']:.0f} MB  /  {sys_stat['sys_ram_peak_mb']:.0f} MB\n")
            f.write(f"  Host RAM %   avg / peak : {sys_stat['sys_ram_avg_pct']:.1f}%  /  {sys_stat['sys_ram_peak_pct']:.1f}%\n")
            f.write(f"  Monitor Samples         : {sys_stat['samples']}\n")
        else:
            f.write("  No resource data collected.\n")
        f.write("\n")

        # ─── OVERLOAD ANALYSIS ─────────────────────────────────────
        f.write("OVERLOAD ANALYSIS  (root cause)\n" + sep("-"))
        f.write(f"  Chunks Incoming / sec   : {chunks_per_sec:.1f}  ({NUM_CLIENTS} clients × {1000//CHUNK_MS} chunks/s)\n")
        f.write(f"  Server Processing Rate  : {processing_rate:.4f} chunks/sec  (1 / {avg_lat_ms:.0f}ms)\n")
        f.write(f"  Overload Factor         : {overload_factor:.1f}×\n")
        f.write(f"  Real-Time Factor        : {rtf:.1f}×  (server is {rtf:.0f}× slower than real-time)\n")
        f.write(f"  Queue Buildup Rate      : {backlog_per_sec:.1f} chunks/sec\n")
        f.write(f"  Estimated Peak Backlog  : ~{int(peak_backlog):,} chunks\n\n")
        f.write("  Queue growth over time:\n")
        for t in [10, 30, 60]:
            f.write(f"    After {t:>3}s : ~{int(backlog_per_sec * t):,} pending chunks\n")
        f.write("\n")

        # ─── CRASH THRESHOLD ──────────────────────────────────────
        f.write("CRASH THRESHOLD ANALYSIS\n" + sep("-"))
        t1 = (PING_TIMEOUT / 1000) / (avg_lat_ms / 1000) / (1000 / CHUNK_MS)
        t2 = processing_rate / (1000 / CHUNK_MS)
        f.write(f"  Threshold 1 (latency > ping timeout) : ~{int(t1)} clients\n")
        f.write(f"  Threshold 2 (incoming > processing)  : ~{int(t2)+1} clients\n")
        f.write(f"  Current clients                      : {NUM_CLIENTS}\n")
        if NUM_CLIENTS >= 10:
            f.write("  Assessment : ✗✗✗  WELL ABOVE crash threshold\n")
        elif NUM_CLIENTS >= 6:
            f.write("  Assessment : ✗✗   ABOVE crash threshold\n")
        else:
            f.write("  Assessment : ✗    APPROACHING crash threshold\n")
        f.write("\n")

        
        
       

        # ─── PER-CLIENT TABLE ──────────────────────────────────────
        f.write("PER-CLIENT BREAKDOWN\n" + sep("-"))
        hdr = (f"{'ID':<5} {'Session':<9} {'Status':<11} {'Dur(s)':<8} "
               f"{'Bytes Sent':>11} {'Chunks':>8} {'Reconnects':>11} {'Fails':>6} "
               f"{'AvgLat(ms)':>12} {'AvgWait(ms)':>12} {'Txns':>5}")
        f.write(hdr + "\n" + "-" * W + "\n")

        for s in sorted(summaries, key=lambda x: x["client_id"]):
            cid  = s["client_id"]
            b    = s["bytes_sent"]
            c    = s["chunks_sent"]
            bpct = f"{(b/len(audio_bytes))*100:.0f}%" if audio_bytes else "?"
            cpct = f"{(c/TOTAL_CHUNKS)*100:.0f}%"   if TOTAL_CHUNKS else "?"
            row  = (f"C{cid:<4} {s['session_id']:<9} {s['status']:<11} {s['duration_sec']:<8.1f} "
                    f"{b:>8,} {bpct:>3} {c:>5} {cpct:>3} {s['reconnects']:>11} {s['failures']:>6} "
                    f"{s['avg_latency_ms']:>12.1f} {s['avg_chunk_wait_ms']:>12.1f} {s['transcripts']:>5}")
            f.write(row + "\n")
        f.write("\n")

        # ─── PER-CLIENT DETAIL (chunks / bytes / latency wait) ─────
        f.write("PER-CLIENT DETAIL  (chunks, bytes, latency, wait times)\n" + sep("-"))
        for client_id in sorted(metrics_dict.keys()):
            m = metrics_dict[client_id]
            bpct = (m.total_bytes_sent / len(audio_bytes)) * 100 if audio_bytes else 0
            cpct = (m.total_chunks_sent / TOTAL_CHUNKS)     * 100 if TOTAL_CHUNKS else 0

            f.write(f"\n  CLIENT {client_id}  |  session={m.session_id}  |  status={m.status}\n")
            f.write(f"    Bytes  sent : {m.total_bytes_sent:>10,}  /  {len(audio_bytes):,}  ({bpct:.1f}%)\n")
            f.write(f"    Chunks sent : {m.total_chunks_sent:>10,}  /  {TOTAL_CHUNKS}  ({cpct:.1f}%)\n")
            f.write(f"    Duration    : {(m.end_time-m.start_time).total_seconds():.2f} s\n" if m.end_time else "")
            f.write(f"    Reconnects  : {m.reconnects}\n")

            if m.chunk_wait_ms:
                f.write(f"    Chunk wait  : avg={np.mean(m.chunk_wait_ms):.1f} ms  "
                        f"min={min(m.chunk_wait_ms):.1f}  max={max(m.chunk_wait_ms):.1f}  "
                        f"(time client stalled per ws.send)\n")

            if m.latencies_ms:
                f.write(f"    Txn latency : avg={np.mean(m.latencies_ms):.0f} ms  "
                        f"min={min(m.latencies_ms):.0f}  max={max(m.latencies_ms):.0f}  "
                        f"median={np.median(m.latencies_ms):.0f}  std={np.std(m.latencies_ms):.0f}\n")
                f.write(f"    Latency vals: {[f'{l:.0f}' for l in m.latencies_ms]}\n")

            if m.failures:
                f.write(f"    Failures ({len(m.failures)}):\n")
                for fi, fail in enumerate(m.failures, 1):
                    prog = (fail["bytes"] / len(audio_bytes)) * 100 if audio_bytes else 0
                    f.write(f"      [{fi}] retry={fail['retry']}  bytes={fail['bytes']:,} ({prog:.1f}%)"
                            f"  ts={fail['ts'].strftime('%H:%M:%S.%f')[:-3]}\n")
                    f.write(f"           {fail['reason'][:90]}\n")

            if m.transcripts:
                f.write(f"    Transcripts ({len(m.transcripts)}):\n")
                for ti, t in enumerate(m.transcripts, 1):
                    f.write(f"      [{ti}] lat={t['latency_ms']:.0f} ms  \"{t['text'][:80]}\"\n")

        f.write("\n")

        # ─── FOOTER ───────────────────────────────────────────────
        f.write(sep())
        f.write(f"  END OF REPORT  |  {report_path}\n")
        f.write(sep())

    print(f"\n[SAVED] Unified report → {os.path.abspath(report_path)}")
    return report_path

# ==============================
# CSV (one row per client)
# ==============================
def write_csv():
    csv_path  = os.path.join(LOG_DIR, f"metrics_{TIMESTAMP}.csv")
    summaries = [m.summary() for m in metrics_dict.values()]
    if not summaries:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(f"[SAVED] CSV metrics   → {os.path.abspath(csv_path)}")

# ==============================
# MAIN
# ==============================
async def main():
    tasks = [simulate_client(i) for i in range(NUM_CLIENTS)]
    # stagger starts slightly so not all hit server at t=0
    for i, coro in enumerate(tasks):
        await asyncio.sleep(0.2)
    await asyncio.gather(*[simulate_client(i) for i in range(NUM_CLIENTS)])

if __name__ == "__main__":
    print(f"\nStarting load test  |  clients={NUM_CLIENTS}  chunk={CHUNK_MS}ms  audio={AUDIO_DURATION_SEC:.1f}s\n")
    system_monitor.start()
    asyncio.run(main())
    system_monitor.stop()
    time.sleep(1)          # let monitor flush last sample

    print("\nGenerating report …\n")
    write_unified_report()
    write_csv()
    print(f"\nDone. Logs in: {os.path.abspath(LOG_DIR)}\n")