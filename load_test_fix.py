import asyncio
import websockets
import uuid
import wave
import time
import numpy as np
import statistics
import json
import psutil
from datetime import datetime

# ================= CONFIG =================
WS_URL = "ws://localhost:8000/ws/consultation"
AUDIO_FILE = "input.wav"

TARGET_SR = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(TARGET_SR * CHUNK_DURATION)

MAX_CLIENTS = 2
STEP = 1
ITERATIONS = 3
RAMP_DELAY = 3

REPORT_FILE = "server_capacity_report.txt"
# ==========================================

# -------- GLOBAL TRACKING --------
TOTAL_CHUNKS_SENT = 0
TOTAL_TRANSCRIPTS = 0

CPU_HISTORY = []
RAM_HISTORY = []
STEP_RESULTS = []
START_TIME = None


# ---------- SYSTEM ----------
def get_memory_usage():
    return psutil.virtual_memory().used / (1024 * 1024)


async def monitor_system():
    global CPU_HISTORY, RAM_HISTORY, START_TIME
    while True:
        now = time.time() - START_TIME
        CPU_HISTORY.append((now, psutil.cpu_percent()))
        RAM_HISTORY.append((now, get_memory_usage()))
        await asyncio.sleep(5)


# ---------- AUDIO ----------
def load_audio(filename):
    wf = wave.open(filename, 'rb')
    sr = wf.getframerate()
    data = wf.readframes(wf.getnframes())
    wf.close()

    audio = np.frombuffer(data, dtype=np.int16)

    if sr != TARGET_SR:
        duration = len(audio) / sr
        target_length = int(duration * TARGET_SR)
        audio = np.interp(
            np.linspace(0, len(audio), target_length),
            np.arange(len(audio)),
            audio
        ).astype(np.int16)

    return audio


# ---------- STREAM ----------
async def stream_audio(ws, audio):
    global TOTAL_CHUNKS_SENT

    idx = 0
    start_time = time.perf_counter()

    while idx < len(audio):
        chunk = audio[idx:idx + CHUNK_SIZE]

        TOTAL_CHUNKS_SENT += 1
        await ws.send(chunk.tobytes())

        idx += CHUNK_SIZE

        expected_time = start_time + (idx / TARGET_SR)
        now = time.perf_counter()
        if expected_time > now:
            await asyncio.sleep(expected_time - now)

    await ws.send("END")


# ---------- CLIENT ----------
async def run_client(audio):
    global TOTAL_TRANSCRIPTS

    uri = f"{WS_URL}/{uuid.uuid4()}"
    messages = 0

    try:
        async with websockets.connect(uri, max_size=10_000_000) as ws:
            sender = asyncio.create_task(stream_audio(ws, audio))

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=5)
                    messages += 1
                    TOTAL_TRANSCRIPTS += 1

                    if '"type": "end"' in msg:
                        break

                except asyncio.TimeoutError:
                    continue
                except:
                    break

            await sender

    except:
        pass

    return messages


# ---------- REPORT ----------
def write_report(content):
    with open(REPORT_FILE, "a", encoding="utf-8") as f:
        f.write(content + "\n")


# ---------- RUN STEP ----------
async def run_step(num_clients, audio):
    global TOTAL_CHUNKS_SENT, TOTAL_TRANSCRIPTS
    global CPU_HISTORY, RAM_HISTORY, START_TIME, STEP_RESULTS

    # RESET
    TOTAL_CHUNKS_SENT = 0
    TOTAL_TRANSCRIPTS = 0
    CPU_HISTORY = []
    RAM_HISTORY = []

    START_TIME = time.time()
    monitor_task = asyncio.create_task(monitor_system())

    print(f"\n🔹 Testing {num_clients} clients...\n")

    for i in range(ITERATIONS):
        tasks = [run_client(audio) for _ in range(num_clients)]

        start = time.time()
        results = await asyncio.gather(*tasks)
        end = time.time()

        throughput = sum(results) / (end - start)

        print(f"  Iteration {i+1} → Throughput: {throughput:.2f}")

    monitor_task.cancel()

    # ---------- FINAL METRICS ----------
    processing = TOTAL_TRANSCRIPTS / TOTAL_CHUNKS_SENT if TOTAL_CHUNKS_SENT else 0
    loss = 1 - processing

    avg_cpu = statistics.mean([c for _, c in CPU_HISTORY]) if CPU_HISTORY else 0
    avg_ram = statistics.mean([r for _, r in RAM_HISTORY]) if RAM_HISTORY else 0

    # ---------- STATUS ----------
    if processing >= 0.9:
        status = "HEALTHY ✅"
    elif processing >= 0.7:
        status = "DEGRADED ⚠️"
    else:
        status = "FAILED ❌"

    # ---------- TERMINAL OUTPUT ----------
    print("\n  📊 RESULT")
    print(f"  Chunks Sent     : {TOTAL_CHUNKS_SENT}")
    print(f"  Transcripts     : {TOTAL_TRANSCRIPTS}")
    print(f"  Processing Rate : {processing*100:.2f}%")
    print(f"  Loss Rate       : {loss*100:.2f}%")
    print(f"  Avg CPU         : {avg_cpu:.1f}%")
    print(f"  Avg Memory      : {avg_ram:.0f} MB")
    print(f"  Status          : {status}")

    # STORE FOR CAPACITY
    STEP_RESULTS.append({
        "clients": num_clients,
        "processing": processing,
        "cpu": avg_cpu,
        "status": status
    })

    # ---------- REPORT FILE ----------
    report = []
    report.append("\n" + "="*80)
    report.append(f"{num_clients} CLIENTS TEST")
    report.append(f"Chunks Sent: {TOTAL_CHUNKS_SENT}")
    report.append(f"Transcripts: {TOTAL_TRANSCRIPTS}")
    report.append(f"Processing: {processing:.4f}")
    report.append(f"Loss: {loss:.4f}")
    report.append(f"CPU: {avg_cpu:.2f}%")
    report.append(f"Memory: {avg_ram:.2f} MB")
    report.append(f"Status: {status}")

    write_report("\n".join(report))


# ---------- MAIN ----------
async def run_test():
    global STEP_RESULTS

    open(REPORT_FILE, "w").write(
        f"SERVER CAPACITY REPORT\nGenerated: {datetime.now()}\n"
    )

    audio = load_audio(AUDIO_FILE)

    for clients in range(1, MAX_CLIENTS + 1, STEP):
        await run_step(clients, audio)
        await asyncio.sleep(RAMP_DELAY)

    # ---------- CAPACITY ----------
    capacity = None
    crash = None

    for r in STEP_RESULTS:
        if r["status"] == "FAILED" or r["cpu"] >= 95:
            crash = r["clients"]
            break
        else:
            capacity = r["clients"]

    print("\n" + "="*50)
    print("FINAL RESULT")
    print(f"Server Capacity : {capacity}")
    print(f"Crash Point     : {crash}")

    write_report("\nFINAL RESULT")
    write_report(f"Server Capacity: {capacity}")
    write_report(f"Crash Point: {crash}")


# ---------- ENTRY ----------
if __name__ == "__main__":
    asyncio.run(run_test())