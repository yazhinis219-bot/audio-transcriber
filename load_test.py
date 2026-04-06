import asyncio
import websockets
import uuid
import wave
import time
import numpy as np
import statistics
import json
import psutil
import os

# ================= CONFIG =================
WS_URL = "ws://localhost:8000/ws/consultation"
AUDIO_FILE = "input.wav"

TARGET_SR = 16000
CHUNK_DURATION = 0.1
CHUNK_SIZE = int(TARGET_SR * CHUNK_DURATION)

STEP = 5
RAMP_DELAY = 5

# ── CLIENT_TIMEOUT must be >= gateway's tail-pass timeout (800 s) plus
# the time the client spends streaming audio (~audio_duration seconds).
# We add a 60 s network/scheduling buffer on top → 860 s.
# The gateway will ALWAYS send "end" within 800 s of receiving "END",
# so if the client times out first it means a real connection failure.
CLIENT_TIMEOUT = 860          # seconds — must exceed gateway tail timeout (800 s)
RECV_LOOP_TIMEOUT = 5.0       # seconds — short so wall-clock check fires promptly

MAX_CLIENTS = 100

LOG_FILE = "final_results.txt"
# ==========================================


# ---------- LOGGER ----------
def log(msg):
    print(msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------- SYSTEM ----------
def get_system_stats():
    process = psutil.Process(os.getpid())

    system_cpu = psutil.cpu_percent(interval=1)
    process_cpu = process.cpu_percent(interval=1)

    process_mem = process.memory_info().rss / (1024 * 1024)

    mem = psutil.virtual_memory()
    system_mem_pct = mem.percent

    return system_cpu, process_cpu, process_mem, system_mem_pct


# ---------- LOAD AUDIO ----------
def load_audio(filename):
    log("🔍 Loading audio...")

    wf = wave.open(filename, 'rb')
    sr = wf.getframerate()
    channels = wf.getnchannels()

    data = wf.readframes(wf.getnframes())
    wf.close()

    audio = np.frombuffer(data, dtype=np.int16)

    if channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1).astype(np.int16)

    if sr != TARGET_SR:
        duration = len(audio) / sr
        target_length = int(duration * TARGET_SR)

        audio = np.interp(
            np.linspace(0, len(audio), target_length),
            np.arange(len(audio)),
            audio
        ).astype(np.int16)

    log(f"✅ Audio ready: {len(audio)/TARGET_SR:.2f} sec")
    return audio


# ---------- STREAM ----------
async def stream_audio(ws, audio, first_send_time):
    idx = 0
    total_samples = len(audio)

    while idx < total_samples:
        chunk = audio[idx:idx + CHUNK_SIZE]

        if first_send_time[0] is None:
            first_send_time[0] = time.time()

        await ws.send(chunk.tobytes())
        idx += CHUNK_SIZE

        # small jitter
        await asyncio.sleep(CHUNK_DURATION + np.random.uniform(0, 0.02))

    await ws.send("END")


# ---------- CLIENT ----------
async def run_client(audio):
    session_id = str(uuid.uuid4())
    uri = f"{WS_URL}/{session_id}"

    messages = 0
    first_send_time = [None]
    first_response_time = None
    last_recv_time = None
    end_time = None

    completed = False
    timeout_flag = False

    # ── FIX 2: Collect text from "final_complete" (the authoritative full
    # transcript the gateway assembles) instead of stitching individual
    # "final" chunks.  Falls back to stitched chunks if "final_complete"
    # never arrives (e.g. completely silent audio).
    final_complete_text = ""
    stitched_chunks: list[str] = []

    try:
        async with websockets.connect(uri, max_size=2**23) as ws:
            sender = asyncio.create_task(stream_audio(ws, audio, first_send_time))
            start_time = time.time()

            while True:
                # ── FIX 3: Use a short poll timeout (RECV_LOOP_TIMEOUT) so
                # the overall CLIENT_TIMEOUT wall-clock check fires promptly.
                # Previously the recv() awaited CLIENT_TIMEOUT seconds, meaning
                # a silent server would stall the client for 900 s between polls.
                elapsed = time.time() - start_time
                if elapsed > CLIENT_TIMEOUT:
                    timeout_flag = True
                    break

                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=RECV_LOOP_TIMEOUT)

                    recv_time = time.time()
                    messages += 1
                    last_recv_time = recv_time

                    if first_response_time is None:
                        first_response_time = recv_time

                    try:
                        data = json.loads(msg)
                    except Exception:
                        data = {}

                    msg_type = data.get("type", "")

                    # ── FIX 4: Capture individual "final" chunks as fallback
                    if msg_type == "final":
                        stitched_chunks.append(data.get("text", ""))

                    # ── FIX 5: "final_complete" is the single authoritative
                    # full transcript emitted by the gateway at session end.
                    # This is what we should measure completeness against.
                    if msg_type == "final_complete":
                        final_complete_text = data.get("text", "")

                    if msg_type == "end":
                        completed = True
                        end_time = recv_time
                        break

                except asyncio.TimeoutError:
                    # Normal — server is still processing; keep polling
                    continue
                except Exception:
                    break

            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass

            latency = None
            if first_send_time[0] and first_response_time:
                latency = first_response_time - first_send_time[0]

            processing_time = None
            if first_send_time[0]:
                if end_time:
                    processing_time = end_time - first_send_time[0]
                elif last_recv_time:
                    processing_time = last_recv_time - first_send_time[0]

            # ── FIX 6: Resolve the best available transcript text.
            # Priority: final_complete > stitched chunks > empty
            full_text = final_complete_text or " ".join(stitched_chunks).strip()

            return {
                "latency": latency,
                "processing_time": processing_time,
                "messages": messages,
                "completed": completed,
                "timeout": timeout_flag,
                "full_text": full_text,
            }

    except Exception:
        return {
            "latency": None,
            "processing_time": None,
            "messages": 0,
            "completed": False,
            "timeout": True,
            "full_text": "",
        }


# ---------- RUN STEP ----------
async def run_step(num_clients, audio):
    log(f"\n🔹 Testing {num_clients} clients...\n")

    sys_cpu_b, proc_cpu_b, proc_mem_b, sys_mem_b = get_system_stats()

    log(f"⚙️ CPU BEFORE: {sys_cpu_b:.2f}% (System) | {proc_cpu_b:.2f}% (Process)")
    log(f"🧠 Memory BEFORE: {proc_mem_b:.2f} MB (Process) | {sys_mem_b:.2f}% (System)\n")

    tasks = [run_client(audio) for _ in range(num_clients)]
    results = await asyncio.gather(*tasks)

    log("📋 PER CLIENT (FULL METRICS)\n")

    throughputs = []
    processings = []
    queues = []
    latencies = []

    error_count = 0
    timeout_count = 0

    for i, r in enumerate(results):
        latency = r["latency"]
        processing = r["processing_time"]
        messages = r["messages"]

        # ── FIX 7: Error = session completed but transcript is genuinely
        # empty (server produced nothing).  A short transcript is NOT an
        # error — the audio clip may simply be short.  We no longer
        # penalise clients whose transcript is <20 chars; instead we only
        # flag clients where completed=False (connection dropped / timed
        # out before receiving "end") OR where completed=True but the
        # server returned absolutely no text at all.
        transcript_missing = r["completed"] and len(r["full_text"].strip()) == 0
        connection_failed  = not r["completed"]
        error = 1 if (transcript_missing or connection_failed) else 0
        timeout = 1 if r["timeout"] else 0

        error_count += error
        timeout_count += timeout

        valid = messages > 0 and processing is not None and processing > 0

        log(f"↳ Client {i+1}")

        if valid:
            throughput = messages / processing
            queue = latency * throughput if latency else 0

            throughputs.append(throughput)
            processings.append(processing)
            queues.append(queue)

            if latency:
                latencies.append(latency)

            log(f"   ↳ Throughput: {throughput:.2f}")
            log(f"   ↳ Processing: {processing:.2f}")
            log(f"   ↳ Queue: {queue:.2f}")
        else:
            log(f"   ↳ Throughput: N/A")
            log(f"   ↳ Processing: N/A")
            log(f"   ↳ Queue: N/A")

        log(f"   ↳ Error Rate: {error:.2f}")
        log(f"   ↳ Timeout Rate: {timeout:.2f}")
        log(f"   ↳ Latency: {latency if latency else 'N/A'}")
        # ── FIX 8: Show transcript length so you can spot silent/empty results
        log(f"   ↳ Transcript Length: {len(r['full_text'])} chars\n")

    log("------------------ FINAL SUMMARY ------------------\n")

    avg_throughput = statistics.mean(throughputs) if throughputs else 0
    avg_processing = statistics.mean(processings) if processings else 0
    avg_queue = statistics.mean(queues) if queues else 0

    error_rate = error_count / num_clients
    timeout_rate = timeout_count / num_clients

    p50 = statistics.median(latencies) if latencies else 0
    p95 = np.percentile(latencies, 95) if latencies else 0

    sys_cpu_a, proc_cpu_a, proc_mem_a, sys_mem_a = get_system_stats()

    log(f"↳ Throughput: {avg_throughput:.2f}")
    log(f"↳ Processing: {avg_processing:.2f}")
    log(f"↳ Queue: {avg_queue:.2f}")
    log(f"↳ Error Rate: {error_rate:.2f}")
    log(f"↳ Timeout Rate: {timeout_rate:.2f}")
    log(f"↳ Latency p50: {p50:.2f}")
    log(f"↳ Latency p95: {p95:.2f}")
    log(f"↳ CPU AFTER: {sys_cpu_a:.2f}% (System) | {proc_cpu_a:.2f}% (Process)")
    log(f"↳ Memory AFTER: {proc_mem_a:.2f} MB (Process) | {sys_mem_a:.2f}% (System)\n")

    # STOP CONDITION — only break when connections or the server truly fail
    if error_rate > 0.2 or timeout_rate > 0.2:
        log(f"\n🔥 SYSTEM BREAKING at {num_clients} clients\n")
        return False

    return True


# ---------- MAIN ----------
async def run_test():
    open(LOG_FILE, "w").close()

    log("🚀 STARTING LOAD TEST\n")

    audio = load_audio(AUDIO_FILE)

    clients = 1

    while clients <= MAX_CLIENTS:
        success = await run_step(clients, audio)

        if not success:
            log(f"\n✅ MAX CLIENTS BEFORE FAILURE: {clients}\n")
            break

        clients += STEP
        await asyncio.sleep(RAMP_DELAY)


# ---------- ENTRY ----------
if __name__ == "__main__":
    asyncio.run(run_test())