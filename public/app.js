// ── Elements ──
const recordBtn     = document.getElementById('record-btn');
const recordStatus  = document.getElementById('record-status');
const sessionIdEl   = document.getElementById('session-id');
const wsDot         = document.getElementById('ws-dot');
const wsLabel       = document.getElementById('ws-label');
const statusBadge   = document.getElementById('status-badge');
const finalEl       = document.getElementById('final-text');
const partialEl     = document.getElementById('partial-text');
const placeholder   = document.getElementById('placeholder');
const transcriptArea = document.getElementById('transcript-area');
const wordCountEl   = document.getElementById('word-count');

// ── State ──
let websocket    = null;
let isRecording  = false;
let sessionId    = crypto.randomUUID();
let audioContext = null;
let mediaStream  = null;
let scriptNode   = null;
let finalWords   = 0;   // count for word-count badge

sessionIdEl.textContent = sessionId.slice(0, 8) + '…';

// ── WS status ──
function setWsStatus(state) {
  // state: 'connected' | 'disconnected' | 'recording'
  wsDot.className = 'dot';
  statusBadge.className = 'status-badge';
  if (state === 'connected' || state === 'recording') {
    wsDot.classList.add(state === 'recording' ? 'pulse' : 'on');
    statusBadge.classList.add('connected');
    wsLabel.textContent = state === 'recording' ? 'Recording' : 'Connected';
  } else {
    wsLabel.textContent = 'Disconnected';
  }
}

// ── Transcript rendering ──
function appendFinal(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  // Add a space separator if there's already content
  const sep = finalEl.textContent.length > 0 ? ' ' : '';
  finalEl.textContent += sep + trimmed;
  finalWords += trimmed.split(/\s+/).filter(Boolean).length;
  updateWordCount();
  hidePlaceholder();
  scrollBottom();
}

// ── NEW: Replace entire transcript with the authoritative full-audio result ──
function replaceFinalTranscript(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  // Wipe everything that was shown during rolling passes and show the
  // single accurate result from the full-audio final pass instead.
  finalEl.textContent = trimmed;
  clearPartial();
  finalWords = trimmed.split(/\s+/).filter(Boolean).length;
  updateWordCount();
  hidePlaceholder();
  scrollBottom();
}

function setPartial(text) {
  const trimmed = text.trim();
  if (trimmed === partialEl.textContent.trim()) return;

  // Animate on change
  partialEl.classList.remove('partial-updated');
  void partialEl.offsetWidth; // reflow
  partialEl.classList.add('partial-updated');

  const sep = finalEl.textContent.length > 0 ? ' ' : '';
  partialEl.textContent = trimmed ? sep + trimmed : '';
  partialEl.classList.toggle('active', trimmed.length > 0);

  if (trimmed) hidePlaceholder();
  scrollBottom();
}

function clearPartial() {
  partialEl.textContent = '';
  partialEl.classList.remove('active', 'partial-updated');
}

function hidePlaceholder() {
  placeholder.classList.remove('visible');
}

function updateWordCount() {
  wordCountEl.textContent = `${finalWords} word${finalWords !== 1 ? 's' : ''}`;
}

function scrollBottom() {
  transcriptArea.scrollTop = transcriptArea.scrollHeight;
}

function resetTranscript() {
  finalEl.textContent = '';
  clearPartial();
  placeholder.classList.add('visible');
  finalWords = 0;
  updateWordCount();
}

// ── Audio helpers ──
function floatTo16BitPCM(float32Array) {
  const buf = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buf);
  for (let i = 0; i < float32Array.length; i++) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

function downsampleBuffer(buffer, inSR, outSR) {
  if (outSR >= inSR) return buffer;
  const ratio = inSR / outSR;
  const newLen = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLen);
  let offsetResult = 0, offsetBuffer = 0;
  while (offsetResult < result.length) {
    const nextOffset = Math.round((offsetResult + 1) * ratio);
    let acc = 0, count = 0;
    for (let i = offsetBuffer; i < nextOffset && i < buffer.length; i++, count++) acc += buffer[i];
    result[offsetResult++] = count ? acc / count : 0;
    offsetBuffer = nextOffset;
  }
  return result;
}

// ── Recording ──
async function startRecording() {
  if (isRecording) return;

  resetTranscript();
  sessionId = crypto.randomUUID();
  sessionIdEl.textContent = sessionId.slice(0, 8) + '…';

  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

  audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });

  const source = audioContext.createMediaStreamSource(mediaStream);
  scriptNode = audioContext.createScriptProcessor(4096, 1, 1);

  websocket = new WebSocket(`ws://${location.host}/ws/consultation/${sessionId}`);
  websocket.binaryType = 'arraybuffer';

  websocket.onopen = () => {
    setWsStatus('recording');
    recordBtn.textContent = '';
    recordBtn.insertAdjacentHTML('afterbegin', `
      <svg class="mic-icon" viewBox="0 0 24 24"><rect x="9" y="7" width="6" height="10" rx="1"/></svg>
      Stop Recording
    `);
    recordBtn.classList.add('recording');
    recordBtn.disabled = false;
    recordStatus.textContent = 'Recording… speak clearly.';
    isRecording = true;

    websocket.send(JSON.stringify({ type: 'start', sample_rate_hz: 16000 }));

    scriptNode.onaudioprocess = (e) => {
      if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
      const ds = downsampleBuffer(e.inputBuffer.getChannelData(0), audioContext.sampleRate, 16000);
      websocket.send(floatTo16BitPCM(ds));
    };

    source.connect(scriptNode);
    scriptNode.connect(audioContext.destination);
  };

  websocket.onmessage = (event) => {
    if (typeof event.data !== 'string') return;
    try {
      const msg = JSON.parse(event.data);

      if (msg.type === 'final') {
        // Rolling segment confirmed during live recording
        appendFinal(msg.text);
        clearPartial();

      } else if (msg.type === 'partial') {
        // Live "typing..." preview
        setPartial(msg.text);

      } else if (msg.type === 'final_complete') {
        // ── THIS IS THE KEY ADDITION ──
        // The backend has finished transcribing the full audio in one pass.
        // This single result is the most accurate — replace everything shown
        // during rolling passes with this authoritative transcript.
        replaceFinalTranscript(msg.text);
        recordStatus.textContent = 'Transcription complete.';

      } else if (msg.type === 'end') {
        // Session fully closed
        recordStatus.textContent = 'Transcription complete.';
        clearPartial();
        if (msg.transcript_file) {
          showTranscriptSaved(msg.transcript_file);
        }

      } else if (msg.type === 'warning') {
        console.warn('[Transcription warning]', msg.text);
      }

    } catch {
      // Legacy plain-text fallback
      appendFinal(event.data);
    }
  };

  websocket.onclose = () => {
    setWsStatus('disconnected');
    resetRecordButton();
    if (isRecording) recordStatus.textContent = 'Connection closed.';
    isRecording = false;
  };

  websocket.onerror = () => {
    setWsStatus('disconnected');
    recordStatus.textContent = 'WebSocket error — check console.';
  };

  recordBtn.disabled = true;
}

function stopRecording() {
  if (!isRecording) return;
  isRecording = false;

  if (scriptNode) { scriptNode.disconnect(); scriptNode.onaudioprocess = null; scriptNode = null; }
  if (audioContext) { audioContext.close(); audioContext = null; }
  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }

  if (websocket && websocket.readyState === WebSocket.OPEN) websocket.send('END');

  resetRecordButton();
  setWsStatus('connected');
  recordStatus.textContent = 'Stopped. Finalising…';
}

function resetRecordButton() {
  recordBtn.innerHTML = `
    <svg class="mic-icon" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 1a4 4 0 0 1 4 4v6a4 4 0 0 1-8 0V5a4 4 0 0 1 4-4zm6.5 9a.5.5 0 0 1 .5.5A7.001 7.001 0 0 1 12.5 17.93V21h2.5a.5.5 0 0 1 0 1h-6a.5.5 0 0 1 0-1H11.5v-3.07A7.001 7.001 0 0 1 5 10.5a.5.5 0 0 1 1 0 6 6 0 0 0 12 0 .5.5 0 0 1 .5-.5z"/>
    </svg>
    Start Recording`;
  recordBtn.classList.remove('recording');
  recordBtn.disabled = false;
}

function showTranscriptSaved(filename) {
  const notification = document.createElement('div');
  notification.className = 'transcript-notification';
  notification.innerHTML = `
    <div class="notification-content">
      <strong>Transcript Saved!</strong><br>
      File: ${filename}<br>
      <small>Location: /transcripts/${filename}</small>
    </div>
    <button class="notification-close" onclick="this.parentElement.remove()">×</button>
  `;
  
  if (!document.querySelector('#notification-styles')) {
    const style = document.createElement('style');
    style.id = 'notification-styles';
    style.textContent = `
      .transcript-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: #22c55e;
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        z-index: 1000;
        max-width: 350px;
        animation: slideIn 0.3s ease;
        display: flex;
        align-items: flex-start;
        gap: 10px;
      }
      .notification-content { flex: 1; }
      .notification-close {
        background: none;
        border: none;
        color: white;
        font-size: 18px;
        cursor: pointer;
        padding: 0;
        opacity: 0.8;
      }
      .notification-close:hover { opacity: 1; }
      @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to   { transform: translateX(0);   opacity: 1; }
      }
    `;
    document.head.appendChild(style);
  }
  
  document.body.appendChild(notification);
  setTimeout(() => { if (notification.parentElement) notification.remove(); }, 5000);
}

recordBtn.addEventListener('click', () => {
  if (!isRecording) {
    startRecording().catch(err => {
      console.error(err);
      recordStatus.textContent = 'Failed to start. Check microphone permissions.';
    });
  } else {
    stopRecording();
  }
});