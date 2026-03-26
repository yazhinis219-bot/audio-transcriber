/**
 * useTranscription — React hook for real-time NHS audio transcription.
 */

import { useState, useRef, useCallback, useEffect } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || `ws://${window.location.host}/ws/consultation`
const SAMPLE_RATE = 16000

// We prepend Date.now() (float64, 8 bytes) to every frame structure
const WORKLET_CODE = `
class PCMCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super()
    this._buffer = []
    this._bufferSize = 4800  // 300ms at 16kHz
  }

  process(inputs) {
    const input = inputs[0]
    if (!input || !input[0]) return true

    const float32 = input[0]
    for (let i = 0; i < float32.length; i++) {
      this._buffer.push(float32[i])
    }

    if (this._buffer.length >= this._bufferSize) {
      // 8 bytes for Float64 timestamp + rest for Int16 PCM
      const msgBuffer = new ArrayBuffer(8 + this._bufferSize * 2)
      const view = new DataView(msgBuffer)
      view.setFloat64(0, Date.now(), true) // true = little-endian

      const pcm = new Int16Array(msgBuffer, 8)
      for (let i = 0; i < this._bufferSize; i++) {
        const s = Math.max(-1, Math.min(1, this._buffer[i]))
        pcm[i] = s < 0 ? s * 32768 : s * 32767
      }

      this.port.postMessage(msgBuffer, [msgBuffer])
      this._buffer = this._buffer.slice(this._bufferSize)
    }
    return true
  }
}
registerProcessor('pcm-capture-processor', PCMCaptureProcessor)
`

export function useTranscription() {
  const [status, setStatus] = useState('idle') 
  const [sessionId, setSessionId] = useState(null)
  const [error, setError] = useState(null)

  const [utterances, setUtterances] = useState([])   
  const [liveText, setLiveText] = useState('')        
  const [stats, setStats] = useState({
    chunksProcessed: 0,
    transcriptsEmitted: 0,
    lastInferenceMs: 0,
    queueLatencyMs: 0,
  })

  // Telemetry storing the timestamps of recent messages
  const [telemetry, setTelemetry] = useState([])

  const wsRef = useRef(null)
  const audioCtxRef = useRef(null)
  const workletNodeRef = useRef(null)
  const streamRef = useRef(null)
  const workletBlobUrlRef = useRef(null)

  useEffect(() => {
    return () => stopRecording()
  }, [])

  const currentParagraphRef = useRef("")

  const connectWS = useCallback((sid) => {
    const url = `${WS_URL}/${sid}`
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      setStatus('connected')
      setError(null)
    }

    ws.onmessage = (evt) => {
      if (typeof evt.data !== 'string') return
      try {
        const msg = JSON.parse(evt.data)
        handleServerMessage(msg)
      } catch (_) {}
    }

    ws.onerror = (e) => {
      setError('WebSocket connection failed')
      setStatus('error')
    }

    ws.onclose = () => {
      setStatus(prev => prev === 'recording' ? 'stopped' : prev)
    }

    wsRef.current = ws
    return ws
  }, [])
  const handleServerMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'connected':
        setSessionId(msg.session_id)
        setStatus('recording')
        break

      case 'transcript':
        setUtterances(prev => {
          let text = msg.text

          // Fix common capitalization and typo issues across the whole text
          text = text.replace(/\b(i|Ok|ok|oky)\b/g, (match) => {
             if (match.toLowerCase() === 'i') return 'I';
             if (match.toLowerCase() === 'ok' || match.toLowerCase() === 'oky') return 'OK';
             return match;
          });
          
          // Boundary punctuation naturally handled by Whisper.          
          const words = text.split(" ");
          if (words.length > 0) {
            let lastWord = words[words.length - 1].replace(/[.,!?]/g, "");
            if (lastWord.toLowerCase() === "teh") words[words.length - 1] = words[words.length - 1].replace(/teh/i, "the");
            if (lastWord.toLowerCase() === "ike") words[words.length - 1] = words[words.length - 1].replace(/ike/i, "like");
            if (lastWord.toLowerCase() === "runned") words[words.length - 1] = words[words.length - 1].replace(/runned/i, "ran");
            text = words.join(" ");
          }

          const idx = prev.findIndex(u => u.utterance_id === msg.utterance_id)
          
          // Overlap deduplication: if this is a brand new confirmed chunk
          if (idx === -1 && prev.length > 0 && msg.is_final) {
              const lastText = prev[prev.length - 1].text || "";
              const oldWords = lastText.split(" ").filter(w => w);
              const newWords = text.split(" ").filter(w => w);
              
              let overlapCount = 0;
              const maxOverlap = Math.min(oldWords.length, newWords.length);
              
              for (let i = 1; i <= maxOverlap; i++) {
                  const suffix = oldWords.slice(-i).join(" ").toLowerCase().replace(/[.,!?]/g, "");
                  const prefix = newWords.slice(0, i).join(" ").toLowerCase().replace(/[.,!?]/g, "");
                  if (suffix === prefix) {
                      overlapCount = i;
                  }
              }
              if (overlapCount > 0) {
                  text = newWords.slice(overlapCount).join(" ");
              }
              
              // If fully duplicated after dedup, skip entirely
              if (!text.trim()) {
                  return prev;
              }
          }

          const newEntry = {
            utterance_id: msg.utterance_id || crypto.randomUUID(),
            text: text,
            is_final: msg.is_final,
            timestamp: Date.now(),
            inference_ms: msg.inference_ms,
            timestamps: msg.timestamps,
          }
          if (idx >= 0) {
            const updated = [...prev]
            updated[idx] = newEntry
            return updated
          }
          return [...prev, newEntry]
        })
        
        if (msg.timestamps) {
            setTelemetry(prev => {
                const updated = [...prev, {
                    utterance_id: msg.utterance_id,
                    text: msg.text,
                    timestamps: msg.timestamps
                }];
                return updated.slice(-10);
            });
        }
        
        setLiveText('')
        setStats(prev => ({
          ...prev,
          transcriptsEmitted: prev.transcriptsEmitted + 1,
          lastInferenceMs: Math.round(msg.inference_ms || 0),
        }))
        break

      case 'partial':
        setUtterances(prev => {
          setLiveText(msg.text);
          return prev;
        });
        break

      case 'silence':
        setLiveText('')
        break

      case 'chunk_ack':
        setStats(prev => ({
          ...prev,
          chunksProcessed: msg.chunk_index,
        }))
        break

      case 'error':
        setError(msg.error)
        break

      case 'ping':
        wsRef.current?.send(JSON.stringify({ type: 'ping' }))
        break

      default:
        break
    }
  }, [])


  const startRecording = useCallback(async () => {
    if (status === 'recording') return

    setStatus('connecting')
    setError(null)
    setUtterances([])
    setLiveText('')
    setTelemetry([])

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })
      streamRef.current = stream

      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
      audioCtxRef.current = ctx

      const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' })
      const blobUrl = URL.createObjectURL(blob)
      workletBlobUrlRef.current = blobUrl
      await ctx.audioWorklet.addModule(blobUrl)

      const source = ctx.createMediaStreamSource(stream)
      const workletNode = new AudioWorkletNode(ctx, 'pcm-capture-processor')
      workletNodeRef.current = workletNode
      source.connect(workletNode)
      workletNode.connect(ctx.destination)

      const sid = crypto.randomUUID()
      connectWS(sid)

      workletNode.port.onmessage = (evt) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
          wsRef.current.send(evt.data)
        }
      }

    } catch (err) {
      setError(err.message || 'Failed to start recording')
      setStatus('error')
    }
  }, [status, connectWS])

  const stopRecording = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'stop' }))
      wsRef.current.close()
    }
    wsRef.current = null

    workletNodeRef.current?.disconnect()
    workletNodeRef.current = null

    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null

    audioCtxRef.current?.close()
    audioCtxRef.current = null

    if (workletBlobUrlRef.current) {
      URL.revokeObjectURL(workletBlobUrlRef.current)
      workletBlobUrlRef.current = null
    }

    setStatus('stopped')
    setLiveText('')
  }, [])

  const clearTranscript = useCallback(() => {
    setUtterances([])
    setLiveText('')
    setTelemetry([])
  }, [])

  return {
    status,
    sessionId,
    error,
    utterances,
    liveText,
    stats,
    telemetry,
    startRecording,
    stopRecording,
    clearTranscript,
    isRecording: status === 'recording',
    isConnecting: status === 'connecting' || status === 'connected',
  }
}
