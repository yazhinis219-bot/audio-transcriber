import { useRef, useEffect } from 'react'
import { useTranscription } from './hooks/useTranscription'
import './App.css'

function StatusBadge({ status }) {
  const map = {
    idle:       { label: 'Ready',       color: '#005EB8' },
    connecting: { label: 'Connecting…', color: '#ffb81c' },
    connected:  { label: 'Starting…',   color: '#ffb81c' },
    recording:  { label: 'Live',        color: '#DA291C' },
    stopped:    { label: 'Stopped',     color: '#425563' },
    error:      { label: 'Error',       color: '#DA291C' },
  }
  const { label, color } = map[status] || map.idle
  const pulse = status === 'recording'

  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
      <span style={{
        width: 10, height: 10, borderRadius: '50%',
        background: color,
        animation: pulse ? 'pulse 1.4s ease-in-out infinite' : 'none',
      }} />
      <span style={{ fontSize: 13, fontWeight: 500, color }}>{label}</span>
    </span>
  )
}



export default function App() {
  const {
    status, sessionId, error,
    utterances, liveText, stats,
    startRecording, stopRecording, clearTranscript,
    isRecording, isConnecting,
  } = useTranscription()

  const transcriptEndRef = useRef(null)

  // Auto-scroll to latest utterance
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [utterances, liveText])

  let fullText = utterances.map(u => u.text).join(' ')

  let displayLiveText = liveText;

  return (
    <div style={{
      minHeight: '100vh',
      background: '#f0f4f5',
      fontFamily: '"Frutiger", "Arial", sans-serif',
    }}>
      {/* NHS Header */}
      <header style={{
        background: '#005EB8',
        color: 'white',
        padding: '0 24px',
        height: 56,
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
      }}>
        <svg width="48" height="24" viewBox="0 0 48 24" fill="none">
          <rect width="48" height="24" rx="2" fill="white"/>
          <text x="24" y="17" textAnchor="middle" fill="#005EB8" fontSize="13" fontWeight="bold">NHS</text>
        </svg>
        <span style={{ fontSize: 18, fontWeight: 600 }}>Online Consultation</span>
        <span style={{ marginLeft: 'auto', fontSize: 13, opacity: 0.8 }}>
          Live Transcription (No Docker/Queue)
        </span>
      </header>

      <main style={{ maxWidth: 900, margin: '0 auto', padding: '24px 16px' }}>

        {/* Control panel */}
        <div style={{
          background: 'white',
          borderRadius: 8,
          padding: 20,
          marginBottom: 16,
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>

            {/* Mic button */}
            {!isRecording && !isConnecting ? (
              <button onClick={startRecording} style={{
                background: '#007f3b',
                color: 'white',
                border: 'none',
                borderRadius: 6,
                padding: '10px 20px',
                fontSize: 15,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}>
                <MicIcon /> Start Consultation
              </button>
            ) : isConnecting ? (
              <button disabled style={{
                background: '#ccc',
                color: '#555',
                border: 'none',
                borderRadius: 6,
                padding: '10px 20px',
                fontSize: 15,
                fontWeight: 600,
                cursor: 'not-allowed',
              }}>
                Connecting…
              </button>
            ) : (
              <button onClick={stopRecording} style={{
                background: '#DA291C',
                color: 'white',
                border: 'none',
                borderRadius: 6,
                padding: '10px 20px',
                fontSize: 15,
                fontWeight: 600,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
              }}>
                <StopIcon /> Stop Recording
              </button>
            )}

            <StatusBadge status={status} />

            {utterances.length > 0 && (
              <button onClick={clearTranscript} style={{
                background: 'transparent',
                color: '#005EB8',
                border: '1px solid #005EB8',
                borderRadius: 6,
                padding: '8px 14px',
                fontSize: 13,
                cursor: 'pointer',
                marginLeft: 'auto',
              }}>
                Clear
              </button>
            )}
          </div>

          {error && (
            <div style={{
              marginTop: 12,
              padding: '10px 14px',
              background: '#fde8e7',
              border: '1px solid #DA291C',
              borderRadius: 4,
              color: '#DA291C',
              fontSize: 13,
            }}>
              {error}
            </div>
          )}

          {/* Stats bar */}
          {isRecording && (
            <div style={{
              marginTop: 14,
              display: 'flex',
              gap: 20,
              fontSize: 12,
              color: '#768692',
              flexWrap: 'wrap',
            }}>
              <span>Chunks: <b style={{ color: '#212b32' }}>{stats.chunksProcessed}</b></span>
              <span>Transcripts: <b style={{ color: '#212b32' }}>{stats.transcriptsEmitted}</b></span>
              {stats.lastInferenceMs > 0 && (
                <span>Last inference: <b style={{ color: '#212b32' }}>{stats.lastInferenceMs}ms</b></span>
              )}
              {sessionId && (
                <span style={{ marginLeft: 'auto', fontFamily: 'monospace', opacity: 0.6 }}>
                  {sessionId.substring(0, 8)}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Transcript panel */}
        <div style={{
          background: 'white',
          borderRadius: 8,
          padding: 20,
          boxShadow: '0 1px 3px rgba(0,0,0,0.1)',
          minHeight: 300,
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 16,
            paddingBottom: 12,
            borderBottom: '1px solid #e8edee',
          }}>
            <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600, color: '#212b32' }}>
              Consultation Transcript
            </h2>
            {utterances.length > 0 && (
              <button
                onClick={() => navigator.clipboard.writeText(fullText)}
                style={{
                  background: 'transparent',
                  border: '1px solid #bfc9cd',
                  borderRadius: 4,
                  padding: '4px 10px',
                  fontSize: 12,
                  cursor: 'pointer',
                  color: '#425563',
                }}
              >
                Copy all
              </button>
            )}
          </div>

          <div style={{ maxHeight: '60vh', overflowY: 'auto', paddingRight: 4 }}>
            {utterances.length === 0 && !liveText && (
              <div style={{
                textAlign: 'center',
                color: '#768692',
                fontSize: 15,
                paddingTop: 60,
              }}>
                {status === 'idle' || status === 'stopped'
                  ? 'Start recording to begin transcription'
                  : 'Listening…'}
              </div>
            )}

            {/* Display full text as continuous paragraph and inline live text */}
            {(utterances.length > 0 || liveText) && (
              <p style={{
                fontSize: 16,
                color: '#212b32',
                lineHeight: 1.6,
                padding: '10px 14px',
                borderLeft: '3px solid #005EB8',
                background: 'white',
                margin: 0,
                whiteSpace: 'pre-wrap'
              }}>
                {fullText}
                {displayLiveText && (
                  <span style={{ marginLeft: fullText ? '5px' : '0' }}>
                    {(() => {
                      const words = displayLiveText.split(' ');
                      if (words.length <= 4) {
                        return (
                          <span style={{ color: '#79561b', fontStyle: 'italic', backgroundColor: '#fffbf0', padding: '2px 4px', borderRadius: '4px' }}>
                            {displayLiveText}
                          </span>
                        );
                      }
                      const confirmedPart = words.slice(0, -4).join(' ');
                      const unconfirmedPart = ' ' + words.slice(-4).join(' ');
                      return (
                        <>
                          <span style={{ color: '#212b32' }}>{confirmedPart}</span>
                          <span style={{ color: '#79561b', fontStyle: 'italic', backgroundColor: '#fffbf0', padding: '2px 4px', borderRadius: '4px', marginLeft: '4px' }}>
                            {unconfirmedPart}
                          </span>
                        </>
                      );
                    })()}
                  </span>
                )}
                {isRecording && !liveText && utterances.length > 0 && (
                  <span style={{
                    display: 'inline-block',
                    marginLeft: '6px',
                    width: '8px',
                    height: '16px',
                    background: '#005EB8',
                    verticalAlign: 'middle',
                    animation: 'pulse 0.8s infinite alternate',
                    opacity: 0.8
                  }} />
                )}
              </p>
            )}

            <div ref={transcriptEndRef} style={{height: 1}} />
          </div>
          


        </div>

        {/* Instructions */}
        <div style={{
          marginTop: 16,
          padding: '12px 16px',
          background: '#e8edee',
          borderRadius: 6,
          fontSize: 13,
          color: '#425563',
          lineHeight: 1.6,
        }}>
          <b>How it works:</b> Speak naturally. The system handles continuous audio without Docker or Redis.
          Transcripts are shown as a single flowing paragraph, with automatic local rewrites for common misrecognized token endpoints.
        </div>
      </main>
    </div>
  )
}

function MicIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
      <line x1="12" y1="19" x2="12" y2="23"/>
      <line x1="8" y1="23" x2="16" y2="23"/>
    </svg>
  )
}

function StopIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
      <rect x="4" y="4" width="16" height="16" rx="2"/>
    </svg>
  )
}
