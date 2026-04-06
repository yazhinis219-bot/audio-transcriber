# Real-time Transcription System

A professional real-time transcription system built with FastAPI, faster-whisper, and WebSocket communication. This system replicates the original architecture without Redis queuing, using in-memory session management instead.

## Features

- **Real-time audio transcription** using faster-whisper with VAD
- **Professional UI** with dark theme and smooth animations
- **Session management** with unique session IDs
- **Automatic transcript saving** to files
- **Partial and final text** rendering
- **Word count** and status indicators
- **WebSocket communication** for low-latency streaming

## Architecture

```
├── backend/
│   ├── gateway/
│   │   └── main.py              # FastAPI WebSocket server with session management
│   └── worker/
│       └── main.py              # faster-whisper transcription worker
├── public/
│   ├── index.html              # Professional frontend UI
│   ├── app.js                 # Frontend JavaScript with audio recording
│   └── style.css              # Dark theme styling (embedded in HTML)
├── transcripts/                # Saved transcript files
├── config.py                  # Application configuration
├── run_gateway.py             # Gateway startup script
├── run_worker.py              # Worker compatibility script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Key Differences from Original

- **No Redis dependency**: Uses in-memory session management
- **Integrated worker**: Worker functionality is integrated into the gateway process
- **Simplified deployment**: Single process deployment instead of separate gateway/worker
- **Same behavior**: Maintains identical transcription behavior and UI experience

## Setup Instructions

### Prerequisites

- Python (v3.8 or higher)
- FFmpeg (required for faster-whisper)

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the application:**
   ```bash
   python run_gateway.py
   ```

3. **Open your browser** and navigate to:
   ```
   http://localhost:8000
   ```

## Usage

1. **Allow microphone access** when prompted
2. **Click "Start Recording"** to begin transcription
3. **Speak clearly** - the system shows partial and confirmed text
4. **Click "Stop Recording"** when finished
5. **Transcripts are automatically saved** to the `/transcripts` folder

## UI Features

- **Status indicators**: Connection and recording status
- **Session ID**: Unique identifier for each session
- **Partial text**: Live transcription that updates as you speak
- **Final text**: Confirmed portions of the transcript
- **Word count**: Real-time word counting
- **Save notifications**: Alerts when transcripts are saved

## Configuration

The system uses `config.py` for configuration:

### Whisper Settings
- `WHISPER_MODEL_SIZE`: Model size (default: "small.en")
- `WHISPER_DEVICE`: Device type ("cpu" or "cuda")
- `WHISPER_LANGUAGE`: Language (default: "en")
- `WHISPER_VAD_FILTER`: Enable VAD filter (default: true)
- `WHISPER_BEAM_SIZE`: Beam size for decoding (default: 2)

### Audio Settings
- `AUDIO_SAMPLE_RATE`: Sample rate (default: 16000)
- `TRANSCRIBE_EVERY_MS`: Transcription interval (default: 100ms)
- `MIN_TRANSCRIBE_WINDOW_MS`: Minimum audio window (default: 300ms)
- `MAX_BUFFER_SEC`: Maximum buffer duration (default: 20s)

### Environment Variables
- `PORT`: Server port (default: 8000)
- `RELOAD`: Enable auto-reload (default: 0)

## Transcription Behavior

The system provides:
- **Ultra-low latency**: 100ms transcription intervals
- **Smart buffering**: Maintains 20-second rolling buffer
- **VAD filtering**: Built-in voice activity detection
- **Partial updates**: Live text updates during speech
- **Final confirmation**: Stable text for confirmed portions
- **Automatic saving**: Transcripts saved on session end

## File Structure

### Transcripts
Saved transcripts include:
- Session ID
- Timestamp
- Full transcript text
- File naming: `{session_id}.txt`

### Session Management
- In-memory session storage
- Automatic cleanup on disconnect
- Support for concurrent sessions
- WebSocket-based communication

## Development

### Running in Development Mode
```bash
RELOAD=1 python run_gateway.py
```

### Health Check
Visit `http://localhost:8000/health` to see system status.

## Dependencies

### Python
- `fastapi[all]` - Web framework and WebSocket support
- `uvicorn[standard]` - ASGI server
- `faster-whisper` - Optimized Whisper implementation
- `numpy` - Audio processing
- `python-multipart` - Form data handling
- `websockets` - WebSocket client library

## Troubleshooting

### Microphone Issues
- Check browser permissions
- Ensure microphone is not muted
- Try refreshing the page

### Model Loading Issues
- Ensure FFmpeg is installed and in PATH
- Check available disk space (models are ~150MB)
- Verify internet connection for first-time download

### Performance Issues
- Use GPU with `WHISPER_DEVICE=cuda`
- Reduce `TRANSCRIBE_EVERY_MS` for lower latency
- Adjust `WHISPER_MODEL_SIZE` for speed/accuracy tradeoff

## Performance

### CPU Usage
- Optimized for CPU inference
- Throttled transcription to prevent 100% CPU usage
- Smart audio buffering to prevent memory issues

### Latency
- 100ms transcription intervals
- WebSocket communication for minimal delay
- Efficient audio processing pipeline

### Memory
- 20-second rolling audio buffer
- Automatic session cleanup
- Efficient numpy array handling

## License

MIT License
