class RealTimeTranscription {
    constructor() {
        this.ws = null;
        this.isRecording = false;
        this.sessionId = null;
        this.mediaRecorder = null;
        this.audioContext = null;
        this.microphone = null;
        this.processor = null;
        this.audioChunks = [];
        
        this.initializeElements();
        this.setupEventListeners();
        this.connectWebSocket();
    }
    
    initializeElements() {
        this.recordBtn = document.getElementById('recordBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.transcript = document.getElementById('transcript');
        this.connectionStatus = document.querySelector('.status-dot');
        this.statusText = document.querySelector('.status-text');
        this.sessionIdElement = document.getElementById('sessionId');
        this.vadDot = document.querySelector('.vad-dot');
        this.vadText = document.querySelector('.vad-text');
    }
    
    setupEventListeners() {
        this.recordBtn.addEventListener('click', () => this.toggleRecording());
        this.clearBtn.addEventListener('click', () => this.clearTranscript());
    }
    
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            this.updateConnectionStatus(true);
            console.log('WebSocket connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onclose = () => {
            this.updateConnectionStatus(false);
            console.log('WebSocket disconnected');
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.updateConnectionStatus(false);
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'session_created':
                this.sessionId = data.sessionId;
                this.sessionIdElement.textContent = `Session: ${this.sessionId.substring(0, 8)}...`;
                break;
                
            case 'transcript':
                this.addTranscriptSegment(data.text, data.isFinal);
                this.updateVADIndicator(true);
                break;
                
            case 'transcript_saved':
                this.showTranscriptSaved(data.filename);
                break;
                
            case 'session_ended':
                this.stopRecording();
                this.updateVADIndicator(false);
                break;
        }
    }
    
    updateConnectionStatus(connected) {
        if (connected) {
            this.connectionStatus.classList.add('connected');
            this.statusText.textContent = 'Connected';
        } else {
            this.connectionStatus.classList.remove('connected');
            this.statusText.textContent = 'Disconnected';
        }
    }
    
    updateVADIndicator(isSpeaking) {
        if (isSpeaking) {
            this.vadDot.className = 'vad-dot speaking';
            this.vadText.textContent = 'Voice Activity: Speaking';
        } else {
            this.vadDot.className = 'vad-dot';
            this.vadText.textContent = 'Voice Activity: None';
        }
        
        // Reset to normal after a delay
        if (isSpeaking) {
            setTimeout(() => {
                if (this.vadDot.classList.contains('speaking')) {
                    this.vadDot.className = 'vad-dot active';
                    this.vadText.textContent = 'Voice Activity: Active';
                }
            }, 1000);
        }
    }
    
    async toggleRecording() {
        if (this.isRecording) {
            await this.stopRecording();
        } else {
            await this.startRecording();
        }
    }
    
    async startRecording() {
        try {
            // Request microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });
            
            // Create audio context
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 16000
            });
            
            // Create microphone source
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            
            // Create script processor for real-time processing
            this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            this.processor.onaudioprocess = (event) => {
                if (this.isRecording) {
                    const inputBuffer = event.inputBuffer.getChannelData(0);
                    const audioData = this.convertFloat32ToInt16(inputBuffer);
                    this.sendAudioData(audioData);
                }
            };
            
            // Connect the audio graph
            this.microphone.connect(this.processor);
            this.processor.connect(this.audioContext.destination);
            
            this.isRecording = true;
            this.updateRecordingButton(true);
            
            // Send session start to Python service
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({
                    type: 'start_session',
                    sessionId: this.sessionId
                }));
            }
            
        } catch (error) {
            console.error('Error starting recording:', error);
            alert('Could not access microphone. Please check permissions.');
        }
    }
    
    stopRecording() {
        if (this.processor) {
            this.processor.disconnect();
            this.processor = null;
        }
        
        if (this.microphone) {
            this.microphone.disconnect();
            this.microphone = null;
        }
        
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }
        
        this.isRecording = false;
        this.updateRecordingButton(false);
        
        // Send session end signal
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'end_session'
            }));
        }
        
        this.updateVADIndicator(false);
    }
    
    convertFloat32ToInt16(float32Array) {
        const int16Array = new Int16Array(float32Array.length);
        for (let i = 0; i < float32Array.length; i++) {
            let sample = Math.max(-1, Math.min(1, float32Array[i]));
            int16Array[i] = sample < 0 ? sample * 32768 : sample * 32767;
        }
        return int16Array;
    }
    
    sendAudioData(audioData) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            // Convert Int16Array to buffer
            const buffer = audioData.buffer;
            this.ws.send(buffer);
        }
    }
    
    updateRecordingButton(isRecording) {
        if (isRecording) {
            this.recordBtn.classList.add('recording');
            this.recordBtn.querySelector('.btn-text').textContent = 'Stop Recording';
        } else {
            this.recordBtn.classList.remove('recording');
            this.recordBtn.querySelector('.btn-text').textContent = 'Start Recording';
        }
    }
    
    addTranscriptSegment(text, isFinal = false) {
        const segment = document.createElement('div');
        segment.className = `segment ${isFinal ? 'final' : ''}`;
        segment.textContent = text;
        
        this.transcript.appendChild(segment);
        this.transcript.scrollTop = this.transcript.scrollHeight;
        
        // Remove non-final segments when final version comes
        if (isFinal) {
            const nonFinalSegments = this.transcript.querySelectorAll('.segment:not(.final)');
            nonFinalSegments.forEach(seg => seg.remove());
        }
    }
    
    clearTranscript() {
        this.transcript.innerHTML = '';
    }
    
    showTranscriptSaved(filename) {
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
        
        document.body.appendChild(notification);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new RealTimeTranscription();
});
