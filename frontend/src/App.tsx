import { useEffect, useRef, useState } from 'react'
import './App.css'

const SESSION_STORAGE_KEY = 'suvit_chat_session_id'

// Voice-activity detection tuning
const SILENCE_THRESHOLD = 0.015  // RMS level below which we consider silence
const SILENCE_DURATION_MS = 1800 // consecutive ms of silence → auto-stop
const MIN_SPEECH_MS = 400        // don't trigger silence detection in first N ms

// Farewell phrases that should end the call after the bot responds.
const GOODBYE_PATTERNS = [
  /\b(bye|goodbye|good\s*b[uy]e?|see\s+you|see\s+ya|take\s+care|that'?s?\s+all|end\s+call|hang\s+up|talk\s+later|thanks?\s+bye|have\s+a\s+good\s+(day|night)|have\s+a\s+great\s+(day|night)|farewell|ciao|ttyl|later\s+then|i'?m\s+done)\b/i,
  /\b(thank\s+you|thanks|thank\s+u|thankyou|thx)\b/i,
  /\b(alvida|shukriya|dhanyavaad|band\s+karo|bas\s+kar|theek\s+hai\s+bye)\b/i,
]

const BOT_GOODBYE_PATTERNS = [
  /\b(goodbye|bye|take\s+care|have\s+a\s+(good|great|nice)\s+(day|night)|see\s+you|farewell|talk\s+(to\s+you\s+)?later|it\s+was\s+(a\s+)?pleasure|happy\s+to\s+help|feel\s+free\s+to\s+(reach|call|contact))\b/i,
  /\b(you'?re\s+welcome|my\s+pleasure|glad\s+(I\s+could|to)\s+help|anytime|no\s+problem)\b/i,
]

const isGoodbye = (text: string) => GOODBYE_PATTERNS.some((re) => re.test(text))
const isBotFarewell = (text: string) => BOT_GOODBYE_PATTERNS.some((re) => re.test(text))

type Phase = 'idle' | 'recording' | 'processing' | 'speaking'

export default function App() {
  const [sessionId, setSessionId] = useState(
    () => sessionStorage.getItem(SESSION_STORAGE_KEY) ?? '',
  )
  const [phase, setPhase] = useState<Phase>('idle')
  const [callActive, setCallActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Ref mirrors callActive so async callbacks always see the latest value
  const callActiveRef = useRef(false)

  // Silence-detection resources
  const silenceCtxRef = useRef<AudioContext | null>(null)
  const animFrameRef = useRef<number | null>(null)

  // AudioContext scheduling for gapless playback
  const nextStartTimeRef = useRef(0)

  useEffect(() => {
    callActiveRef.current = callActive
  }, [callActive])

  // ── Helpers ──────────────────────────────────────────────────────────────

  const rememberSession = (id: string) => {
    sessionStorage.setItem(SESSION_STORAGE_KEY, id)
    setSessionId(id)
  }

  const stopAudio = () => {
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {})
      audioCtxRef.current = null
    }
    nextStartTimeRef.current = 0
  }

  const stopSilenceDetection = () => {
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = null
    }
    silenceCtxRef.current?.close().catch(() => {})
    silenceCtxRef.current = null
  }

  const closeWebSocket = () => {
    if (wsRef.current) {
      wsRef.current.onmessage = null
      wsRef.current.onerror = null
      wsRef.current.onclose = null
      try { wsRef.current.close() } catch { /* ignore */ }
      wsRef.current = null
    }
  }

  // ── Call lifecycle ────────────────────────────────────────────────────────

  const endCall = () => {
    callActiveRef.current = false
    setCallActive(false)
    stopAudio()
    stopSilenceDetection()
    closeWebSocket()
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== 'inactive'
    ) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream?.getTracks().forEach((t) => t.stop())
    }
    setPhase('idle')
    setError(null)
  }

  const startNewChat = () => {
    sessionStorage.removeItem(SESSION_STORAGE_KEY)
    setSessionId('')
    endCall()
  }

  // ── Audio playback helper ────────────────────────────────────────────────

  /**
   * Decode MP3/audio bytes and schedule them for gapless playback.
   * Uses a shared AudioContext and a wall-clock schedule pointer.
   */
  const decodeAndQueue = async (arrayBuffer: ArrayBuffer): Promise<void> => {
    let ctx = audioCtxRef.current
    if (!ctx || ctx.state === 'closed') {
      ctx = new AudioContext()
      audioCtxRef.current = ctx
      nextStartTimeRef.current = ctx.currentTime
    }
    try {
      const audioBuf = await ctx.decodeAudioData(arrayBuffer)
      if (ctx.state === 'closed') return
      const source = ctx.createBufferSource()
      source.buffer = audioBuf
      source.connect(ctx.destination)
      const startAt = Math.max(ctx.currentTime, nextStartTimeRef.current)
      source.start(startAt)
      nextStartTimeRef.current = startAt + audioBuf.duration
    } catch {
      // Ignore decode errors for individual chunks
    }
  }

  // ── WebSocket real-time pipeline ─────────────────────────────────────────

  /**
   * Called when the pipeline finishes (success or error).
   * If endAfter is true (farewell detected), the call ends gracefully.
   * delayMs adds a pause before restarting listening — used after errors
   * to prevent a tight reconnect loop that would flood Vite's WS proxy.
   */
  const onPipelineDone = (endAfter = false, delayMs = 0) => {
    if (endAfter || !callActiveRef.current) {
      endCall()
    } else if (delayMs > 0) {
      setTimeout(() => { if (callActiveRef.current) void startListening() }, delayMs)
    } else {
      void startListening()
    }
  }

  /**
   * Start microphone recording and open a WebSocket to /ws/voice.
   *
   * Audio flow:
   *   MediaRecorder chunks → WebSocket (binary) → Deepgram live STT
   *   On silence → send {"type":"stop"} → LLM + Aura-2 TTS
   *   Server sends binary MP3 chunks back → play via AudioContext
   *   Server sends {"type":"done",...} → restart listening or end call
   */
  const startListening = async () => {
    if (!callActiveRef.current) return
    setError(null)
    setPhase('recording')

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (e) {
      setError(
        e instanceof Error && e.name === 'NotAllowedError'
          ? 'Microphone access denied. Please allow microphone permission.'
          : e instanceof Error
            ? e.message
            : 'Could not access microphone',
      )
      endCall()
      return
    }

    // ── Open WebSocket ────────────────────────────────────────────────────
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const ws = new WebSocket(`${proto}//${window.location.host}/ws/voice`)
    wsRef.current = ws
    ws.binaryType = 'arraybuffer'

    let farewell = false

    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'config', session_id: sessionId || null }))
    }

    ws.onerror = () => {
      setError('WebSocket connection error')
      stopSilenceDetection()
      onPipelineDone(false, 1500)  // delay to avoid tight reconnect loop
    }

    ws.onmessage = async (event) => {
      if (!callActiveRef.current) return

      if (typeof event.data === 'string') {
        try {
          const msg = JSON.parse(event.data) as {
            type?: string
            text?: string
            is_final?: boolean
            session_id?: string
            answer?: string
            detail?: string
          }
          if (msg.type === 'transcript' && msg.is_final && msg.text) {
            if (isGoodbye(msg.text)) farewell = true
            setPhase('processing')
          } else if (msg.type === 'speaking') {
            setPhase('speaking')
          } else if (msg.type === 'done') {
            if (msg.session_id) rememberSession(msg.session_id)
            const answer = (msg.answer ?? '').trim()
            if (!farewell && isBotFarewell(answer)) farewell = true

            // Wait for any buffered audio to finish playing
            const ctx = audioCtxRef.current
            if (ctx && ctx.state !== 'closed') {
              const remaining = (nextStartTimeRef.current - ctx.currentTime) * 1000
              if (remaining > 50 && callActiveRef.current) {
                await new Promise<void>((res) => setTimeout(res, remaining + 150))
              }
            }
            // If no answer was produced (STT got nothing), add a small pause
            // before re-listening to avoid an instant reconnect loop.
            onPipelineDone(farewell, answer ? 0 : 800)
          } else if (msg.type === 'error') {
            setError(msg.detail ?? 'Server error')
            onPipelineDone(false, 1500)  // delay on server-side errors
          }
        } catch { /* ignore malformed JSON */ }
      } else if (event.data instanceof ArrayBuffer) {
        // MP3 audio chunk from Deepgram Aura-2
        setPhase('speaking')
        await decodeAndQueue(event.data)
      }
    }

    ws.onclose = () => {
      wsRef.current = null
    }

    // ── Silence detection via Web Audio AnalyserNode ─────────────────────
    const silCtx = new AudioContext()
    silenceCtxRef.current = silCtx
    const micSource = silCtx.createMediaStreamSource(stream)
    const analyser = silCtx.createAnalyser()
    analyser.fftSize = 512
    micSource.connect(analyser)

    const bufLen = analyser.frequencyBinCount
    const data = new Float32Array(bufLen)
    let hasSpeech = false
    let silenceStart: number | null = null
    const recordStart = Date.now()
    let silenceStopped = false

    const checkSilence = () => {
      if (!callActiveRef.current || silenceStopped) return
      analyser.getFloatTimeDomainData(data)
      const rms = Math.sqrt(data.reduce((sum, v) => sum + v * v, 0) / bufLen)

      if (rms > SILENCE_THRESHOLD) {
        hasSpeech = true
        silenceStart = null
      } else if (hasSpeech && Date.now() - recordStart > MIN_SPEECH_MS) {
        if (silenceStart === null) {
          silenceStart = Date.now()
        } else if (Date.now() - silenceStart > SILENCE_DURATION_MS) {
          silenceStopped = true
          // Signal end of speech over WebSocket
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'stop' }))
          }
          // Stop MediaRecorder and silence detection
          stopSilenceDetection()
          if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
            mediaRecorderRef.current.stop()
            mediaRecorderRef.current.stream?.getTracks().forEach((t) => t.stop())
          }
          return
        }
      }
      animFrameRef.current = requestAnimationFrame(checkSilence)
    }
    animFrameRef.current = requestAnimationFrame(checkSilence)

    // ── MediaRecorder → WebSocket ─────────────────────────────────────────
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : ''
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
    mediaRecorderRef.current = recorder

    recorder.addEventListener('dataavailable', (e) => {
      if (
        e.data.size > 0 &&
        wsRef.current &&
        wsRef.current.readyState === WebSocket.OPEN
      ) {
        e.data.arrayBuffer().then((buf) => {
          if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(buf)
          }
        })
      }
    })

    recorder.start(100) // 100 ms chunks for low latency
  }

  // ── Welcome greeting ─────────────────────────────────────────────────────

  const playWelcomeGreeting = async () => {
    const welcomeText = 'Welcome to Suvit customer support! How can I help you?'
    try {
      const r = await fetch('/v1/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: welcomeText, language_code: 'en-IN' }),
      })
      if (!r.ok || !r.body) return

      const ctx = new AudioContext()
      audioCtxRef.current = ctx
      nextStartTimeRef.current = ctx.currentTime
      setPhase('speaking')

      const reader = r.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (!callActiveRef.current) { reader.cancel().catch(() => {}); break }
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''
        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed) continue
          const binary = atob(trimmed)
          const bytes = new Uint8Array(binary.length)
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
          await decodeAndQueue(bytes.buffer as ArrayBuffer)
        }
      }
      if (buf.trim() && callActiveRef.current) {
        const binary = atob(buf.trim())
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
        await decodeAndQueue(bytes.buffer as ArrayBuffer)
      }

      const remaining = (nextStartTimeRef.current - ctx.currentTime) * 1000
      if (remaining > 0 && callActiveRef.current) {
        await new Promise<void>((res) => setTimeout(res, remaining + 150))
      }
      if (ctx.state !== 'closed') await ctx.close()
      audioCtxRef.current = null
      nextStartTimeRef.current = 0
    } catch {
      // Greeting failed — proceed to listening anyway
    }
  }

  // ── Begin Call ────────────────────────────────────────────────────────────

  const beginCall = async () => {
    callActiveRef.current = true
    setCallActive(true)
    setError(null)
    await playWelcomeGreeting()
    await startListening()
  }

  // ── Cleanup on unmount ───────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      callActiveRef.current = false
      stopSilenceDetection()
      closeWebSocket()
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state !== 'inactive'
      ) {
        mediaRecorderRef.current.stop()
        mediaRecorderRef.current.stream?.getTracks().forEach((t) => t.stop())
      }
      audioCtxRef.current?.close().catch(() => {})
    }
  }, [])

  // ── Labels & derived state ────────────────────────────────────────────────

  const phaseLabel: Record<Phase, string> = {
    idle: 'Ready',
    recording: 'Listening…',
    processing: 'Processing…',
    speaking: 'Agent Speaking…',
  }

  const statusTone = callActive
    ? phase === 'idle'
      ? 'ready'
      : phase
    : 'idle-off'

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="app">
      <main className="voice-stage">
        <section className="voice-card">
          <header className="header">
            <h1>Suvit Voice Agent</h1>
            <p className="subtext">Voice support in one tap</p>
          </header>

          {/* Status indicator */}
          <div className="status-row">
            <span className={`status-dot ${statusTone}`} />
            <span className="status-text">
              {callActive ? phaseLabel[phase] : 'Ready to connect'}
            </span>
          </div>

          {/* Visual call state */}
          {callActive && (
            <div className={`call-visualizer phase-${phase}`}>
              {phase === 'recording' && (
                <div className="pulse-rings">
                  <span /><span /><span />
                </div>
              )}
              {phase === 'speaking' && (
                <div className="wave-bars">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} style={{ animationDelay: `${i * 0.1}s` }} />
                  ))}
                </div>
              )}
              {phase === 'processing' && (
                <div className="spinner" />
              )}
              <p className="phase-label">{phaseLabel[phase]}</p>
            </div>
          )}

          {/* Main action button */}
          {!callActive ? (
            <button
              type="button"
              className="call-btn begin"
              onClick={() => void beginCall()}
              aria-label="Begin Call"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.84 19.84 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6A19.84 19.84 0 0 1 2.12 4.18 2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z" />
              </svg>
              <span>Begin Call</span>
            </button>
          ) : (
            <button
              type="button"
              className="call-btn end"
              onClick={endCall}
              aria-label="End Call"
            >
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M10.68 13.31a16 16 0 0 0 3.41 2.6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7 2 2 0 0 1 1.72 2v3a2 2 0 0 1-2.18 2 19.84 19.84 0 0 1-8.63-3.07A19.84 19.84 0 0 1 2.12 4.18 2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91" />
                <line x1="23" y1="1" x2="1" y2="23" />
              </svg>
              <span>End Call</span>
            </button>
          )}

          {error && (
            <div className="banner error" role="alert">
              {error}
            </div>
          )}

          <button
            type="button"
            className="btn secondary reset-btn"
            onClick={startNewChat}
          >
            New session
          </button>
        </section>
      </main>
    </div>
  )
}
