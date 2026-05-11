import { useEffect, useRef, useState } from 'react'
import './App.css'

const SESSION_STORAGE_KEY = 'suvit_chat_session_id'

// Voice-activity detection tuning
const SILENCE_THRESHOLD = 0.015  // RMS level below which we consider silence
const SILENCE_DURATION_MS = 1800 // consecutive ms of silence → auto-stop
const MIN_SPEECH_MS = 400        // don't trigger silence detection in first N ms

// Farewell phrases that should end the call after the bot responds.
// Also covers common STT mis-transcriptions (e.g. "good buy" for "goodbye").
const GOODBYE_PATTERNS = [
  /\b(bye|goodbye|good\s*b[uy]e?|see\s+you|see\s+ya|take\s+care|that'?s?\s+all|end\s+call|hang\s+up|talk\s+later|thanks?\s+bye|have\s+a\s+good\s+(day|night)|have\s+a\s+great\s+(day|night)|farewell|ciao|ttyl|later\s+then|i'?m\s+done)\b/i,
  /\b(thank\s+you|thanks|thank\s+u|thankyou|thx)\b/i,                           // "Thank you" closes the call
  /\b(alvida|shukriya|dhanyavaad|band\s+karo|bas\s+kar|theek\s+hai\s+bye)\b/i,  // Hindi farewells
  /\b(aavjo|jai\s+shree\s+krishna|jay\s+shree\s+krishna|ram\s+ram|hu\s+jaish|jaish|aavjo\s+bhai|shukriya\s+bhai)\b/i, // Gujarati farewells
]

// Check bot reply too — the LLM often recognises the farewell intent even
// when the STT transcription is imperfect (e.g. "good buy" → "goodbye").
const BOT_GOODBYE_PATTERNS = [
  /\b(goodbye|bye|take\s+care|have\s+a\s+(good|great|nice)\s+(day|night)|see\s+you|farewell|talk\s+(to\s+you\s+)?later|it\s+was\s+(a\s+)?pleasure|happy\s+to\s+help|feel\s+free\s+to\s+(reach|call|contact))\b/i,
  /\b(you'?re\s+welcome|my\s+pleasure|glad\s+(I\s+could|to)\s+help|anytime|no\s+problem)\b/i,
]

const isGoodbye = (text: string) =>
  GOODBYE_PATTERNS.some((re) => re.test(text))

const isBotFarewell = (text: string) =>
  BOT_GOODBYE_PATTERNS.some((re) => re.test(text))

type Phase = 'idle' | 'recording' | 'processing' | 'speaking'

export default function App() {
  const [sessionId, setSessionId] = useState(
    () => sessionStorage.getItem(SESSION_STORAGE_KEY) ?? '',
  )
  const [phase, setPhase] = useState<Phase>('idle')
  const [callActive, setCallActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const audioCtxRef = useRef<AudioContext | null>(null)

  // Ref mirrors callActive so async callbacks always see the latest value
  const callActiveRef = useRef(false)

  // Silence-detection resources
  const silenceCtxRef = useRef<AudioContext | null>(null)
  const animFrameRef = useRef<number | null>(null)

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
  }

  const stopSilenceDetection = () => {
    if (animFrameRef.current !== null) {
      cancelAnimationFrame(animFrameRef.current)
      animFrameRef.current = null
    }
    silenceCtxRef.current?.close().catch(() => {})
    silenceCtxRef.current = null
  }

  // ── Call lifecycle ────────────────────────────────────────────────────────

  const endCall = () => {
    callActiveRef.current = false
    setCallActive(false)
    stopAudio()
    stopSilenceDetection()
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== 'inactive'
    ) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream?.getTracks().forEach((t) => t.stop())
    }
    audioChunksRef.current = []
    setPhase('idle')
    setError(null)
  }

  const startNewChat = () => {
    sessionStorage.removeItem(SESSION_STORAGE_KEY)
    setSessionId('')
    endCall()
  }

  // ── Pipeline: STT → Chat → TTS ───────────────────────────────────────────

  /**
   * Called when pipeline finishes (success or error).
   * If endAfter is true (user said goodbye), the call ends gracefully instead
   * of restarting listening.
   */
  const onPipelineDone = (endAfter = false) => {
    if (endAfter || !callActiveRef.current) {
      endCall()
    } else {
      void startListening()
    }
  }

  const runPipeline = async (blob: Blob, mimeType: string) => {
    if (!callActiveRef.current) return
    setPhase('processing')
    setError(null)

    // 1. Transcribe
    let userText = ''
    let detectedLang = 'en-IN'
    let farewell = false
    try {
      const form = new FormData()
      const ext = mimeType.includes('ogg')
        ? 'ogg'
        : mimeType.includes('mp4')
          ? 'mp4'
          : 'webm'
      form.append('file', blob, `recording.${ext}`)
      const r = await fetch('/v1/transcribe', { method: 'POST', body: form })
      const data = (await r.json().catch(() => ({}))) as {
        text?: string
        language_code?: string
        detail?: unknown
      }
      if (!r.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : r.statusText || 'Transcription failed',
        )
      }
      userText = (data.text ?? '').trim()
      detectedLang = data.language_code ?? 'en-IN'
      farewell = isGoodbye(userText)
      if (!userText) {
        // Nothing heard — go straight back to listening
        onPipelineDone()
        return
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Transcription failed')
      onPipelineDone()
      return
    }

    // 2+3. Real-time voice pipeline — LLM streams tokens → TTS fires per sentence → audio plays.
    // The server sends newline-delimited lines: base64 WAV chunks + a final `data:{…}` JSON line.
    if (!callActiveRef.current) return
    try {
      const r = await fetch('/v1/voice', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [{ role: 'user', content: userText }],
          session_id: sessionId || null,
          include_sources: false,
          language_code: detectedLang,
        }),
      })
      if (!r.ok || !r.body) {
        const data = (await r.json().catch(() => ({}))) as { detail?: unknown }
        throw new Error(
          typeof data.detail === 'string' ? data.detail : 'Voice pipeline failed',
        )
      }

      const ctx = new AudioContext()
      audioCtxRef.current = ctx
      let nextStartTime = ctx.currentTime
      setPhase('speaking')

      const decodeAndQueue = async (b64line: string): Promise<void> => {
        const binary = atob(b64line)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
        const audioBuf = await ctx.decodeAudioData(bytes.buffer.slice(0))
        if (ctx.state === 'closed') return
        const source = ctx.createBufferSource()
        source.buffer = audioBuf
        source.connect(ctx.destination)
        const startAt = Math.max(ctx.currentTime, nextStartTime)
        source.start(startAt)
        nextStartTime = startAt + audioBuf.duration
      }

      const reader = r.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (!callActiveRef.current) {
          reader.cancel().catch(() => {})
          break
        }
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop() ?? ''
        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed) continue
          if (trimmed.startsWith('data:')) {
            // Final metadata line — update session and detect farewell.
            try {
              const meta = JSON.parse(trimmed.slice(5)) as {
                session_id?: string
                answer?: string
              }
              if (meta.session_id) rememberSession(meta.session_id)
              const answer = (meta.answer ?? '').trim()
              if (!farewell && isBotFarewell(answer)) farewell = true
            } catch { /* ignore malformed metadata */ }
          } else {
            await decodeAndQueue(trimmed)
          }
        }
      }
      // Process any remaining buffered content.
      if (buf.trim() && callActiveRef.current) {
        const trimmed = buf.trim()
        if (trimmed.startsWith('data:')) {
          try {
            const meta = JSON.parse(trimmed.slice(5)) as {
              session_id?: string
              answer?: string
            }
            if (meta.session_id) rememberSession(meta.session_id)
            const answer = (meta.answer ?? '').trim()
            if (!farewell && isBotFarewell(answer)) farewell = true
          } catch { /* ignore */ }
        } else {
          await decodeAndQueue(trimmed)
        }
      }

      const remaining = (nextStartTime - ctx.currentTime) * 1000
      if (remaining > 0 && callActiveRef.current) {
        await new Promise<void>((res) => setTimeout(res, remaining + 150))
      }
      if (ctx.state !== 'closed') await ctx.close()
      audioCtxRef.current = null

      onPipelineDone(farewell)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Voice pipeline failed')
      onPipelineDone(farewell)
    }
  }

  // ── Recording ─────────────────────────────────────────────────────────────

  const stopRecording = async () => {
    stopSilenceDetection()
    const recorder = mediaRecorderRef.current
    if (!recorder || recorder.state === 'inactive') return
    recorder.stop()
    await new Promise<void>((resolve) => {
      recorder.addEventListener('stop', () => resolve(), { once: true })
    })
    recorder.stream?.getTracks().forEach((t) => t.stop())
    const mimeType = recorder.mimeType || 'audio/webm'
    const blob = new Blob(audioChunksRef.current, { type: mimeType })
    audioChunksRef.current = []
    if (blob.size > 0) {
      await runPipeline(blob, mimeType)
    } else {
      onPipelineDone()
    }
  }

  /**
   * Start microphone recording with automatic silence detection.
   * When silence is detected after the user speaks, recording stops and
   * the pipeline is triggered automatically.
   */
  const startListening = async () => {
    if (!callActiveRef.current) return
    setError(null)

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

    // ── Silence-detection via Web Audio AnalyserNode ─────────────────────
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
    let stopped = false

    const checkSilence = () => {
      if (!callActiveRef.current || stopped) return
      analyser.getFloatTimeDomainData(data)
      const rms = Math.sqrt(
        data.reduce((sum, v) => sum + v * v, 0) / bufLen,
      )

      if (rms > SILENCE_THRESHOLD) {
        hasSpeech = true
        silenceStart = null
      } else if (hasSpeech && Date.now() - recordStart > MIN_SPEECH_MS) {
        if (silenceStart === null) {
          silenceStart = Date.now()
        } else if (Date.now() - silenceStart > SILENCE_DURATION_MS) {
          stopped = true
          void stopRecording()
          return
        }
      }
      animFrameRef.current = requestAnimationFrame(checkSilence)
    }
    animFrameRef.current = requestAnimationFrame(checkSilence)

    // ── MediaRecorder setup ───────────────────────────────────────────────
    const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : ''
    const recorder = new MediaRecorder(
      stream,
      mimeType ? { mimeType } : undefined,
    )
    mediaRecorderRef.current = recorder
    audioChunksRef.current = []
    recorder.addEventListener('dataavailable', (e) => {
      if (e.data.size > 0) audioChunksRef.current.push(e.data)
    })
    recorder.start(250)
    setPhase('recording')
  }

  // ── Begin Call ────────────────────────────────────────────────────────────

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
      let nextStartTime = ctx.currentTime
      setPhase('speaking')

      const decodeAndQueue = async (b64line: string): Promise<void> => {
        const binary = atob(b64line)
        const bytes = new Uint8Array(binary.length)
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
        const audioBuf = await ctx.decodeAudioData(bytes.buffer.slice(0))
        if (ctx.state === 'closed') return
        const source = ctx.createBufferSource()
        source.buffer = audioBuf
        source.connect(ctx.destination)
        const startAt = Math.max(ctx.currentTime, nextStartTime)
        source.start(startAt)
        nextStartTime = startAt + audioBuf.duration
      }

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
          if (line.trim()) await decodeAndQueue(line.trim())
        }
      }
      if (buf.trim() && callActiveRef.current) await decodeAndQueue(buf.trim())

      const remaining = (nextStartTime - ctx.currentTime) * 1000
      if (remaining > 0 && callActiveRef.current) {
        await new Promise<void>((res) => setTimeout(res, remaining + 150))
      }
      if (ctx.state !== 'closed') await ctx.close()
      audioCtxRef.current = null
    } catch {
      // greeting failed silently — proceed to listening anyway
    }
  }

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
              {/* Phone icon */}
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
              {/* Hang-up icon */}
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
