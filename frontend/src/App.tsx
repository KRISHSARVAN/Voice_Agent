import { useEffect, useRef, useState } from 'react'
import './App.css'

const SESSION_STORAGE_KEY = 'suvit_chat_session_id'

// Voice-activity detection tuning
const SILENCE_THRESHOLD = 0.015  // RMS level below which we consider silence
const SILENCE_DURATION_MS = 1800 // consecutive ms of silence → auto-stop
const MIN_SPEECH_MS = 400        // don't trigger silence detection in first N ms
const MIN_RECORDING_MS = 600     // discard recordings shorter than this (avoids sending noise/empty audio)
const MAX_RECORDING_MS = 30000   // hard cap — force-stop recording after this (safety net)

// Barge-in detection tuning
const BARGE_IN_THRESHOLD = 0.025 // RMS level to trigger barge-in (slightly above silence threshold)
const BARGE_IN_CONFIRM_MS = 250  // ms of continuous speech required to confirm barge-in

// Farewell phrases that should end the call after the bot responds.
// Also covers common STT mis-transcriptions (e.g. "good buy" for "goodbye").
const GOODBYE_PATTERNS = [
  /\b(bye|goodbye|good\s*b[uy]e?|see\s+you|see\s+ya|take\s+care|that'?s?\s+all|end\s+call|hang\s+up|talk\s+later|thanks?\s+bye|have\s+a\s+good\s+(day|night)|have\s+a\s+great\s+(day|night)|farewell|ciao|ttyl|later\s+then|i'?m\s+done)\b/i,
  /\b(thank\s+you|thanks|thank\s+u|thankyou|thx)\b/i,                           // "Thank you" closes the call
  /\b(alvida|shukriya|dhanyavaad|band\s+karo|bas\s+kar|theek\s+hai\s+bye)\b/i,  // Hindi farewells
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

/**
 * Convert any audio Blob (e.g. WebM/Opus from MediaRecorder) to a 16-bit
 * mono 16 kHz WAV Blob that Sarvam Saaras v3 accepts.
 * Uses the Web Audio API to decode → resample → re-encode as raw PCM WAV.
 */
async function blobToWav(blob: Blob): Promise<Blob> {
  const arrayBuffer = await blob.arrayBuffer()
  // Decode and resample to 16 kHz in one step
  const ctx = new AudioContext({ sampleRate: 16000 })
  let audioBuffer: AudioBuffer
  try {
    audioBuffer = await ctx.decodeAudioData(arrayBuffer)
  } finally {
    ctx.close().catch(() => {})
  }

  // Mix down to mono using channel 0
  const pcm = audioBuffer.getChannelData(0)
  const numSamples = pcm.length
  const wavBuffer = new ArrayBuffer(44 + numSamples * 2)
  const view = new DataView(wavBuffer)

  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i))
  }

  const sampleRate = 16000
  writeStr(0, 'RIFF')
  view.setUint32(4, 36 + numSamples * 2, true)
  writeStr(8, 'WAVE')
  writeStr(12, 'fmt ')
  view.setUint32(16, 16, true)       // PCM chunk size
  view.setUint16(20, 1, true)        // PCM format
  view.setUint16(22, 1, true)        // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)  // byteRate
  view.setUint16(32, 2, true)        // blockAlign
  view.setUint16(34, 16, true)       // bitsPerSample
  writeStr(36, 'data')
  view.setUint32(40, numSamples * 2, true)

  let offset = 44
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }

  return new Blob([wavBuffer], { type: 'audio/wav' })
}

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

  // Barge-in resources
  const bargeInCtxRef = useRef<AudioContext | null>(null)
  const bargeInFrameRef = useRef<number | null>(null)
  const bargeInStreamRef = useRef<MediaStream | null>(null)
  const bargingInRef = useRef(false)

  // Ref to the active pipeline stream reader so barge-in can cancel it
  const currentReaderRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null)

  // AbortController for the active fetch so we can signal the server on interrupt
  const abortCtrlRef = useRef<AbortController | null>(null)

  // Track scheduled AudioBufferSourceNodes so we can .stop() them immediately
  const activeSourcesRef = useRef<AudioBufferSourceNode[]>([])

  // Prevent concurrent pipeline runs that cause audio overlap
  const pipelineRunningRef = useRef(false)

  // Generation counter: each runPipeline increments this. Stale pipelines
  // check their captured generation before touching shared state and bail out
  // if a newer pipeline has started.
  const pipelineGenRef = useRef(0)

  // Safety-net timer to force-stop recording if silence detection stalls
  const maxRecordTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    callActiveRef.current = callActive
  }, [callActive])

  // ── Helpers ──────────────────────────────────────────────────────────────

  const rememberSession = (id: string) => {
    sessionStorage.setItem(SESSION_STORAGE_KEY, id)
    setSessionId(id)
  }

  const stopAudio = () => {
    for (const src of activeSourcesRef.current) {
      try { src.stop() } catch { /* already stopped */ }
    }
    activeSourcesRef.current = []
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

  /**
   * Stop barge-in VAD loop and free its AudioContext.
   * Pass keepStream=true when the mic stream will be reused for recording.
   */
  const stopBargeInMonitor = (keepStream = false) => {
    if (bargeInFrameRef.current !== null) {
      cancelAnimationFrame(bargeInFrameRef.current)
      bargeInFrameRef.current = null
    }
    bargeInCtxRef.current?.close().catch(() => {})
    bargeInCtxRef.current = null
    if (!keepStream && bargeInStreamRef.current) {
      bargeInStreamRef.current.getTracks().forEach((t) => t.stop())
      bargeInStreamRef.current = null
    }
  }

  /** Cancel the active HTTP stream reader, abort fetch, and stop audio playback. */
  const cancelCurrentPlayback = () => {
    // Abort the fetch first — this signals the server to stop LLM + TTS work
    if (abortCtrlRef.current) {
      abortCtrlRef.current.abort()
      abortCtrlRef.current = null
    }
    if (currentReaderRef.current) {
      currentReaderRef.current.cancel().catch(() => {})
      currentReaderRef.current = null
    }
    stopAudio()
  }

  // ── Call lifecycle ────────────────────────────────────────────────────────

  const clearMaxRecordTimer = () => {
    if (maxRecordTimerRef.current !== null) {
      clearTimeout(maxRecordTimerRef.current)
      maxRecordTimerRef.current = null
    }
  }

  const endCall = () => {
    callActiveRef.current = false
    setCallActive(false)
    bargingInRef.current = false
    pipelineRunningRef.current = false
    cancelCurrentPlayback()
    stopSilenceDetection()
    clearMaxRecordTimer()
    stopBargeInMonitor()
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

  const runPipeline = async (blob: Blob, _mimeType?: string) => {
    if (!callActiveRef.current) return

    // Kill any leftover audio from a previous pipeline run
    cancelCurrentPlayback()
    pipelineRunningRef.current = true

    // Capture a generation number — if a newer pipeline starts while this
    // one is still awaiting async work, the stale pipeline will see a
    // mismatched gen and bail out instead of stomping on shared state.
    const gen = ++pipelineGenRef.current
    const isStale = () => pipelineGenRef.current !== gen

    setPhase('processing')
    setError(null)

    let farewell = false

    try {
      // 1. Transcribe
      let userText = ''
      let detectedLang = 'en-IN'
      try {
        let uploadBlob: Blob
        try {
          uploadBlob = await blobToWav(blob)
        } catch {
          if (!isStale()) onPipelineDone()
          return
        }
        if (uploadBlob.size < 1024) {
          if (!isStale()) onPipelineDone()
          return
        }
        const form = new FormData()
        form.append('file', uploadBlob, 'recording.wav')
        const r = await fetch('/v1/transcribe', { method: 'POST', body: form })
        if (isStale()) return
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
          if (!isStale()) onPipelineDone()
          return
        }
      } catch (e) {
        if (!isStale()) {
          setError(e instanceof Error ? e.message : 'Transcription failed')
          onPipelineDone()
        }
        return
      }

      // 2+3. Voice pipeline: LLM streams → TTS per sentence → audio plays.
      if (!callActiveRef.current || isStale()) return
      try {
        const abortCtrl = new AbortController()
        abortCtrlRef.current = abortCtrl
        const r = await fetch('/v1/voice', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: [{ role: 'user', content: userText }],
            session_id: sessionId || null,
            include_sources: false,
            language_code: detectedLang,
          }),
          signal: abortCtrl.signal,
        })
        if (isStale()) return
        if (!r.ok || !r.body) {
          const data = (await r.json().catch(() => ({}))) as { detail?: unknown }
          throw new Error(
            typeof data.detail === 'string' ? data.detail : 'Voice pipeline failed',
          )
        }

        stopAudio()

        const ctx = new AudioContext()
        audioCtxRef.current = ctx
        activeSourcesRef.current = []
        let nextStartTime = ctx.currentTime
        setPhase('speaking')

        void startBargeInMonitor()

        const decodeAndQueue = async (b64line: string): Promise<void> => {
          const binary = atob(b64line)
          const bytes = new Uint8Array(binary.length)
          for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
          const audioBuf = await ctx.decodeAudioData(bytes.buffer.slice(0))
          if (ctx.state === 'closed' || isStale()) return
          const source = ctx.createBufferSource()
          source.buffer = audioBuf
          source.connect(ctx.destination)
          const startAt = Math.max(ctx.currentTime, nextStartTime)
          source.start(startAt)
          nextStartTime = startAt + audioBuf.duration
          activeSourcesRef.current.push(source)
          source.onended = () => {
            activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== source)
          }
        }

        const reader = r.body.getReader()
        currentReaderRef.current = reader
        const decoder = new TextDecoder()
        let buf = ''

        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          if (!callActiveRef.current || bargingInRef.current || isStale()) {
            reader.cancel().catch(() => {})
            currentReaderRef.current = null
            break
          }
          buf += decoder.decode(value, { stream: true })
          const lines = buf.split('\n')
          buf = lines.pop() ?? ''
          for (const line of lines) {
            const trimmed = line.trim()
            if (!trimmed) continue
            if (trimmed.startsWith('data:')) {
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

        currentReaderRef.current = null
        abortCtrlRef.current = null

        // Stale pipeline — a newer one took over, don't touch shared state
        if (isStale()) return

        // Barge-in happened — handleBargeIn already started recording
        if (bargingInRef.current) {
          stopAudio()
          stopBargeInMonitor()
          return
        }

        // Process remaining buffered content
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

        if (isStale()) return

        const remaining = (nextStartTime - ctx.currentTime) * 1000
        if (remaining > 0 && callActiveRef.current && !isStale()) {
          await new Promise<void>((res) => setTimeout(res, remaining + 150))
        }
        if (isStale()) return

        activeSourcesRef.current = []
        if (ctx.state !== 'closed') await ctx.close()
        audioCtxRef.current = null

        stopBargeInMonitor()
        onPipelineDone(farewell)
      } catch (e) {
        if (isStale()) return
        abortCtrlRef.current = null
        stopBargeInMonitor()
        if (!bargingInRef.current) {
          if (e instanceof DOMException && e.name === 'AbortError') {
            onPipelineDone(farewell)
          } else {
            setError(e instanceof Error ? e.message : 'Voice pipeline failed')
            onPipelineDone(farewell)
          }
        }
      }
    } finally {
      if (!isStale()) pipelineRunningRef.current = false
    }
  }

  // ── Recording ─────────────────────────────────────────────────────────────

  const recordStartRef = useRef<number>(0)

  const stopRecording = async () => {
    stopSilenceDetection()
    clearMaxRecordTimer()
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

    const elapsed = Date.now() - recordStartRef.current
    if (blob.size > 0 && elapsed >= MIN_RECORDING_MS) {
      await runPipeline(blob, mimeType)
    } else {
      onPipelineDone()
    }
  }

  /**
   * Core recording function. Accepts an already-opened MediaStream so barge-in
   * can hand off its mic stream without re-prompting for permission.
   * @param stream       Live mic MediaStream to record from.
   * @param hasSpeechNow Pass true when the user is already speaking (barge-in case)
   *                     so silence detection doesn't wait for the first utterance.
   */
  const startListeningWithStream = (stream: MediaStream, hasSpeechNow = false) => {
    if (!callActiveRef.current) {
      stream.getTracks().forEach((t) => t.stop())
      return
    }
    setError(null)

    // Safety-net: force-stop recording after MAX_RECORDING_MS
    clearMaxRecordTimer()
    let stopped = false
    maxRecordTimerRef.current = setTimeout(() => {
      if (!stopped) {
        stopped = true
        void stopRecording()
      }
    }, MAX_RECORDING_MS)

    // ── Silence-detection via Web Audio AnalyserNode ─────────────────────
    const silCtx = new AudioContext()
    silenceCtxRef.current = silCtx
    const micSource = silCtx.createMediaStreamSource(stream)
    const analyser = silCtx.createAnalyser()
    analyser.fftSize = 512
    micSource.connect(analyser)

    const bufLen = analyser.frequencyBinCount
    const data = new Float32Array(bufLen)
    let hasSpeech = hasSpeechNow
    let silenceStart: number | null = null
    const recordStart = Date.now()

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
          clearMaxRecordTimer()
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
    recordStartRef.current = Date.now()
    setPhase('recording')
  }

  /**
   * Start microphone recording with automatic silence detection.
   * When silence is detected after the user speaks, recording stops and
   * the pipeline is triggered automatically.
   */
  const startListening = async () => {
    if (!callActiveRef.current) return

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      })
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

    startListeningWithStream(stream)
  }

  // ── Barge-in (interrupt bot while it is speaking) ─────────────────────────

  /**
   * Triggered when the barge-in VAD confirms the user is speaking.
   * Cancels the current bot response, hands the open mic stream to the
   * normal recording path, and starts listening immediately.
   */
  const handleBargeIn = () => {
    if (!callActiveRef.current) return
    const stream = bargeInStreamRef.current
    if (!stream) return

    // Stop bot playback and HTTP stream
    cancelCurrentPlayback()

    // Stop barge-in VAD loop but keep the mic stream open for recording
    stopBargeInMonitor(/* keepStream= */ true)
    bargeInStreamRef.current = null

    // Start recording — user is already speaking, so skip initial silence wait
    startListeningWithStream(stream, /* hasSpeechNow= */ true)
  }

  /**
   * Open a background mic listener that watches for barge-in while the bot
   * is speaking.  Stops automatically once barge-in is confirmed or the
   * speaking phase ends (stopBargeInMonitor called externally).
   */
  const startBargeInMonitor = async () => {
    if (!callActiveRef.current) return
    bargingInRef.current = false

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      })
    } catch {
      return // Mic unavailable — barge-in simply won't work this turn
    }

    if (!callActiveRef.current) {
      stream.getTracks().forEach((t) => t.stop())
      return
    }

    bargeInStreamRef.current = stream

    const ctx = new AudioContext()
    bargeInCtxRef.current = ctx
    const micSource = ctx.createMediaStreamSource(stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 512
    micSource.connect(analyser)

    const bufLen = analyser.frequencyBinCount
    const data = new Float32Array(bufLen)
    let speechStart: number | null = null

    const check = () => {
      if (!callActiveRef.current || bargingInRef.current) return
      analyser.getFloatTimeDomainData(data)
      const rms = Math.sqrt(data.reduce((s, v) => s + v * v, 0) / bufLen)

      if (rms > BARGE_IN_THRESHOLD) {
        if (speechStart === null) speechStart = Date.now()
        else if (Date.now() - speechStart >= BARGE_IN_CONFIRM_MS) {
          // Confirmed — set flag first so the loop stops re-entering
          bargingInRef.current = true
          handleBargeIn()
          return
        }
      } else {
        speechStart = null
      }

      bargeInFrameRef.current = requestAnimationFrame(check)
    }
    bargeInFrameRef.current = requestAnimationFrame(check)
  }

  // ── Begin Call ────────────────────────────────────────────────────────────

  const playWelcomeGreeting = async () => {
    const welcomeText = 'Welcome to Suvit customer support! How can I help you?'
    try {
      const abortCtrl = new AbortController()
      abortCtrlRef.current = abortCtrl
      const r = await fetch('/v1/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: welcomeText, language_code: 'en-IN' }),
        signal: abortCtrl.signal,
      })
      if (!r.ok || !r.body) return

      const ctx = new AudioContext()
      audioCtxRef.current = ctx
      activeSourcesRef.current = []
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
        activeSourcesRef.current.push(source)
        source.onended = () => {
          activeSourcesRef.current = activeSourcesRef.current.filter((s) => s !== source)
        }
      }

      const reader = r.body.getReader()
      currentReaderRef.current = reader
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
      currentReaderRef.current = null
      abortCtrlRef.current = null
      if (buf.trim() && callActiveRef.current) await decodeAndQueue(buf.trim())

      const remaining = (nextStartTime - ctx.currentTime) * 1000
      if (remaining > 0 && callActiveRef.current) {
        await new Promise<void>((res) => setTimeout(res, remaining + 150))
      }
      activeSourcesRef.current = []
      if (ctx.state !== 'closed') await ctx.close()
      audioCtxRef.current = null
    } catch {
      // greeting failed or was aborted — proceed to listening anyway
      abortCtrlRef.current = null
      currentReaderRef.current = null
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
      bargingInRef.current = false
      stopSilenceDetection()
      clearMaxRecordTimer()
      // Abort any in-flight fetch and cancel reader
      abortCtrlRef.current?.abort()
      abortCtrlRef.current = null
      currentReaderRef.current?.cancel().catch(() => {})
      currentReaderRef.current = null
      // Stop all scheduled audio sources immediately
      for (const src of activeSourcesRef.current) {
        try { src.stop() } catch { /* already stopped */ }
      }
      activeSourcesRef.current = []
      // Stop barge-in monitor
      if (bargeInFrameRef.current !== null) cancelAnimationFrame(bargeInFrameRef.current)
      bargeInCtxRef.current?.close().catch(() => {})
      bargeInStreamRef.current?.getTracks().forEach((t) => t.stop())
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
    speaking: 'Agent Speaking… (speak to interrupt)',
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
