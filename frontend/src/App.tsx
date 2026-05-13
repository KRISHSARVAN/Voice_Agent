import { useEffect, useRef, useState } from 'react'
import './App.css'

const SESSION_STORAGE_KEY = 'suvit_chat_session_id'

// Voice-activity detection tuning
const SILENCE_THRESHOLD = 0.015  // RMS level below which we consider silence
const SILENCE_DURATION_MS = 1800 // consecutive ms of silence → auto-stop
const MIN_SPEECH_MS = 400        // don't trigger silence detection in first N ms
const MIN_RECORDING_MS = 600     // discard recordings shorter than this (avoids sending noise/empty audio)
const MAX_RECORDING_MS = 30000   // hard cap — force-stop recording after this (safety net)
/** If the mic never crosses the speech threshold (e.g. silent room or suspended AudioContext), stop and continue. */
const LISTEN_NO_SPEECH_GIVE_UP_MS = 14000

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

/** Encode raw bytes as standard base64 for WebSocket JSON payloads (WAV uploads). */
function bytesToBase64(bytes: Uint8Array): string {
  const chunkSize = 0x8000
  let binary = ''
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize)
    binary += String.fromCharCode(...chunk)
  }
  return btoa(binary)
}

async function unlockAudioContext(ctx: AudioContext): Promise<void> {
  if (ctx.state === 'suspended') {
    await ctx.resume().catch(() => {})
  }
}

/** JSON WS frames are serialized so `done` / errors stay ordered vs tokens. */
type MaybePromiseVoid = void | Promise<void>

function audioGraphIsBroken(ctx: AudioContext): boolean {
  return (ctx.state as string) === 'closed'
}

/**
 * Browsers tie audio playback to a user gesture; any `await` between the
 * click and creating an `AudioContext` can leave it suspended forever.
 * Prime the document with a tiny throwaway context in the same synchronous
 * stack as the button handler, before the first `await`.
 */
function primeAudioFromUserGesture(): void {
  try {
    const c = new AudioContext()
    void c.resume().finally(() => {
      if (c.state !== 'closed') c.close().catch(() => {})
    })
  } catch { /* ignore */ }
}

type Phase = 'idle' | 'recording' | 'processing' | 'speaking'

/** Tracks one in-flight WS server operation (query / transcribe / synthesize). */
type WsAwaitingCallbacks =
  | {
      mode: 'query'
      onAudio: (data: ArrayBuffer) => MaybePromiseVoid
      onToken: (text: string) => void
      resolve: (meta: { sessionId: string; answer: string }) => void
      reject: (e: Error) => void
    }
  | {
      mode: 'transcribe'
      resolve: (r: { text: string; language_code: string }) => void
      reject: (e: Error) => void
    }
  | {
      mode: 'synthesize'
      onAudio: (data: ArrayBuffer) => MaybePromiseVoid
      resolve: () => void
      reject: (e: Error) => void
    }

export default function App() {
  const [sessionId, setSessionId] = useState(
    () => sessionStorage.getItem(SESSION_STORAGE_KEY) ?? '',
  )
  const [phase, setPhase] = useState<Phase>('idle')
  const [callActive, setCallActive] = useState(false)
  const [error, setError] = useState<string | null>(null)
  /** Live transcript from STT — shown after the user finishes speaking. */
  const [lastTranscript, setLastTranscript] = useState<string>('')

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  /** Shared output context for greeting + assistant TTS (one per call, stays running). */
  const audioCtxRef = useRef<AudioContext | null>(null)
  /** Serializes inbound JSON frames (tokens, done, transcribe_done, …). */
  const wsJsonChainRef = useRef(Promise.resolve())
  /**
   * Serializes binary TTS decode + scheduling only — **not** fused with JSON.
   * Fusing blocked `decodeAudioData` delays `done`; the pipeline then never
   * finishes and the UI can sit on Listening / Speaking incorrectly.
   */
  const wsWavChainRef = useRef(Promise.resolve())

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

  // ── WebSocket voice pipeline ──────────────────────────────────────────────
  /** Persistent WS connection kept alive for the duration of a call. */
  const wsRef = useRef<WebSocket | null>(null)
  /**
   * One active awaited WS exchange at a time: RAG+v TTS reply, transcription,
   * or plain TTS (welcome greeting).  Server ``interrupt`` cancels whichever
   * task is running.
   */
  const wsAwaitingCallbacksRef = useRef<WsAwaitingCallbacks | null>(null)
  /**
   * Shared next-start-time cursor for the WS audio queue.
   * Mirrors the local `nextStartTime` that the old HTTP path used, but kept
   * as a ref so the async onmessage handler and runPipeline can both read it.
   */
  const wsNextStartTimeRef = useRef<number>(0)

  useEffect(() => {
    callActiveRef.current = callActive
  }, [callActive])

  // ── Helpers ──────────────────────────────────────────────────────────────

  const rememberSession = (id: string) => {
    sessionStorage.setItem(SESSION_STORAGE_KEY, id)
    setSessionId(id)
  }

  const stopPlaybackSources = () => {
    for (const src of activeSourcesRef.current) {
      try { src.stop() } catch { /* already stopped */ }
    }
    activeSourcesRef.current = []
  }

  const closePlaybackContext = () => {
    stopPlaybackSources()
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {})
      audioCtxRef.current = null
    }
  }

  const ensurePlaybackContext = (): AudioContext => {
    let ctx = audioCtxRef.current
    if (!ctx || audioGraphIsBroken(ctx)) {
      ctx = new AudioContext()
      audioCtxRef.current = ctx
    }
    return ctx
  }

  async function primePlaybackAudio(): Promise<AudioContext> {
    primeAudioFromUserGesture()
    const ctx = ensurePlaybackContext()
    await unlockAudioContext(ctx)
    return ctx
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

  /**
   * Cancel any active playback and tell the server to stop the current
   * transcription, TTS, or query pipeline over the WebSocket.
   */
  const cancelCurrentPlayback = () => {
    if (
      wsRef.current?.readyState === WebSocket.OPEN &&
      wsAwaitingCallbacksRef.current
    ) {
      wsRef.current.send(JSON.stringify({ type: 'interrupt' }))
    }
    stopPlaybackSources()
    wsWavChainRef.current = Promise.resolve()
  }

  // ── WebSocket lifecycle ───────────────────────────────────────────────────

  /**
   * Open a WebSocket connection to /ws/voice.
   * Returns a Promise that resolves once the socket is OPEN (or rejects on
   * error / timeout).  The socket is stored in wsRef and must be closed via
   * endCall / cleanup.
   */
  const setupWebSocket = (): Promise<void> =>
    new Promise((resolve, reject) => {
      // Already connected — reuse
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        resolve()
        return
      }

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/voice`)
      ws.binaryType = 'arraybuffer'

      const timeout = setTimeout(() => {
        ws.close()
        reject(new Error('WebSocket connection timed out'))
      }, 6000)

      ws.onopen = () => {
        clearTimeout(timeout)
        resolve()
      }

      ws.onerror = () => {
        clearTimeout(timeout)
        reject(new Error('WebSocket connection failed'))
      }

      ws.onclose = () => {
        wsRef.current = null
        wsJsonChainRef.current = Promise.resolve()
        wsWavChainRef.current = Promise.resolve()
        const cbs = wsAwaitingCallbacksRef.current
        if (cbs) {
          wsAwaitingCallbacksRef.current = null
          cbs.reject(new Error('WebSocket connection closed'))
        }
      }

      wsJsonChainRef.current = Promise.resolve()
      wsWavChainRef.current = Promise.resolve()

      const enqueueWsWavPlayback = (audioBuf: ArrayBuffer) => {
        const cbsBin = wsAwaitingCallbacksRef.current
        if (
          !cbsBin
          || (cbsBin.mode !== 'query' && cbsBin.mode !== 'synthesize')
        ) {
          return
        }
        const onAudio = cbsBin.onAudio
        wsWavChainRef.current = wsWavChainRef.current
          .then(() => Promise.resolve(onAudio(audioBuf)))
          .catch(() => {})
      }

      const handleWsJsonFrame = async (payload: string): Promise<void> => {
        const cbs = wsAwaitingCallbacksRef.current
        if (!cbs) return

        try {
          const msg = JSON.parse(payload) as Record<string, unknown>
          const type = msg.type as string

          const finishError = () => {
            wsAwaitingCallbacksRef.current = null
            const err = new Error((msg.message as string) ?? 'WebSocket error')
            cbs.reject(err)
          }

          switch (type) {
            case 'token':
              if (cbs.mode === 'query')
                cbs.onToken((msg.text as string) ?? '')
              break

            case 'done':
              if (cbs.mode === 'query') {
                wsAwaitingCallbacksRef.current = null
                cbs.resolve({
                  sessionId: (msg.session_id as string) ?? '',
                  answer: (msg.answer as string) ?? '',
                })
              }
              break

            case 'transcribe_done':
              if (cbs.mode === 'transcribe') {
                wsAwaitingCallbacksRef.current = null
                cbs.resolve({
                  text: ((msg.text as string) ?? '').trim(),
                  language_code: (msg.language_code as string) ?? 'en-IN',
                })
              }
              break

            case 'synthesize_done':
              if (cbs.mode === 'synthesize') {
                wsAwaitingCallbacksRef.current = null
                cbs.resolve()
              }
              break

            case 'interrupted':
              wsAwaitingCallbacksRef.current = null
              if (cbs.mode === 'query') {
                cbs.resolve({ sessionId: '', answer: '' })
              } else if (cbs.mode === 'synthesize') {
                cbs.resolve()
              } else {
                cbs.reject(new Error('Interrupted'))
              }
              break

            case 'error':
              finishError()
              break

            default:
              break
          }
        } catch { /* ignore malformed messages */ }
      }

      ws.onmessage = (event: MessageEvent) => {
        const d = event.data

        if (d instanceof ArrayBuffer) {
          enqueueWsWavPlayback(d)
          return
        }

        if (d instanceof Blob) {
          void d.arrayBuffer().then(enqueueWsWavPlayback)
          return
        }

        if (typeof d !== 'string') return

        wsJsonChainRef.current = wsJsonChainRef.current
          .then(() => handleWsJsonFrame(d))
          .catch(() => {})
      }

      wsRef.current = ws
    })

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
    // Close the WebSocket connection for this call
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    wsAwaitingCallbacksRef.current = null
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== 'inactive'
    ) {
      mediaRecorderRef.current.stop()
      mediaRecorderRef.current.stream?.getTracks().forEach((t) => t.stop())
    }
    audioChunksRef.current = []
    closePlaybackContext()
    setPhase('idle')
    setError(null)
    setLastTranscript('')
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
        const ws = wsRef.current
        if (!ws || ws.readyState !== WebSocket.OPEN) {
          throw new Error('Voice connection unavailable. Please end and restart the call.')
        }
        const buf = await uploadBlob.arrayBuffer()
        const b64 = bytesToBase64(new Uint8Array(buf))
        const transcribeResult = await new Promise<{
          text: string
          language_code: string
        }>((resolve, reject) => {
          wsAwaitingCallbacksRef.current = {
            mode: 'transcribe',
            resolve,
            reject,
          }
          ws.send(
            JSON.stringify({ type: 'transcribe', audio_wav_base64: b64 }),
          )
        })
        if (isStale()) return
        userText = transcribeResult.text
        detectedLang = transcribeResult.language_code
        farewell = isGoodbye(userText)
        setLastTranscript(userText)
        if (!userText) {
          if (!isStale()) onPipelineDone()
          return
        }
      } catch (e) {
        if (!isStale()) {
          const msg = e instanceof Error ? e.message : 'Transcription failed'
          if (msg !== 'Interrupted' && msg !== 'WebSocket connection closed') {
            setError(msg)
          }
          onPipelineDone()
        }
        return
      }

      // 2+3. Voice pipeline via WebSocket: LLM streams tokens → binary WAV
      //      chunks arrive as binary frames → decoded and queued for playback.
      if (!callActiveRef.current || isStale()) return
      try {
        const ws2 = wsRef.current
        if (!ws2 || ws2.readyState !== WebSocket.OPEN) {
          throw new Error('Voice connection unavailable. Please end and restart the call.')
        }

        stopPlaybackSources()

        const ctx = ensurePlaybackContext()
        await unlockAudioContext(ctx)

        wsNextStartTimeRef.current = ctx.currentTime
        setPhase('speaking')

        // Reset barge-in flag and start the background VAD monitor
        bargingInRef.current = false
        void startBargeInMonitor()

        const pipelinePromise = new Promise<{ sessionId: string; answer: string }>(
          (resolve, reject) => {
            wsAwaitingCallbacksRef.current = {
              mode: 'query',
              onAudio: async (raw: ArrayBuffer) => {
                if (audioGraphIsBroken(ctx) || isStale() || bargingInRef.current)
                  return
                try {
                  const audioBuf = await ctx.decodeAudioData(raw.slice(0))
                  if (audioGraphIsBroken(ctx) || isStale()) return
                  const source = ctx.createBufferSource()
                  source.buffer = audioBuf
                  source.connect(ctx.destination)
                  const startAt = Math.max(ctx.currentTime, wsNextStartTimeRef.current)
                  source.start(startAt)
                  wsNextStartTimeRef.current = startAt + audioBuf.duration
                  activeSourcesRef.current.push(source)
                  source.onended = () => {
                    activeSourcesRef.current = activeSourcesRef.current.filter(
                      (s) => s !== source,
                    )
                  }
                } catch {
                  /* bad chunk / stale context — skip */
                }
              },
              onToken: () => {},
              resolve,
              reject,
            }
          },
        )

        ws2.send(
          JSON.stringify({
            type: 'query',
            session_id: sessionId || null,
            messages: [{ role: 'user', content: userText }],
            language_code: detectedLang,
          }),
        )

        const meta = await pipelinePromise

        if (isStale()) return

        // Finish decode+schedule for any TTS chunks that were still in flight when
        // `done` arrived (so timing + cleanup match what you hear).
        await wsWavChainRef.current.catch(() => {})

        // Barge-in happened — handleBargeIn already started the next recording
        if (bargingInRef.current) {
          stopPlaybackSources()
          stopBargeInMonitor()
          return
        }

        if (meta.sessionId) rememberSession(meta.sessionId)
        const answer = meta.answer.trim()
        if (!farewell && isBotFarewell(answer)) farewell = true

        // Wait for all queued audio to finish before moving to next turn
        const remaining = (wsNextStartTimeRef.current - ctx.currentTime) * 1000
        if (remaining > 0 && callActiveRef.current && !isStale()) {
          await new Promise<void>((res) => setTimeout(res, remaining + 150))
        }
        if (isStale()) return

        activeSourcesRef.current = []

        stopBargeInMonitor()
        onPipelineDone(farewell)
      } catch (e) {
        if (isStale()) return
        stopBargeInMonitor()
        if (!bargingInRef.current) {
          setError(e instanceof Error ? e.message : 'Voice pipeline failed')
          onPipelineDone(farewell)
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
  const startListeningWithStream = async (
    stream: MediaStream,
    hasSpeechNow = false,
  ) => {
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
    // Must resume — contexts started after async UX are often suspended, which
    // zeros the analyser so we never detect speech and never leave "Listening".
    await silCtx.resume().catch(() => {})

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

      const elapsedListen = Date.now() - recordStart
      // Never detected speech → don't wait for MAX_RECORDING_MS (30 s)
      if (!hasSpeech && elapsedListen >= LISTEN_NO_SPEECH_GIVE_UP_MS) {
        stopped = true
        clearMaxRecordTimer()
        void stopRecording()
        return
      }

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

    await startListeningWithStream(stream)
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

    // Stop bot playback / interrupt server-side work
    cancelCurrentPlayback()

    // Stop barge-in VAD loop but keep the mic stream open for recording
    stopBargeInMonitor(/* keepStream= */ true)
    bargeInStreamRef.current = null

    // Start recording — user is already speaking, so skip initial silence wait
    void startListeningWithStream(stream, /* hasSpeechNow= */ true).catch(() => {
      if (callActiveRef.current) void startListening()
    })
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
    await ctx.resume().catch(() => {})
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
    const welcomeText =
      'Welcome to Suvit customer support! How can I help you?'
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return

    const ctx = await primePlaybackAudio()
    stopPlaybackSources()
    wsNextStartTimeRef.current = ctx.currentTime
    setPhase('speaking')

    try {
      await new Promise<void>((resolve, reject) => {
        wsAwaitingCallbacksRef.current = {
          mode: 'synthesize',
          onAudio: async (raw: ArrayBuffer) => {
            if (audioGraphIsBroken(ctx) || !callActiveRef.current) return
            try {
              const audioBuf = await ctx.decodeAudioData(raw.slice(0))
              if (audioGraphIsBroken(ctx)) return
              const source = ctx.createBufferSource()
              source.buffer = audioBuf
              source.connect(ctx.destination)
              const startAt = Math.max(ctx.currentTime, wsNextStartTimeRef.current)
              source.start(startAt)
              wsNextStartTimeRef.current = startAt + audioBuf.duration
              activeSourcesRef.current.push(source)
              source.onended = () => {
                activeSourcesRef.current = activeSourcesRef.current.filter(
                  (s) => s !== source,
                )
              }
            } catch {
              /* skip malformed TTS chunk */
            }
          },
          resolve,
          reject,
        }
        ws.send(
          JSON.stringify({
            type: 'synthesize',
            text: welcomeText,
            language_code: 'en-IN',
          }),
        )
      })

      await wsWavChainRef.current.catch(() => {})

      const remaining = (wsNextStartTimeRef.current - ctx.currentTime) * 1000
      if (remaining > 0 && callActiveRef.current) {
        await new Promise<void>((res) => setTimeout(res, remaining + 150))
      }
    } catch {
      // interrupted or aborted — proceed to listening
    }
  }

  const beginCall = async () => {
    callActiveRef.current = true
    setCallActive(true)
    setError(null)
    primeAudioFromUserGesture()
    ensurePlaybackContext()
    void audioCtxRef.current?.resume().catch(() => {})
    // Open the WebSocket before the greeting so it is ready by the first turn
    try {
      await setupWebSocket()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not connect to voice server')
      endCall()
      return
    }
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
      // Close WebSocket and drop any pending pipeline callbacks
      wsRef.current?.close()
      wsRef.current = null
      wsAwaitingCallbacksRef.current = null
      // Stop barge-in monitor / mic helpers
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
      closePlaybackContext()
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

          {/* Live transcript — what you said (STT), after each utterance */}
          {callActive && lastTranscript && (
            <div className="transcript-bubble user">
              <span className="bubble-label">You</span>
              <p>{lastTranscript}</p>
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
