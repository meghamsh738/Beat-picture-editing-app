import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type React from 'react'
import { get, set } from 'idb-keyval'
import './App.css'

type Clip = {
  id: string
  assetId?: string
  title: string
  track: string
  color: string
  start: number // seconds
  duration: number // seconds
  url?: string
  assetType?: string
  waveform?: number[] | null
  thumb?: string | null
  mediaDuration?: number
  mediaOffset?: number
  gain?: number
  fadeIn?: number
  fadeOut?: number
}

type Marker = { time: number; label: string; color: string }

type Track = { id: string; name: string; type: 'video' | 'audio' }
type TrackState = Track & { mute?: boolean; solo?: boolean; locked?: boolean; height?: 'normal' | 'compact' }

type ProjectState = {
  tracks: TrackState[]
  clips: Clip[]
  markers: Marker[]
}

type SelectionState = {
  clipIds: string[]
  marquee: { x1: number; x2: number; y1: number; y2: number } | null
}

type SnapState = {
  position: number | null
  label: string | null
}

type DragHandle = null | 'loop-start' | 'loop-end'

type SnapPoint = { time: number; label: string }

type ProgramImageMode = 'none' | 'still' | 'poster'

type Asset = {
  id: string
  name: string
  type: string
  duration: number
  url?: string
  waveform?: number[]
  thumb?: string
}

type BeatAnalysis = {
  assetId: string
  beats: number[]
  bpm: number
  createdAt: number
}

type KeyMap = {
  playPause: string
  nudgeLeft: string
  nudgeRight: string
  shuttleBack: string
  shuttleForward: string
  stop: string
}

type Draft = {
  id: string
  name: string
  savedAt: number
  payload: {
    project: ProjectState
    beatAnalyses: Record<string, BeatAnalysis>
  }
}

const DEFAULT_TRACKS: TrackState[] = [
  { id: 'v1', name: 'V1 · Motion', type: 'video', height: 'normal' },
  { id: 'v2', name: 'V2 · Titles', type: 'video', height: 'normal' },
  { id: 'a1', name: 'A1 · Music', type: 'audio', height: 'normal' },
  { id: 'a2', name: 'A2 · Foley', type: 'audio', height: 'normal' }
]

const DEFAULT_CLIPS: Clip[] = [
  { id: 'c1', assetId: 'asset-mp4', title: 'Sample 5s MP4', track: 'v1', color: '#4ade80', start: 0, duration: 5, url: '/samples/sample-5s.mp4', assetType: 'video/mp4', thumb: '/samples/sample-photo.jpg', mediaDuration: 5, gain: 1, fadeIn: 0.12, fadeOut: 0.12 },
  { id: 'c2', title: 'Scene A', track: 'v1', color: '#60a5fa', start: 6.5, duration: 8, gain: 1, fadeIn: 0.12, fadeOut: 0.12 },
  { id: 'c3', title: 'Lower Third', track: 'v2', color: '#f472b6', start: 6.7, duration: 4, gain: 1, fadeIn: 0.12, fadeOut: 0.12 },
  { id: 'c4', assetId: 'asset-tone', title: 'Free Tone', track: 'a1', color: '#fbbf24', start: 2, duration: 10, url: '/samples/free-tone-10s.wav', assetType: 'audio/wav', waveform: Array.from({ length: 90 }, (_, i) => 0.28 + 0.38 * Math.sin(i * 0.34) ** 2), mediaDuration: 10, gain: 1, fadeIn: 0.2, fadeOut: 0.4 },
  { id: 'c5', assetId: 'asset-wav', title: 'Sample WAV', track: 'a2', color: '#a78bfa', start: 4, duration: 3.2, url: '/samples/sample-3s.wav', assetType: 'audio/wav', waveform: Array.from({ length: 72 }, (_, i) => 0.25 + 0.3 * Math.sin(i * 0.28 + 0.4) ** 2), mediaDuration: 3.2, gain: 1, fadeIn: 0.12, fadeOut: 0.12 }
]

const DEFAULT_MARKERS: Marker[] = [
  { time: 2, label: 'Beat drop', color: '#22d3ee' },
  { time: 8.5, label: 'Cut to B-roll', color: '#f97316' },
  { time: 12, label: 'Logo hit', color: '#e11d48' }
]

const DEFAULT_ASSETS: Asset[] = [
  {
    id: 'asset-tone',
    name: 'Free Tone 10s',
    type: 'audio/wav',
    duration: 10,
    url: '/samples/free-tone-10s.wav',
    waveform: Array.from({ length: 96 }, (_, i) => 0.35 + 0.25 * Math.sin(i * 0.42) ** 2)
  },
  {
    id: 'asset-mp4',
    name: 'Sample 5s MP4',
    type: 'video/mp4',
    duration: 5,
    url: '/samples/sample-5s.mp4',
    thumb: '/samples/sample-photo.jpg'
  },
  {
    id: 'asset-mars',
    name: 'Mars Still',
    type: 'image/jpeg',
    duration: 8,
    url: '/samples/mars-1280.jpg',
    thumb: '/samples/mars-1280.jpg'
  },
  {
    id: 'asset-wav',
    name: 'Sample 3s WAV',
    type: 'audio/wav',
    duration: 3.2,
    url: '/samples/sample-3s.wav',
    waveform: Array.from({ length: 90 }, (_, i) => 0.3 + 0.32 * Math.sin(i * 0.33 + 0.6) ** 2)
  }
]

const CLIP_COLORS = ['#4ade80', '#60a5fa', '#f472b6', '#fbbf24', '#a78bfa', '#22d3ee']

const TOTAL_DURATION = 18
const GRID_STEP = 0.25
const MIN_CLIP = 0.2
const ZOOM_MIN = 0.6
const ZOOM_MAX = 3
const VIEW_PADDING_PX = 60

const DEFAULT_SNAP_THRESHOLD = 0.12
const DEFAULT_KEYMAP: KeyMap = {
  playPause: ' ',
  nudgeLeft: 'ArrowLeft',
  nudgeRight: 'ArrowRight',
  shuttleBack: 'j',
  shuttleForward: 'l',
  stop: 'k'
}
const DRAFTS_KEY = 'timeline-builder-drafts'

const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v))

const ensureAudioContext = async (ref: React.MutableRefObject<AudioContext | null>) => {
  if (!ref.current) ref.current = new AudioContext()
  if (ref.current.state === 'suspended') await ref.current.resume()
  return ref.current
}

const isTrackCompatible = (assetType: string, trackType: Track['type']) => {
  if (assetType.startsWith('audio')) return trackType === 'audio'
  if (assetType.startsWith('video') || assetType.startsWith('image')) return trackType === 'video'
  return true
}

const isClipFromAsset = (asset: Asset, clip: Clip) => {
  if (clip.assetId && clip.assetId === asset.id) return true
  if (asset.url && clip.url && clip.url === asset.url) return true
  return clip.title === asset.name && clip.assetType === asset.type
}

const loadMediaDuration = (file: File): Promise<number> => new Promise(resolve => {
  try {
    const url = URL.createObjectURL(file)
    const media = document.createElement(file.type.startsWith('audio') ? 'audio' : 'video')
    media.preload = 'metadata'
    media.src = url
    media.onloadedmetadata = () => {
      const dur = media.duration
      URL.revokeObjectURL(url)
      if (!Number.isFinite(dur)) return resolve(5)
      resolve(clamp(dur, 0.5, 120))
    }
    media.onerror = () => resolve(5)
  } catch {
    resolve(5)
  }
})

const decodeWaveform = async (file: File): Promise<number[] | null> => {
  if (typeof window === 'undefined' || typeof AudioContext === 'undefined') return null
  try {
    const buffer = await file.arrayBuffer()
    const ctx = new AudioContext({ sampleRate: 48000 })
    const audio = await ctx.decodeAudioData(buffer)
    const channelL = audio.getChannelData(0)
    const channelR = audio.numberOfChannels > 1 ? audio.getChannelData(1) : null
    const buckets = Math.min(512, Math.max(96, Math.floor(channelL.length / 4000)))
    const step = Math.max(1, Math.floor(channelL.length / buckets))
    const samplesL: number[] = []
    const samplesR: number[] = []
    for (let i = 0; i < buckets; i++) {
      let sumL = 0
      let sumR = 0
      for (let j = 0; j < step; j++) {
        const idx = i * step + j
        sumL += Math.abs(channelL[idx])
        if (channelR) sumR += Math.abs(channelR[idx])
      }
      samplesL.push(sumL / step)
      if (channelR) samplesR.push(sumR / step)
    }
    const max = Math.max(...samplesL, ...(samplesR.length ? samplesR : [0.001]), 0.001)
    return samplesL.map((s, i) => (s + (samplesR[i] || s)) / (samplesR.length ? 2 : 1) / max)
  } catch (err) {
    console.warn('waveform decode failed', err)
    return null
  }
}

const analyzeBeatsFromBuffer = (audio: AudioBuffer) => {
  const channel = audio.getChannelData(0)
  const hop = 512
  const frame = 1024
  const sr = audio.sampleRate
  const energies: number[] = []
  for (let i = 0; i + frame < channel.length; i += hop) {
    let sum = 0
    for (let j = 0; j < frame; j++) {
      const v = channel[i + j] || 0
      sum += v * v
    }
    energies.push(Math.sqrt(sum / frame))
  }
  const mean = energies.reduce((a, b) => a + b, 0) / Math.max(1, energies.length)
  const variance = energies.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, energies.length)
  const std = Math.sqrt(variance)
  const threshold = mean + std * 0.9
  const beats: number[] = []
  let lastBeat = -1
  energies.forEach((e, idx) => {
    const t = (idx * hop) / sr
    if (e > threshold && e > (energies[idx - 1] || 0) && e >= (energies[idx + 1] || 0)) {
      if (lastBeat < 0 || t - lastBeat > 0.23) {
        beats.push(Number(t.toFixed(3)))
        lastBeat = t
      }
    }
  })
  const bpm = beats.length > 1 ? Math.round(60 * (beats.length - 1) / (beats[beats.length - 1] - beats[0])) : 0
  return { beats, bpm }
}

const loadThumb = (file: File): Promise<string | null> => new Promise(resolve => {
  if (file.type.startsWith('image')) {
    const reader = new FileReader()
    reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null)
    reader.onerror = () => resolve(null)
    reader.readAsDataURL(file)
    return
  }
  if (file.type.startsWith('video')) {
    const url = URL.createObjectURL(file)
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.src = url
    video.muted = true
    video.playsInline = true
    const onError = () => { URL.revokeObjectURL(url); resolve(null) }
    video.onerror = onError
    video.onloadeddata = async () => {
      try {
        video.currentTime = Math.min(0.25, (video.duration || 1) * 0.1)
        await video.play().catch(() => {})
        video.pause()
        const canvas = document.createElement('canvas')
        canvas.width = 320
        canvas.height = Math.max(1, Math.round((video.videoHeight / video.videoWidth) * 320))
        const ctx = canvas.getContext('2d')
        if (!ctx) throw new Error('no ctx')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        const data = canvas.toDataURL('image/jpeg', 0.75)
        URL.revokeObjectURL(url)
        resolve(data)
      } catch {
        URL.revokeObjectURL(url)
        resolve(null)
      }
    }
    return
  }
  resolve(null)
})

const STORAGE_KEY = 'timeline-builder-project-v1'

type HistoryOpts = { push?: boolean }

function useHistoryState<T>(initial: T, limit = 80) {
  const [state, setState] = useState<T>(initial)
  const historyRef = useRef<T[]>([initial])
  const pointerRef = useRef(0)

  const commit = (updater: T | ((prev: T) => T), opts: HistoryOpts = { push: true }) => {
    setState(prev => {
      const next = typeof updater === 'function' ? (updater as (p: T) => T)(prev) : updater
      if (opts.push !== false) {
        const sliced = historyRef.current.slice(0, pointerRef.current + 1)
        const nextHistory = [...sliced, next]
        if (nextHistory.length > limit) nextHistory.shift()
        historyRef.current = nextHistory
        pointerRef.current = historyRef.current.length - 1
      }
      return next
    })
  }

  const pushCheckpoint = () => {
    const current = historyRef.current[pointerRef.current]
    const sliced = historyRef.current.slice(0, pointerRef.current + 1)
    const nextHistory = [...sliced, current]
    if (nextHistory.length > limit) nextHistory.shift()
    historyRef.current = nextHistory
    pointerRef.current = historyRef.current.length - 1
  }

  const undo = () => {
    if (pointerRef.current === 0) return
    pointerRef.current -= 1
    setState(historyRef.current[pointerRef.current])
  }

  const redo = () => {
    if (pointerRef.current >= historyRef.current.length - 1) return
    pointerRef.current += 1
    setState(historyRef.current[pointerRef.current])
  }

  const canUndo = pointerRef.current > 0
  const canRedo = pointerRef.current < historyRef.current.length - 1

  return { state, set: commit, undo, redo, canUndo, canRedo, pushCheckpoint }
}

const formatTime = (t: number) => {
  const mins = Math.floor(t / 60)
  const secs = t % 60
  return `${String(mins).padStart(2, '0')}:${secs.toFixed(2).padStart(5, '0')}`
}

function App() {
  const [activeTab, setActiveTab] = useState<'edit' | 'assets' | 'export'>('edit')
  const [playhead, setPlayhead] = useState(4.5)
  const [zoom, setZoom] = useState(1.4) // multiplier for px/sec
  const [playing, setPlaying] = useState(false)
  const [allowOverlap, setAllowOverlap] = useState(false)
  const [rippleEdit, setRippleEdit] = useState(false)
  const [rollEdit, setRollEdit] = useState(false)
  const [loopEnabled, setLoopEnabled] = useState(false)
  const [loopRange, setLoopRange] = useState<{ start: number; end: number }>({ start: 0, end: 8 })
  const [snapStrength, setSnapStrength] = useState(() => {
    const cached = localStorage.getItem('timeline-snap-threshold')
    return cached ? Number(cached) || DEFAULT_SNAP_THRESHOLD : DEFAULT_SNAP_THRESHOLD
  })
  const [assets, setAssets] = useState<Asset[]>(DEFAULT_ASSETS)
  const [trackHeightScale, setTrackHeightScale] = useState(1)
  const [exportPreset, setExportPreset] = useState<'json' | 'mp4' | 'webm'>('json')
  const [exportStatus, setExportStatus] = useState<string | null>(null)
  const [exportProgress, setExportProgress] = useState(0)
  const exportAbortRef = useRef<{ cancelled: boolean }>({ cancelled: false })
  const [beatAnalyses, setBeatAnalyses] = useState<Record<string, BeatAnalysis>>({})
  const [beatStatus, setBeatStatus] = useState<{ state: 'idle' | 'analyzing' | 'ready' | 'error'; message?: string; assetId?: string }>({ state: 'idle' })
  const [selectedBeatAsset, setSelectedBeatAsset] = useState<string | null>(null)
  const [analysisStatuses, setAnalysisStatuses] = useState<Record<string, 'pending' | 'processing' | 'cached' | 'done' | 'error'>>({})
  const [assetMenu, setAssetMenu] = useState<{ assetId: string; x: number; y: number } | null>(null)
  const pendingFilesRef = useRef<Map<string, File>>(new Map())
  const processingAnalysisRef = useRef(false)
  const analysisStatusRef = useRef<Record<string, 'pending' | 'processing' | 'cached' | 'done' | 'error'>>({})
  const [programImage, setProgramImage] = useState<string | null>(null)
  const [programImageMode, setProgramImageMode] = useState<ProgramImageMode>('none')
  const [programStatus, setProgramStatus] = useState<'idle' | 'video' | 'image' | 'loading' | 'error'>('idle')
  const [programClip, setProgramClip] = useState<{ id: string; label: string } | null>(null)
  const programImageIdRef = useRef<string | null>(null)
  const programClipIdRef = useRef<string | null>(null)
  const programKindRef = useRef<'none' | 'video' | 'image'>('none')
  const [drafts, setDrafts] = useState<Draft[]>([])
  const [activeDraftId, setActiveDraftId] = useState<string | null>(null)
  const [draftName, setDraftName] = useState('')
  const [masterLevel, setMasterLevel] = useState(0)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [keymap, setKeymap] = useState<KeyMap>(() => {
    try {
      const cached = localStorage.getItem('timeline-keymap')
      if (cached) return { ...DEFAULT_KEYMAP, ...JSON.parse(cached) }
    } catch { /* ignore */ }
    return DEFAULT_KEYMAP
  })
  const { state: project, set: setProject, undo, redo, canUndo, canRedo, pushCheckpoint } = useHistoryState<ProjectState>(
    { tracks: DEFAULT_TRACKS, clips: DEFAULT_CLIPS, markers: DEFAULT_MARKERS }
  )
  const { tracks, clips, markers } = project
  const [selection, setSelection] = useState<SelectionState>({ clipIds: [], marquee: null })
  const [snap, setSnap] = useState<SnapState>({ position: null, label: null })
  const loopHandleRef = useRef<DragHandle>(null)

  const audioCtxRef = useRef<AudioContext | null>(null)
  const bufferCacheRef = useRef<Map<string, AudioBuffer>>(new Map())
  const bufferDurRef = useRef<Map<string, number>>(new Map())
  const sourcesRef = useRef<AudioBufferSourceNode[]>([])
  const masterGainRef = useRef<GainNode | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const playheadStartRef = useRef<number>(0)
  const playStartTimeRef = useRef<number>(0)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const activeVideoIdRef = useRef<string | null>(null)
  const assetMenuRef = useRef<HTMLDivElement | null>(null)
  const assetMenuActionRef = useRef<'unassign' | null>(null)

  const timelineRef = useRef<HTMLDivElement | null>(null)
  const [viewWindow, setViewWindow] = useState({ start: 0, duration: 8 })
  const rafRef = useRef<number | null>(null)
  const minimapRef = useRef<HTMLDivElement | null>(null)

  const trackIndex = useMemo(() => Object.fromEntries(tracks.map(t => [t.id, t])), [tracks])
  const trackOrder = useMemo(() => Object.fromEntries(tracks.map((t, idx) => [t.id, idx])), [tracks])
  const anySolo = useMemo(() => tracks.some(t => t.solo), [tracks])
  const heroAsset = useMemo(() => assets.find(a => a.type.startsWith('video') || a.type.startsWith('image')), [assets])
  const assetMenuAsset = useMemo(() => {
    if (!assetMenu) return null
    return assets.find(a => a.id === assetMenu.assetId) || null
  }, [assetMenu, assets])
  const assetMenuClipCount = useMemo(() => {
    if (!assetMenu) return 0
    const asset = assets.find(a => a.id === assetMenu.assetId)
    if (!asset) return 0
    return clips.filter(c => isClipFromAsset(asset, c)).length
  }, [assetMenu, assets, clips])
  const primaryClip = useMemo(() => clips.find(c => c.id === selection.clipIds[0]), [clips, selection.clipIds])
  const sourcePreview = useMemo(() => {
    if (primaryClip) {
      return {
        id: primaryClip.id,
        label: primaryClip.title,
        type: primaryClip.assetType || '',
        thumb: primaryClip.thumb || null,
        waveform: primaryClip.waveform || null
      }
    }
    if (!heroAsset) return null
    return {
      id: heroAsset.id,
      label: heroAsset.name,
      type: heroAsset.type,
      thumb: heroAsset.thumb || null,
      waveform: heroAsset.waveform || null
    }
  }, [primaryClip, heroAsset])

  useEffect(() => {
    if (!selectedBeatAsset) {
      const firstAudio = assets.find(a => a.type.startsWith('audio'))
      if (firstAudio) setSelectedBeatAsset(firstAudio.id)
    }
  }, [assets, selectedBeatAsset])

  useEffect(() => {
    localStorage.setItem('timeline-snap-threshold', String(snapStrength))
  }, [snapStrength])

  type DragInfo = {
    id: string
    mode: 'move' | 'trim-start' | 'trim-end' | 'slip' | 'slide'
    startX: number
    origStart: number
    origDuration: number
  }
  const dragRef = useRef<DragInfo | null>(null)
  const dragStartedState = useRef<ProjectState | null>(null)

  // Restore project from localStorage once
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try {
        const parsed = JSON.parse(saved)
        setProject({
          tracks: parsed.tracks ?? DEFAULT_TRACKS,
          clips: parsed.clips ?? DEFAULT_CLIPS,
          markers: parsed.markers ?? DEFAULT_MARKERS
        })
        if (parsed.beatAnalyses) setBeatAnalyses(parsed.beatAnalyses)
      } catch (err) {
        console.warn('Failed to parse saved project', err)
      }
    }
  }, [setProject, setBeatAnalyses])

  useEffect(() => {
    try {
      const stored = localStorage.getItem(DRAFTS_KEY)
      if (stored) {
        const parsed = JSON.parse(stored) as Draft[]
        setDrafts(parsed)
        if (parsed.length) setActiveDraftId(parsed[0].id)
      }
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem('timeline-keymap', JSON.stringify(keymap))
    } catch { /* ignore */ }
  }, [keymap])

  // Persist project
  useEffect(() => {
    const payload = JSON.stringify({ ...project, beatAnalyses })
    localStorage.setItem(STORAGE_KEY, payload)
  }, [project, beatAnalyses])

  useEffect(() => {
    analysisStatusRef.current = analysisStatuses
  }, [analysisStatuses])

  useEffect(() => {
    if (!assetMenu) return
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null
      if (!target || !assetMenuRef.current) {
        setAssetMenu(null)
        return
      }
      if (assetMenuRef.current.contains(target)) return
      setAssetMenu(null)
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setAssetMenu(null)
    }
    window.addEventListener('pointerdown', handlePointerDown)
    window.addEventListener('keydown', handleKeyDown)
    const raf = window.requestAnimationFrame(() => {
      assetMenuRef.current?.querySelector('button')?.focus()
    })
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown)
      window.removeEventListener('keydown', handleKeyDown)
      window.cancelAnimationFrame(raf)
    }
  }, [assetMenu])

  useEffect(() => {
    if (!assetMenu) {
      assetMenuActionRef.current = null
      return
    }
    if (assetMenuActionRef.current === 'unassign' && assetMenuClipCount === 0) {
      assetMenuActionRef.current = null
      setAssetMenu(null)
    }
  }, [assetMenu, assetMenuClipCount])

  useEffect(() => {
    if (!isScrubbing) return
    const stop = () => setIsScrubbing(false)
    window.addEventListener('pointerup', stop)
    window.addEventListener('pointercancel', stop)
    window.addEventListener('blur', stop)
    return () => {
      window.removeEventListener('pointerup', stop)
      window.removeEventListener('pointercancel', stop)
      window.removeEventListener('blur', stop)
    }
  }, [isScrubbing])

  useEffect(() => {
    const autosave = window.setInterval(() => {
      const draft: Draft = {
        id: 'autosave',
        name: 'Autosave',
        savedAt: Date.now(),
        payload: { project, beatAnalyses }
      }
      setDrafts(prev => {
        const others = prev.filter(d => d.id !== 'autosave')
        const next = [draft, ...others].slice(0, 6)
        localStorage.setItem(DRAFTS_KEY, JSON.stringify(next))
        setActiveDraftId('autosave')
        return next
      })
    }, 90_000)
    return () => window.clearInterval(autosave)
  }, [project, beatAnalyses])

  // playback loop (audio-backed)
  useEffect(() => {
    if (playing) {
      startPlayback()
    } else {
      stopPlayback()
    }
    return () => stopPlayback()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, loopEnabled, loopRange, clips])

  // keep video preview synced while scrubbing
  useEffect(() => {
    syncVideoToPlayhead(playhead)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playhead, clips, anySolo, trackIndex, trackOrder, playing])

  useEffect(() => {
    const vid = videoRef.current
    if (!vid) return
    const onLoadStart = () => {
      if (programKindRef.current !== 'video') return
      if (programStatus !== 'loading') setProgramStatus('loading')
    }
    const onLoaded = () => {
      if (programKindRef.current !== 'video') return
      if (programImageMode === 'poster') {
        setProgramImage(null)
        setProgramImageMode('none')
      }
      if (programStatus !== 'video') setProgramStatus('video')
    }
    const onError = () => {
      if (programKindRef.current !== 'video') return
      setProgramStatus('error')
    }
    vid.addEventListener('loadstart', onLoadStart)
    vid.addEventListener('loadeddata', onLoaded)
    vid.addEventListener('error', onError)
    return () => {
      vid.removeEventListener('loadstart', onLoadStart)
      vid.removeEventListener('loadeddata', onLoaded)
      vid.removeEventListener('error', onError)
    }
  }, [programImageMode, programStatus])

  useEffect(() => {
    let raf: number | null = null
    const data = new Uint8Array(256)
    const tick = () => {
      const analyser = analyserRef.current
      if (analyser) {
        analyser.getByteTimeDomainData(data)
        let sum = 0
        for (let i = 0; i < data.length; i++) {
          const v = (data[i] - 128) / 128
          sum += v * v
        }
        const rms = Math.sqrt(sum / data.length)
        setMasterLevel(rms)
      }
      raf = requestAnimationFrame(tick)
    }
    tick()
    return () => { if (raf) cancelAnimationFrame(raf) }
  }, [])

  // Keyboard: nudge, playback toggles, undo/redo, delete/duplicate
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (target) {
        const tag = target.tagName
        const isFormField = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT'
        if (isFormField || target.isContentEditable) return
      }
      const step = e.shiftKey ? 0.5 : 0.1
      const match = (binding: string) => {
        const key = binding.toLowerCase()
        if (binding === ' ') return e.code === 'Space' || e.key === ' '
        return e.key.toLowerCase() === key
      }
      if (match(keymap.nudgeLeft) || match(keymap.nudgeRight)) {
        setProject(prev => ({
          ...prev,
          clips: prev.clips.map(c => selection.clipIds.includes(c.id)
            ? { ...c, start: Math.max(0, c.start + (match(keymap.nudgeLeft) ? -step : step)) }
            : c)
        }), { push: false })
        pushCheckpoint()
      }
      if (match(keymap.playPause)) {
        e.preventDefault()
        setPlaying(p => !p)
      }
      if (match(keymap.shuttleForward) && !e.metaKey && !e.ctrlKey) setPlayhead(p => clampTime(p + 0.5))
      if (match(keymap.shuttleBack) && !e.metaKey && !e.ctrlKey) setPlayhead(p => clampTime(p - 0.5))
      if (match(keymap.stop) && !e.metaKey && !e.ctrlKey) setPlaying(false)
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (selection.clipIds.length) {
          setProject(prev => ({ ...prev, clips: prev.clips.filter(c => !selection.clipIds.includes(c.id)) }))
          setSelection({ clipIds: [], marquee: null })
          pushCheckpoint()
        }
      }
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'd') {
        e.preventDefault()
        if (!selection.clipIds.length) return
        setProject(prev => {
          const clones: Clip[] = []
          selection.clipIds.forEach((id, idx) => {
            const source = prev.clips.find(c => c.id === id)
            if (!source) return
            const newStart = clampTime(source.start + 0.5 + idx * 0.1)
            clones.push({ ...source, id: `${Date.now()}-${idx}`, start: newStart })
          })
          return { ...prev, clips: [...prev.clips, ...clones] }
        })
        pushCheckpoint()
      }
      if (e.metaKey || e.ctrlKey) {
        if (e.key.toLowerCase() === 'z' && !e.shiftKey) undo()
        if (e.key.toLowerCase() === 'z' && e.shiftKey) redo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [selection.clipIds, setProject, pushCheckpoint, undo, redo, keymap])

  const pxPerSec = useMemo(() => 80 * zoom, [zoom])

  const timelineWidth = TOTAL_DURATION * pxPerSec + 200

  const timeTicks = useMemo(() => {
    const ticks = []
    for (let t = 0; t <= TOTAL_DURATION; t += 0.5) ticks.push(t)
    return ticks
  }, [])

  const clampTime = (value: number) => Math.min(Math.max(value, 0), TOTAL_DURATION)
  const handleScrub = (value: number) => setPlayhead(clampTime(value))

  const stopPlayback = () => {
    sourcesRef.current.forEach(s => {
      try { s.stop() } catch { /* ignore */ }
    })
    sourcesRef.current = []
    if (masterGainRef.current) {
      try { masterGainRef.current.disconnect() } catch { /* ignore */ }
      masterGainRef.current = null
    }
    if (analyserRef.current) {
      try { analyserRef.current.disconnect() } catch { /* ignore */ }
      analyserRef.current = null
    }
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
    const vid = videoRef.current
    if (vid) {
      vid.pause()
      activeVideoIdRef.current = null
    }
  }

  const fetchBuffer = async (ctx: AudioContext, url?: string) => {
    if (!url) return null
    const cache = bufferCacheRef.current
    if (cache.has(url)) return cache.get(url) || null
    const res = await fetch(url)
    const arr = await res.arrayBuffer()
    const buf = await ctx.decodeAudioData(arr)
    cache.set(url, buf)
    bufferDurRef.current.set(url, buf.duration)
    return buf
  }

  const runBeatAnalysis = async (assetId: string) => {
    const asset = assets.find(a => a.id === assetId)
    if (!asset || !(asset.type || '').startsWith('audio') || !asset.url) {
      setBeatStatus({ state: 'error', message: 'Select an audio asset with a valid URL.' })
      return
    }
    try {
      setBeatStatus({ state: 'analyzing', assetId })
      const ctx = new AudioContext()
      const res = await fetch(asset.url)
      const arr = await res.arrayBuffer()
      const audio = await ctx.decodeAudioData(arr)
      const { beats, bpm } = analyzeBeatsFromBuffer(audio)
      setBeatAnalyses(prev => ({ ...prev, [asset.id]: { assetId: asset.id, beats, bpm, createdAt: Date.now() } }))
      setBeatStatus({ state: 'ready', assetId, message: `${beats.length} beats · BPM ~ ${bpm || 'n/a'}` })
    } catch (err) {
      console.error('beat analysis failed', err)
      setBeatStatus({ state: 'error', message: 'Beat analysis failed' })
    }
  }

  const applyBeatsToMarkers = (assetId: string) => {
    const analysis = beatAnalyses[assetId]
    const asset = assets.find(a => a.id === assetId)
    if (!analysis || !asset) return
    setProject(prev => {
      const nextMarkers = [...prev.markers]
      analysis.beats.forEach((t, idx) => {
        nextMarkers.push({ time: clampTime(t), label: `${asset.name} · Beat ${idx + 1}`, color: '#22d3ee' })
      })
      return { ...prev, markers: nextMarkers }
    })
    pushCheckpoint()
  }

  const exportBeatCSV = (assetId: string) => {
    const analysis = beatAnalyses[assetId]
    const asset = assets.find(a => a.id === assetId)
    if (!analysis || !asset) return
    const header = 'Index,Time (s)\n'
    const rows = analysis.beats.map((t, i) => `${i + 1},${t.toFixed(3)}`).join('\n')
    const blob = new Blob([header + rows], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${asset.name.replace(/\s+/g, '_')}_beats.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const startPlayback = async () => {
    const ctx = await ensureAudioContext(audioCtxRef)
    stopPlayback()
    playheadStartRef.current = playhead
    playStartTimeRef.current = ctx.currentTime
    const startAt = ctx.currentTime + 0.05
    const master = ctx.createGain()
    master.gain.setValueAtTime(1, startAt)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 512
    master.connect(analyser)
    analyser.connect(ctx.destination)
    masterGainRef.current = master
    analyserRef.current = analyser

    const audioClips = clips.filter(c => {
      if (!(c.assetType || '').startsWith('audio')) return false
      const tr = trackIndex[c.track]
      if (!tr) return false
      if (tr.locked) return false
      if (anySolo) return tr.solo === true
      return !tr.mute
    })
    const resolved = await Promise.all(audioClips.map(async clip => {
      const buf = await fetchBuffer(ctx, clip.url)
      if (!buf) return null
      const mediaDur = clip.mediaDuration ?? bufferDurRef.current.get(clip.url || '') ?? buf.duration
      const clipOffset = clip.mediaOffset ?? 0
      const offset = Math.max(0, playheadStartRef.current - clip.start + clipOffset)
      const dur = Math.max(0, Math.min(mediaDur - offset, clip.duration))
      if (dur <= 0) return null
      const when = startAt + Math.max(0, clip.start - playheadStartRef.current)
      const src = ctx.createBufferSource()
      src.buffer = buf
      const gainNode = ctx.createGain()
      const targetGain = Math.max(0.0001, clip.gain ?? 1)
      const fadeIn = Math.max(0, clip.fadeIn ?? 0.12)
      const fadeOut = Math.max(0, clip.fadeOut ?? 0.12)
      gainNode.gain.setValueAtTime(targetGain, when)
      if (fadeIn > 0) {
        gainNode.gain.setValueAtTime(0.0001, when)
        gainNode.gain.linearRampToValueAtTime(targetGain, when + Math.min(fadeIn, dur * 0.6))
      }
      if (fadeOut > 0) {
        gainNode.gain.setValueAtTime(targetGain, when + Math.max(0, dur - fadeOut))
        gainNode.gain.linearRampToValueAtTime(0.0001, when + dur)
      }
      src.connect(gainNode)
      gainNode.connect(master)
      src.start(when, offset, dur)
      return src
    }))
    sourcesRef.current = resolved.filter(Boolean) as AudioBufferSourceNode[]

    const stopAt = loopEnabled ? loopRange.end : TOTAL_DURATION

    const tick = () => {
      const ctxNow = ctx.currentTime
      const elapsed = ctxNow - playStartTimeRef.current
      const nextPos = playheadStartRef.current + elapsed
      if (loopEnabled && nextPos >= loopRange.end) {
        setPlayhead(loopRange.start)
        playheadStartRef.current = loopRange.start
        playStartTimeRef.current = ctx.currentTime
        startPlayback()
        return
      }
      if (!loopEnabled && nextPos >= stopAt) {
        setPlayhead(stopAt)
        setPlaying(false)
        return
      }
      setPlayhead(clampTime(nextPos))
      syncVideoToPlayhead(nextPos)
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
  }

  const syncVideoToPlayhead = (pos: number) => {
    const vid = videoRef.current
    if (!vid) return
    const active = clips
      .filter(c => {
        const type = c.assetType || ''
        if (!(type.startsWith('video') || type.startsWith('image'))) return false
        const tr = trackIndex[c.track]
        if (!tr || tr.locked) return false
        if (anySolo && !tr.solo) return false
        if (!anySolo && tr.mute) return false
        return c.url && pos >= c.start && pos <= c.start + c.duration
      })
      .sort((a, b) => {
        const orderA = trackOrder[a.track] ?? 0
        const orderB = trackOrder[b.track] ?? 0
        if (orderA !== orderB) return orderB - orderA
        const endA = a.start + a.duration
        const endB = b.start + b.duration
        return endB - endA
      })[0]
    if (!active) {
      programKindRef.current = 'none'
      vid.pause()
      activeVideoIdRef.current = null
      if (programImageIdRef.current || programImageMode !== 'none') {
        programImageIdRef.current = null
        setProgramImage(null)
        setProgramImageMode('none')
      }
      if (programStatus !== 'idle') setProgramStatus('idle')
      if (programClipIdRef.current) {
        programClipIdRef.current = null
        setProgramClip(null)
      }
      return
    }
    if (programClipIdRef.current !== active.id) {
      const trackName = trackIndex[active.track]?.name || active.track
      const rangeStart = formatTime(active.start)
      const rangeEnd = formatTime(active.start + active.duration)
      programClipIdRef.current = active.id
      setProgramClip({ id: active.id, label: `${trackName} · ${active.title} · ${rangeStart} - ${rangeEnd}` })
    }
    const isImage = (active.assetType || '').startsWith('image')
    if (isImage) {
      programKindRef.current = 'image'
      if (programImageIdRef.current !== active.id) {
        programImageIdRef.current = active.id
        setProgramImage(active.thumb ?? active.url ?? null)
      }
      if (programImageMode !== 'still') setProgramImageMode('still')
      if (programStatus !== 'image') setProgramStatus('image')
      if (activeVideoIdRef.current) {
        vid.pause()
        activeVideoIdRef.current = null
      }
      return
    }
    if (programImageMode === 'still') {
      programImageIdRef.current = null
      setProgramImage(null)
      setProgramImageMode('none')
    }
    programKindRef.current = 'video'
    if (activeVideoIdRef.current !== active.id) {
      vid.src = active.url || ''
      const offset = active.mediaOffset ?? 0
      vid.currentTime = Math.max(0, pos - active.start + offset)
      if (playing) {
        vid.play().catch(() => { /* ignore autoplay blocks */ })
      } else {
        vid.pause()
      }
      activeVideoIdRef.current = active.id
    } else {
      const offset = active.mediaOffset ?? 0
      const target = Math.max(0, pos - active.start + offset)
      if (Math.abs(vid.currentTime - target) > 0.05) vid.currentTime = target
      if (playing) {
        if (vid.paused) vid.play().catch(() => {})
      } else if (!vid.paused) {
        vid.pause()
      }
    }
    const ready = vid.readyState >= 2
    if (ready) {
      if (programStatus !== 'video') setProgramStatus('video')
    } else if (programStatus !== 'loading') {
      setProgramStatus('loading')
    }
    const poster = active.thumb ?? null
    if (poster && !ready) {
      if (programImageIdRef.current !== active.id || programImageMode !== 'poster') {
        programImageIdRef.current = active.id
        setProgramImage(poster)
        setProgramImageMode('poster')
      }
    } else if (programImageMode === 'poster') {
      programImageIdRef.current = null
      setProgramImage(null)
      setProgramImageMode('none')
    }
  }

  const collectSnapPoints = useCallback((trackId: string, excludeId?: string): SnapPoint[] => {
    const pts: SnapPoint[] = []
    markers.forEach(m => pts.push({ time: m.time, label: `Marker · ${m.label}` }))
    const trackClips = clips.filter(c => c.track === trackId && c.id !== excludeId).sort((a, b) => a.start - b.start)
    trackClips.forEach((c, idx) => {
      pts.push({ time: c.start, label: `Edge · ${c.title}` })
      pts.push({ time: c.start + c.duration, label: `Edge · ${c.title}` })
      const prev = trackClips[idx - 1]
      if (prev) {
        const gapStart = prev.start + prev.duration
        const gapEnd = c.start
        if (gapEnd - gapStart >= MIN_CLIP * 0.8) {
          pts.push({ time: gapStart, label: 'Gap start' })
          pts.push({ time: gapEnd, label: 'Gap end' })
        }
      }
    })
    for (let t = 0; t <= TOTAL_DURATION; t += GRID_STEP) pts.push({ time: Number(t.toFixed(3)), label: 'Grid' })
    return pts
  }, [markers, clips])

  const snapTime = useCallback((candidate: number, snaps: SnapPoint[]) => {
    let best = candidate
    let label: string | null = null
    let minDelta = snapStrength
    for (const s of snaps) {
      const d = Math.abs(candidate - s.time)
      if (d < minDelta) {
        minDelta = d
        best = s.time
        label = s.label
      }
    }
    return { time: best, label }
  }, [snapStrength])

  const addMarker = () => {
    const palette = ['#22d3ee', '#f97316', '#e11d48', '#a78bfa', '#22c55e']
    const color = palette[(markers.length) % palette.length]
    const label = `Marker ${markers.length + 1}`
    setProject(prev => ({ ...prev, markers: [...prev.markers, { time: playhead, label, color }] }))
  }

  const handleRulerClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const seconds = x / pxPerSec
    setPlayhead(clampTime(seconds))
  }

  const jumpToMarker = (m: Marker) => {
    setPlayhead(clampTime(m.time))
  }

  const zoomToRange = (start: number, end: number) => {
    if (!timelineRef.current?.parentElement) return
    const span = Math.max(0.6, end - start)
    const viewPx = timelineRef.current.parentElement.clientWidth || 960
    const targetZoom = clamp((viewPx - 120) / (span * 80), ZOOM_MIN, ZOOM_MAX)
    setZoom(targetZoom)
    const px = start * 80 * targetZoom
    timelineRef.current.parentElement.scrollLeft = Math.max(0, px - 60)
  }

  const zoomToSelection = () => {
    if (!selection.clipIds.length) return
    const selClips = clips.filter(c => selection.clipIds.includes(c.id))
    if (!selClips.length) return
    const start = Math.min(...selClips.map(c => c.start))
    const end = Math.max(...selClips.map(c => c.start + c.duration))
    zoomToRange(start, end)
  }

  const zoomToClip = (clip: Clip) => {
    zoomToRange(clip.start, clip.start + clip.duration)
  }

  const handleAssetUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || [])
    if (!files.length) return
    const now = Date.now()
    const stubAssets: Asset[] = files.map((f, idx) => ({
      id: `${now}-${idx}-${f.name}`,
      name: f.name,
      type: f.type || 'unknown',
      duration: Math.max(3, Math.min(20, f.size / 1_000_000)),
      url: URL.createObjectURL(f)
    }))
    setAssets(prev => [...prev, ...stubAssets])
    stubAssets.forEach(stub => {
      const file = files.find(f => stub.id.startsWith(String(now)) && stub.name === f.name)
      if (!file) return
      pendingFilesRef.current.set(stub.id, file)
      setAnalysisStatuses(prev => ({ ...prev, [stub.id]: 'pending' }))
    })
    runNextAnalysis()
    e.target.value = ''
  }

  const handleAssetContextMenu = useCallback((event: React.MouseEvent, asset: Asset) => {
    if (!asset.type.startsWith('image')) {
      setAssetMenu(null)
      return
    }
    event.preventDefault()
    event.stopPropagation()
    const menuWidth = 220
    const menuHeight = 140
    const maxX = Math.max(12, window.innerWidth - menuWidth - 12)
    const maxY = Math.max(12, window.innerHeight - menuHeight - 12)
    const nextX = clamp(event.clientX, 12, maxX)
    const nextY = clamp(event.clientY, 12, maxY)
    setAssetMenu({ assetId: asset.id, x: nextX, y: nextY })
  }, [setAssetMenu])

  const unassignAssetFromTimeline = useCallback((assetId: string) => {
    const asset = assets.find(a => a.id === assetId)
    if (!asset) return
    const removedIds = new Set(clips.filter(c => isClipFromAsset(asset, c)).map(c => c.id))
    if (!removedIds.size) return
    setProject(prev => ({ ...prev, clips: prev.clips.filter(c => !isClipFromAsset(asset, c)) }))
    setSelection(prev => ({ ...prev, clipIds: prev.clipIds.filter(id => !removedIds.has(id)) }))
    pushCheckpoint()
  }, [assets, clips, pushCheckpoint, setProject, setSelection])

  const triggerAssetUnassign = useCallback((assetId: string) => {
    if (assetMenuActionRef.current === 'unassign') return
    assetMenuActionRef.current = 'unassign'
    unassignAssetFromTimeline(assetId)
  }, [unassignAssetFromTimeline])

  const runNextAnalysis = useCallback(async () => {
    if (processingAnalysisRef.current) return
    const pendingEntry = Object.entries(analysisStatusRef.current).find(([, status]) => status === 'pending')
    if (!pendingEntry) return
    const [assetId] = pendingEntry
    const file = pendingFilesRef.current.get(assetId)
    if (!file) {
      setAnalysisStatuses(prev => ({ ...prev, [assetId]: 'error' }))
      return
    }
    processingAnalysisRef.current = true
    setAnalysisStatuses(prev => ({ ...prev, [assetId]: 'processing' }))
    const cacheKey = `asset-meta-${file.name}-${file.size}-${file.lastModified}`
    try {
      const cached = await get(cacheKey) as { duration: number; waveform?: number[] | null; thumb?: string | null } | undefined
      if (cached) {
        setAssets(prev => prev.map(a => a.id === assetId ? { ...a, duration: cached.duration, waveform: cached.waveform ?? a.waveform, thumb: cached.thumb ?? a.thumb } : a))
        setAnalysisStatuses(prev => ({ ...prev, [assetId]: 'cached' }))
      } else {
        const [duration, waveform, thumb] = await Promise.all([
          file.type.startsWith('audio') || file.type.startsWith('video') ? loadMediaDuration(file) : Promise.resolve(Math.max(3, Math.min(20, file.size / 1_000_000))),
          file.type.startsWith('audio') ? decodeWaveform(file) : Promise.resolve(null),
          loadThumb(file)
        ])
        const meta = { duration, waveform, thumb }
        await set(cacheKey, meta)
        setAssets(prev => prev.map(a => a.id === assetId ? { ...a, duration, waveform: waveform ?? a.waveform, thumb: thumb ?? a.thumb } : a))
        setAnalysisStatuses(prev => ({ ...prev, [assetId]: 'done' }))
      }
    } catch (err) {
      console.error('analysis failed', err)
      setAnalysisStatuses(prev => ({ ...prev, [assetId]: 'error' }))
    } finally {
      pendingFilesRef.current.delete(assetId)
      processingAnalysisRef.current = false
      const pendingLeft = Object.values(analysisStatusRef.current).some(v => v === 'pending')
      if (pendingLeft) {
        setTimeout(runNextAnalysis, 20)
      }
    }
  }, [setAnalysisStatuses, setAssets])

  useEffect(() => {
    if (Object.values(analysisStatuses).some(s => s === 'pending') && !processingAnalysisRef.current) {
      runNextAnalysis()
    }
  }, [analysisStatuses, runNextAnalysis])

  const saveDraft = (name?: string) => {
    const trimmed = (name || draftName || 'Draft').trim()
    const id = `${Date.now()}`
    const draft: Draft = {
      id,
      name: trimmed,
      savedAt: Date.now(),
      payload: { project, beatAnalyses }
    }
    setDrafts(prev => {
      const next = [draft, ...prev.filter(d => d.id !== id)].slice(0, 10)
      localStorage.setItem(DRAFTS_KEY, JSON.stringify(next))
      return next
    })
    setActiveDraftId(id)
    setDraftName('')
  }

  const loadDraft = (id: string) => {
    const draft = drafts.find(d => d.id === id)
    if (!draft) return
    setProject(draft.payload.project)
    setBeatAnalyses(draft.payload.beatAnalyses || {})
    setActiveDraftId(id)
  }

  const deleteDraft = (id: string) => {
    setDrafts(prev => {
      const next = prev.filter(d => d.id !== id)
      localStorage.setItem(DRAFTS_KEY, JSON.stringify(next))
      if (activeDraftId === id) setActiveDraftId(next[0]?.id || null)
      return next
    })
  }

  const placeAssetOnTrack = (asset: Asset, trackId: string) => {
    const track = tracks.find(t => t.id === trackId)
    if (track && !isTrackCompatible(asset.type, track.type)) return
    setProject(prev => {
      const trackClips = prev.clips.filter(c => c.track === trackId)
      const end = trackClips.length ? Math.max(...trackClips.map(c => c.start + c.duration)) : 0
      const color = CLIP_COLORS[(prev.clips.length) % CLIP_COLORS.length]
      const newClip: Clip = {
        id: `${asset.id}-clip`,
        assetId: asset.id,
        title: asset.name,
        track: trackId,
        color,
        start: end + 0.1,
        duration: asset.duration,
        url: asset.url,
        assetType: asset.type,
        waveform: asset.waveform ?? null,
        thumb: asset.thumb ?? null,
        mediaDuration: asset.duration,
        mediaOffset: 0,
        gain: 1,
        fadeIn: 0.12,
        fadeOut: 0.12
      }
      // gap-safe place: if overlap disabled, bump start just after previous
      const prevSibling = trackClips.at(-1)
      if (!allowOverlap && prevSibling && newClip.start < prevSibling.start + prevSibling.duration) {
        newClip.start = prevSibling.start + prevSibling.duration + 0.01
      }
      return { ...prev, clips: [...prev.clips, newClip] }
    })
    pushCheckpoint()
  }

  const exportJson = () => {
    const blob = new Blob([JSON.stringify({ tracks, clips, markers }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'timeline_project.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const importJson = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    file.text().then(txt => {
      const data = JSON.parse(txt)
      setProject({
        tracks: data.tracks ?? DEFAULT_TRACKS,
        clips: data.clips ?? DEFAULT_CLIPS,
        markers: data.markers ?? DEFAULT_MARKERS
      })
    }).catch(err => console.error('Import failed', err))
  }

  const renderPreset = async () => {
    if (exportPreset === 'json') {
      exportJson()
      return
    }
    exportAbortRef.current.cancelled = false
    setExportProgress(4)
    setExportStatus('Preparing encoder…')
    let useFfmpeg = false
    try {
      type FfmpegFactory = (opts: { corePath: string; log: boolean }) => { load: () => Promise<void> }
      type FfmpegModule = { createFFmpeg?: FfmpegFactory; default?: { createFFmpeg?: FfmpegFactory } }
      const ffmpegMod = (await import('@ffmpeg/ffmpeg')) as unknown as FfmpegModule
      const createFFmpeg = ffmpegMod.createFFmpeg ?? ffmpegMod.default?.createFFmpeg
      if (!createFFmpeg) throw new Error('ffmpeg factory missing')
      const pickCorePath = async () => {
        const localCore = '/ffmpeg-core/ffmpeg-core.js'
        try {
          const res = await fetch(localCore, { method: 'HEAD' })
          if (res.ok) return localCore
        } catch { /* fallback */ }
        return 'https://unpkg.com/@ffmpeg/core@0.12.10/dist/ffmpeg-core.js'
      }
      const corePath = await pickCorePath()
      const ffmpeg = createFFmpeg({
        corePath,
        log: false
      })
      setExportStatus('Loading ffmpeg.wasm…')
      await ffmpeg.load()
      useFfmpeg = true
      setExportStatus('ffmpeg.wasm ready – staging timeline…')
    } catch (err) {
      console.warn('ffmpeg wasm unavailable, using lightweight mock', err)
      setExportStatus('Falling back to lightweight renderer…')
    }

    const steps = useFfmpeg
      ? ['Mixing audio', 'Muxing container', 'Optimizing chunks']
      : ['Mixing audio', 'Packaging preset', 'Finalizing']

    for (let i = 0; i < steps.length; i++) {
      if (exportAbortRef.current.cancelled) {
        setExportStatus('Cancelled')
        setExportProgress(0)
        return
      }
      setExportStatus(steps[i])
      setExportProgress(20 + (i + 1) * 20)
      // simulate workload; in ffmpeg path we could pipe real commands later
      await new Promise(res => setTimeout(res, 420))
    }

    if (exportAbortRef.current.cancelled) {
      setExportStatus('Cancelled')
      setExportProgress(0)
      return
    }

    const payload = `Render preset ${exportPreset.toUpperCase()} · tracks:${tracks.length} clips:${clips.length} markers:${markers.length} · ${new Date().toISOString()}`
    const blob = new Blob([payload], { type: exportPreset === 'mp4' ? 'video/mp4' : 'video/webm' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = exportPreset === 'mp4' ? 'preview.mp4' : 'preview.webm'
    a.click()
    URL.revokeObjectURL(url)
    setExportProgress(100)
    setExportStatus('Done')
    setTimeout(() => setExportProgress(0), 800)
  }

  // track scroll -> minimap view window
  useEffect(() => {
    const scroller = timelineRef.current?.parentElement
    if (!scroller) return
    const onScroll = () => {
      const start = (scroller.scrollLeft || 0) / pxPerSec
      const duration = (scroller.clientWidth || 1) / pxPerSec
      setViewWindow({ start: clampTime(start), duration: Math.min(TOTAL_DURATION, duration) })
    }
    onScroll()
    scroller.addEventListener('scroll', onScroll)
    return () => scroller.removeEventListener('scroll', onScroll)
  }, [pxPerSec])

  useEffect(() => {
    const scroller = timelineRef.current?.parentElement
    if (!scroller) return
    const startPx = loopRange.start * pxPerSec
    const endPx = loopRange.end * pxPerSec
    const viewStart = scroller.scrollLeft
    const viewEnd = scroller.scrollLeft + scroller.clientWidth
    if (startPx < viewStart + VIEW_PADDING_PX) {
      scroller.scrollLeft = Math.max(0, startPx - VIEW_PADDING_PX)
    } else if (endPx > viewEnd - VIEW_PADDING_PX) {
      scroller.scrollLeft = Math.max(0, endPx - scroller.clientWidth + VIEW_PADDING_PX)
    }
  }, [loopRange, pxPerSec])

  const startClipDrag = (e: React.MouseEvent<HTMLDivElement>, clip: Clip) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const offsetX = e.clientX - rect.left
    const handleZone = 10
    let mode: DragInfo['mode'] = 'move'
    if (offsetX < handleZone) mode = e.altKey ? 'slip' : 'trim-start'
    else if (offsetX > rect.width - handleZone) mode = e.altKey ? 'slide' : 'trim-end'

    dragRef.current = {
      id: clip.id,
      mode,
      startX: e.clientX,
      origStart: clip.start,
      origDuration: clip.duration
    }
    setSnap({ position: clip.start, label: formatTime(clip.start) })
    dragStartedState.current = project
    if (e.metaKey || e.ctrlKey || e.shiftKey) {
      setSelection((prev) => {
        const has = prev.clipIds.includes(clip.id)
        return { clipIds: has ? prev.clipIds.filter(id => id !== clip.id) : [...prev.clipIds, clip.id], marquee: null }
      })
    } else {
      setSelection({ clipIds: [clip.id], marquee: null })
    }
  }

  const startMarquee = (e: React.MouseEvent<HTMLDivElement>) => {
    const container = timelineRef.current
    if (!container) return
    const bounds = container.getBoundingClientRect()
    const x = e.clientX - bounds.left + (container.parentElement?.scrollLeft || 0)
    const y = e.clientY - bounds.top + (container.parentElement?.scrollTop || 0)
    setSelection({ clipIds: [], marquee: { x1: x, x2: x, y1: y, y2: y } })
  }

  const startLoopHandle = (handle: DragHandle) => (e: React.MouseEvent<HTMLDivElement>) => {
    e.stopPropagation()
    loopHandleRef.current = handle
  }

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (loopHandleRef.current && minimapRef.current) {
        const rect = minimapRef.current.getBoundingClientRect()
        const ratio = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width))
        const time = clampTime(ratio * TOTAL_DURATION)
        setLoopRange(range => {
          if (loopHandleRef.current === 'loop-start') {
            const start = Math.min(time, range.end - 0.2)
            return { ...range, start }
          }
          const end = Math.max(time, range.start + 0.2)
          return { ...range, end }
        })
        return
      }

      // marquee drag
      if (selection.marquee) {
        const container = timelineRef.current
        if (!container) return
        const bounds = container.getBoundingClientRect()
        const x = e.clientX - bounds.left + (container.parentElement?.scrollLeft || 0)
        const y = e.clientY - bounds.top + (container.parentElement?.scrollTop || 0)
        const currentBox = { ...selection.marquee, x2: x, y2: y }
        setSelection(prev => ({ ...prev, marquee: currentBox }))
        const [x1, x2] = [Math.min(currentBox.x1, currentBox.x2), Math.max(currentBox.x1, currentBox.x2)]
        const [y1, y2] = [Math.min(currentBox.y1, currentBox.y2), Math.max(currentBox.y1, currentBox.y2)]
        const picked: string[] = []
        clips.forEach(c => {
          const clipX1 = c.start * pxPerSec
          const clipX2 = (c.start + c.duration) * pxPerSec
          const trackIndex = tracks.findIndex(t => t.id === c.track)
          const laneTop = trackIndex * 76 // approximate lane height including gaps
          const laneBottom = laneTop + 70
          if (clipX2 >= x1 && clipX1 <= x2 && laneBottom >= y1 && laneTop <= y2) {
            picked.push(c.id)
          }
        })
        setSelection(prev => ({ ...prev, clipIds: picked }))
        return
      }

      if (!dragRef.current) return
      const { id, mode, startX, origStart, origDuration } = dragRef.current
      const deltaSec = (e.clientX - startX) / pxPerSec
      const current = project.clips.find(c => c.id === id)
      if (!current) return
      const trackId = current.track
      const snaps = collectSnapPoints(trackId, id)
      const siblings = project.clips
        .filter(c => c.track === trackId && c.id !== id)
        .sort((a, b) => a.start - b.start)
      const minDur = MIN_CLIP
      setProject(prev => {
        let updatedClips: Clip[] = prev.clips.map(c => {
          if (c.id !== id) return c
          if (mode === 'move') {
            let candidate = clampTime(origStart + deltaSec)
            const snapRes = snapTime(candidate, snaps)
            candidate = snapRes.time
            const prevSibling = siblings.filter(s => s.start + s.duration <= candidate).at(-1)
            const nextSibling = siblings.find(s => s.start >= candidate)
            if (!allowOverlap) {
              if (prevSibling && candidate < prevSibling.start + prevSibling.duration) {
                candidate = prevSibling.start + prevSibling.duration + 0.01
              }
              if (nextSibling && candidate + c.duration > nextSibling.start) {
                candidate = Math.max(0, nextSibling.start - c.duration - 0.01)
              }
            }
            candidate = clampTime(candidate)
            return { ...c, start: candidate }
          }
          if (mode === 'slip') {
            // Move media in/out without changing clip position/duration
            const mediaDur = c.mediaDuration ?? origDuration
            const maxOffset = Math.max(0, mediaDur - origDuration)
            const nextOffset = clampTime((c.mediaOffset ?? 0) + deltaSec)
            const boundedOffset = Math.min(Math.max(nextOffset, 0), maxOffset)
            return { ...c, mediaOffset: boundedOffset, start: origStart, duration: origDuration }
          }
          if (mode === 'slide') {
            // Slide keeps length, moves clip while pushing/pulling neighbors on same track (no overlaps/gaps)
            const prevSibling = siblings.filter(s => s.start + s.duration <= origStart).at(-1)
            const nextSibling = siblings.find(s => s.start >= origStart)
            if (!prevSibling || !nextSibling) {
              // fallback to move if no both neighbors
              const candidate = clampTime(origStart + deltaSec)
              return { ...c, start: candidate }
            }
            const maxLeft = prevSibling.start + prevSibling.duration + 0.01
            const maxRight = nextSibling.start - c.duration - 0.01
            let candidate = clampTime(origStart + deltaSec)
            candidate = Math.min(Math.max(candidate, maxLeft), maxRight)
            const deltaSlide = candidate - origStart
            updatedClips = updatedClips.map((o: Clip) => {
              if (o.id === c.id) return { ...o, start: candidate }
              if (o.track !== trackId) return o
              if (o.id === prevSibling.id) return { ...o, duration: Math.max(MIN_CLIP, o.duration + deltaSlide) }
              if (o.id === nextSibling.id) return { ...o, start: clampTime(o.start + deltaSlide), duration: Math.max(MIN_CLIP, o.duration - deltaSlide) }
              return o
            })
            return updatedClips.find(o => o.id === id) as Clip
          }
          if (mode === 'trim-start') {
            const prevSibling = siblings.filter(s => s.start + s.duration <= origStart).at(-1)
            const newStart = clampTime(origStart + deltaSec)
            const snappedStart = snapTime(newStart, snaps).time

            // Roll: adjust boundary between previous sibling and this clip
            if (rollEdit && prevSibling) {
              const deltaBoundary = snappedStart - origStart
              const newPrevDur = Math.max(minDur, prevSibling.duration + deltaBoundary)
              const newThisDur = Math.max(minDur, origDuration - deltaBoundary)
              if (newPrevDur >= minDur && newThisDur >= minDur) {
                updatedClips = updatedClips.map((o: Clip) => {
                  if (o.id === prevSibling.id) return { ...o, duration: newPrevDur }
                  if (o.id === id) return { ...o, start: snappedStart, duration: newThisDur }
                  return o
                })
                return updatedClips.find(o => o.id === id) as Clip
              }
            }

            const newDur = Math.max(minDur, origDuration - (snappedStart - origStart))
            const boundedStart = !allowOverlap && prevSibling ? Math.max(snappedStart, prevSibling.start + prevSibling.duration + 0.01) : snappedStart
            return { ...c, start: clampTime(boundedStart), duration: Math.max(minDur, newDur) }
          }
          // trim-end
          let newDur = Math.max(minDur, origDuration + deltaSec)
          const nextSibling = siblings.find(s => s.start >= origStart)

          if (rollEdit && nextSibling) {
            const snappedEnd = snapTime(origStart + newDur, snaps).time
            const deltaBoundary = snappedEnd - (origStart + origDuration)
            const newNextStart = nextSibling.start + deltaBoundary
            const newNextDur = Math.max(minDur, nextSibling.duration - deltaBoundary)
            const newThisDur = Math.max(minDur, origDuration + deltaBoundary)
            if (newNextDur >= minDur) {
              updatedClips = updatedClips.map((o: Clip) => {
                if (o.id === nextSibling.id) return { ...o, start: newNextStart, duration: newNextDur }
                if (o.id === id) return { ...o, duration: newThisDur }
                return o
              })
              return updatedClips.find(o => o.id === id) as Clip
            }
          }

          if (nextSibling && !allowOverlap) {
            newDur = Math.min(newDur, nextSibling.start - origStart - 0.01)
          }
          newDur = Math.min(newDur, TOTAL_DURATION - origStart)
          newDur = Math.max(minDur, newDur)
          const snappedEnd = snapTime(origStart + newDur, snaps).time
          newDur = Math.max(minDur, snappedEnd - origStart)
          return { ...c, duration: newDur }
        })

        // ripple shift downstream clips to maintain gap continuity
        const moved = updatedClips.find(c => c.id === id)
        if (rippleEdit && moved) {
          const delta = moved.start - origStart
          const deltaDur = moved.duration - origDuration
          updatedClips = updatedClips.map((c: Clip) => {
            if (c.id === id || c.track !== trackId) return c
            if (mode === 'move') {
              if (c.start >= origStart) {
                return { ...c, start: clampTime(c.start + delta) }
              }
              return c
            }
            // trims ripple by shifting clips that begin after the trim point
            const pivot = origStart + (mode === 'trim-start' ? 0 : origDuration)
            if (c.start >= pivot) {
              return { ...c, start: clampTime(c.start + deltaDur) }
            }
            return c
          })
        }

        return { ...prev, clips: updatedClips }
      }, { push: false })

      // snap ghost
      if (mode === 'move') {
        const { time, label } = snapTime(clampTime(origStart + deltaSec), snaps)
        setSnap({ position: time, label: label || formatTime(time) })
      } else if (mode === 'trim-start') {
        const { time, label } = snapTime(clampTime(origStart + deltaSec), snaps)
        setSnap({ position: time, label: label || formatTime(time) })
      } else {
        const { time, label } = snapTime(origStart + Math.max(minDur, origDuration + deltaSec), snaps)
        setSnap({ position: time, label: label || formatTime(time) })
      }
    }

    const onUp = () => {
      if (dragRef.current) {
        dragRef.current = null
        pushCheckpoint()
      }
      if (selection.marquee) {
        setSelection(prev => ({ ...prev, marquee: null }))
      }
      setSnap({ position: null, label: null })
      loopHandleRef.current = null
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [pxPerSec, setProject, pushCheckpoint, project.clips, allowOverlap, selection.marquee, clips, tracks, rippleEdit, rollEdit, collectSnapPoints, snapTime])

  return (
    <div className="app">
      <header className="workspace-bar">
        <div className="brand">
          <span className="logo-dot" />
          <div>
            <p className="eyebrow">fastfx · studio</p>
            <h1>Edit workspace</h1>
          </div>
        </div>
        <div className="workspace-modes">
          {['Media', 'Cut', 'Edit', 'Fusion', 'Color', 'Fairlight', 'Deliver'].map(mode => (
            <button key={mode} className={`mode ${mode === 'Edit' ? 'active' : ''}`}>{mode}</button>
          ))}
        </div>
        <div className="deck">
          <label className="pill ghosty">FPS
            <select defaultValue="30">
              <option value="24">24</option>
              <option value="30">30</option>
              <option value="60">60</option>
            </select>
          </label>
          <div className="pill timecode">TC {formatTime(playhead)}</div>
          <div className="transport chunky">
            <button onClick={() => setPlayhead(0)}>⏮</button>
            <button
              onClick={async () => {
                if (!playing) await ensureAudioContext(audioCtxRef)
                setPlaying(p => !p)
              }}
            >
              {playing ? '⏸' : '▶'}
            </button>
            <button onClick={() => setPlayhead(TOTAL_DURATION)}>⏭</button>
          </div>
          <div className="pill slim">
            Track height
            <input
              type="range"
              min={0.6}
              max={1.5}
              step={0.05}
              value={trackHeightScale}
              onChange={(e) => setTrackHeightScale(parseFloat(e.target.value))}
            />
          </div>
          <div className="pill ghosty">
            <button disabled={!canUndo} onClick={undo}>Undo</button>
            <button disabled={!canRedo} onClick={redo}>Redo</button>
          </div>
        </div>
      </header>

      <div className="tabs subtabs">
        {['edit', 'assets', 'export'].map(tab => (
          <button
            key={tab}
            className={activeTab === tab ? 'tab active' : 'tab'}
            onClick={() => setActiveTab(tab as typeof activeTab)}
          >
            {tab === 'edit' && 'Edit'}
            {tab === 'assets' && 'Assets'}
            {tab === 'export' && 'Export'}
          </button>
        ))}
      </div>

      {activeTab === 'edit' && (
        <>
          <div className="workspace-grid">
            <section className="panel media-panel">
              <div className="panel-head">
                <span>Media Pool</span>
                <span className="muted small">Drop or import</span>
              </div>
              <label className="ghost dropzone">
                Import / drag files
                <input type="file" multiple hidden onChange={handleAssetUpload} />
              </label>
              <div className="asset-grid">
                {assets.map(a => (
                  <div
                    key={a.id}
                    className="asset-card"
                    draggable
                    data-testid={`asset-card-${a.id}`}
                    onDragStart={(e) => {
                      e.dataTransfer.setData('text/asset-id', a.id)
                      e.dataTransfer.effectAllowed = 'copy'
                    }}
                    onContextMenu={(e) => handleAssetContextMenu(e, a)}
                  >
                    <div
                      className="asset-thumb-frame"
                      style={{ backgroundImage: a.thumb ? `url(${a.thumb})` : undefined }}
                    >
                      {!a.thumb && a.waveform && (
                        <div className="waveform soft">
                          {a.waveform.map((v, idx) => (
                            <span key={idx} style={{ height: `${Math.max(4, v * 26)}px` }} />
                          ))}
                        </div>
                      )}
                      {!a.thumb && !a.waveform && <div className="muted small">{a.type.split('/')[0]}</div>}
                    </div>
                    <div className="asset-meta">
                      <strong className="asset-title">{a.name}</strong>
                      <div className="muted small">{a.type || 'file'} · ~{a.duration.toFixed(1)}s</div>
                    </div>
                    <div className="asset-actions">
                      {tracks.filter(t => isTrackCompatible(a.type, t.type)).map(t => (
                        <button key={t.id} className="ghost tiny" onClick={() => placeAssetOnTrack(a, t.id)}>
                          Send to {t.name}
                        </button>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </section>

            <section className="monitor-cluster">
                <div className="monitor source">
                  <div className="monitor-head">
                    <span>Source</span>
                    <span className="muted small" data-testid="source-label">{sourcePreview?.label || 'No source selected'}</span>
                  </div>
                  <div
                  className={`monitor-body source-view ${primaryClip ? 'selected' : ''}`}
                  data-testid="source-view"
                  style={{ backgroundImage: sourcePreview?.thumb ? `url(${sourcePreview.thumb})` : undefined }}
                >
                  {!sourcePreview && <div className="muted">Drop an asset to preview</div>}
                  {sourcePreview && sourcePreview.type.startsWith('audio') && sourcePreview.waveform && (
                    <div className="waveform hero">
                      {sourcePreview.waveform.map((v, idx) => (
                        <span key={idx} style={{ height: `${Math.max(6, v * 36)}px` }} />
                      ))}
                    </div>
                  )}
                  {sourcePreview && !sourcePreview.thumb && !sourcePreview.waveform && (
                    <div className="muted small">{sourcePreview.type.split('/')[0] || 'media'}</div>
                  )}
                </div>
              </div>

              <div className="monitor program">
                <div className="monitor-head">
                  <span>Program</span>
                  <span className="muted small">Playhead {formatTime(playhead)}</span>
                </div>
                <div className="monitor-body program-view">
                  <div className={`video-frame ${programImageMode === 'still' ? 'has-image' : ''} ${programImageMode === 'poster' ? 'has-poster' : ''}`}>
                    <video ref={videoRef} muted playsInline data-testid="program-video" />
                    {programImage && (
                      <img
                        src={programImage}
                        alt={programImageMode === 'still' ? 'Program still' : 'Program poster'}
                        className="program-still"
                        data-testid={programImageMode === 'still' ? 'program-image' : 'program-poster'}
                      />
                    )}
                    {programStatus === 'idle' && (
                      <div className="program-placeholder" data-testid="program-status">No clip at playhead</div>
                    )}
                    {programStatus === 'loading' && (
                      <div className="program-placeholder" data-testid="program-status">Loading preview…</div>
                    )}
                    {programStatus === 'error' && (
                      <div className="program-placeholder" data-testid="program-status">Preview unavailable</div>
                    )}
                    <div className="monitor-overlay">
                      {programClip && (
                        <div className="program-clip" data-testid="program-clip">{programClip.label}</div>
                      )}
                      {isScrubbing && (
                        <div className="scrub-overlay" data-testid="scrub-overlay">Scrubbing · {formatTime(playhead)}</div>
                      )}
                      <div className="time-badge">{formatTime(playhead)}</div>
                      <div className="safe-guides" />
                    </div>
                  </div>
                </div>
                <div className="monitor-footer">
                  <label className={`pill ghosty ${loopEnabled ? 'active' : ''}`}>
                    <input type="checkbox" checked={loopEnabled} onChange={(e) => setLoopEnabled(e.target.checked)} />
                    Loop
                    <input
                      type="number"
                      min={0}
                      max={loopRange.end - 0.1}
                      step={0.1}
                      value={loopRange.start}
                      onChange={(e) => setLoopRange({ ...loopRange, start: clampTime(parseFloat(e.target.value) || 0) })}
                      className="pill-input"
                    />
                    <input
                      type="number"
                      min={loopRange.start + 0.1}
                      max={TOTAL_DURATION}
                      step={0.1}
                      value={loopRange.end}
                      onChange={(e) => setLoopRange({ ...loopRange, end: clampTime(parseFloat(e.target.value) || TOTAL_DURATION) })}
                      className="pill-input"
                    />
                  </label>
                  <label className={`pill ghosty ${rippleEdit ? 'active' : ''}`}>
                    <input type="checkbox" checked={rippleEdit} onChange={(e) => setRippleEdit(e.target.checked)} />
                    Ripple edits
                  </label>
                  <label className={`pill ghosty ${rollEdit ? 'active' : ''}`}>
                    <input type="checkbox" checked={rollEdit} onChange={(e) => setRollEdit(e.target.checked)} />
                    Roll trims
                  </label>
                  <label className={`pill ghosty ${allowOverlap ? 'active' : ''}`}>
                    <input type="checkbox" checked={allowOverlap} onChange={(e) => setAllowOverlap(e.target.checked)} />
                    Allow overlaps
                  </label>
                </div>
              </div>
            </section>

            <section className="panel inspector">
              <div className="panel-head">Markers</div>
              <ul className="marker-list">
                {markers.map((m) => (
                  <li key={m.label} style={{ borderLeftColor: m.color }} onClick={() => jumpToMarker(m)}>
                    <span>{formatTime(m.time)}</span>
                    <strong>{m.label}</strong>
                  </li>
                ))}
              </ul>
              <div className="panel-head">Edit status</div>
              <div className="inspector-cards">
                <div className="pill ghosty">Clips {clips.length}</div>
                <div className="pill ghosty">Tracks {tracks.length}</div>
              </div>
              <div className="panel-head">Keyboard map</div>
              <div className="keymap-grid">
                {[
                  { id: 'playPause', label: 'Play / pause' },
                  { id: 'stop', label: 'Stop' },
                  { id: 'nudgeLeft', label: 'Nudge left' },
                  { id: 'nudgeRight', label: 'Nudge right' },
                  { id: 'shuttleBack', label: 'Shuttle back' },
                  { id: 'shuttleForward', label: 'Shuttle forward' }
                ].map(k => (
                  <label key={k.id} className="keymap-row">
                    <span>{k.label}</span>
                    <input
                      className="key-input"
                      value={keymap[k.id as keyof KeyMap] || ''}
                      onChange={(e) => {
                        const next = { ...keymap, [k.id]: e.target.value }
                        setKeymap(next)
                      }}
                      maxLength={8}
                    />
                  </label>
                ))}
              </div>
              <div className="panel-head">Drafts</div>
              <div className="drafts">
                <div className="draft-inputs">
                  <input
                    className="key-input"
                    placeholder="Name draft"
                    value={draftName}
                    onChange={(e) => setDraftName(e.target.value)}
                  />
                  <button className="ghost" onClick={() => saveDraft(draftName || 'Draft')}>Save</button>
                </div>
                <div className="draft-list">
                  {drafts.length === 0 && <div className="muted small">No drafts yet</div>}
                  {drafts.map(d => (
                    <div key={d.id} className={`draft-row ${activeDraftId === d.id ? 'active' : ''}`}>
                      <button className="ghost slim" onClick={() => loadDraft(d.id)}>
                        {d.name} · {new Date(d.savedAt).toLocaleTimeString()}
                      </button>
                      <button className="ghost danger" onClick={() => deleteDraft(d.id)}>×</button>
                    </div>
                  ))}
                </div>
              </div>
              {primaryClip && (
                <>
                  <div className="panel-head">Clip audio · {primaryClip.title}</div>
                  <div className="inspector-cards stacked">
                    <label className="muted small">Gain {((primaryClip.gain ?? 1) * 100).toFixed(0)}%</label>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={primaryClip.gain ?? 1}
                      onChange={(e) => {
                        const gain = parseFloat(e.target.value)
                        setProject(prev => ({ ...prev, clips: prev.clips.map(c => c.id === primaryClip.id ? { ...c, gain } : c) }))
                        pushCheckpoint()
                      }}
                    />
                    <label className="muted small">Fade in {((primaryClip.fadeIn ?? 0.12)).toFixed(2)}s</label>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={primaryClip.fadeIn ?? 0.12}
                      onChange={(e) => {
                        const fadeIn = parseFloat(e.target.value)
                        setProject(prev => ({ ...prev, clips: prev.clips.map(c => c.id === primaryClip.id ? { ...c, fadeIn } : c) }))
                        pushCheckpoint()
                      }}
                    />
                    <label className="muted small">Fade out {((primaryClip.fadeOut ?? 0.12)).toFixed(2)}s</label>
                    <input
                      type="range"
                      min={0}
                      max={2}
                      step={0.05}
                      value={primaryClip.fadeOut ?? 0.12}
                      onChange={(e) => {
                        const fadeOut = parseFloat(e.target.value)
                        setProject(prev => ({ ...prev, clips: prev.clips.map(c => c.id === primaryClip.id ? { ...c, fadeOut } : c) }))
                        pushCheckpoint()
                      }}
                    />
                  </div>
                </>
              )}
            </section>

            <section className="panel beat-panel" data-testid="beat-panel">
              <div className="panel-head">
                <span>Beat detection (AI-lite)</span>
                <span className="muted small">on-device</span>
              </div>
              <div className="beat-form">
                <label className="muted small">Audio asset</label>
                <select
                  value={selectedBeatAsset || assets.find(a => a.type.startsWith('audio'))?.id || ''}
                  onChange={(e) => setSelectedBeatAsset(e.target.value || null)}
                >
                  {assets.filter(a => a.type.startsWith('audio')).map(a => (
                    <option key={a.id} value={a.id}>{a.name}</option>
                  ))}
                </select>
                <button
                  className="ghost"
                  disabled={(!selectedBeatAsset && !assets.find(a => a.type.startsWith('audio'))) || beatStatus.state === 'analyzing'}
                  onClick={() => runBeatAnalysis(selectedBeatAsset || assets.find(a => a.type.startsWith('audio'))?.id || '')}
                >
                  {beatStatus.state === 'analyzing' ? 'Detecting…' : 'Detect beats'}
                </button>
              </div>
              {beatStatus.message && (
                <div className={`pill ghosty ${beatStatus.state === 'error' ? 'error' : ''}`}>
                  {beatStatus.message}
                </div>
              )}
              {selectedBeatAsset && beatAnalyses[selectedBeatAsset] && (
                <div className="beat-results">
                  <div className="pill ghosty">Beats {beatAnalyses[selectedBeatAsset].beats.length}</div>
                  <div className="pill ghosty">BPM ~ {beatAnalyses[selectedBeatAsset].bpm || 'n/a'}</div>
                  <div className="beat-actions">
                    <button className="ghost" onClick={() => applyBeatsToMarkers(selectedBeatAsset)}>Add beats as markers</button>
                    <button className="ghost" onClick={() => exportBeatCSV(selectedBeatAsset)}>Export CSV</button>
                  </div>
                </div>
              )}
              <p className="muted small">Beat peaks are computed from waveform energy. Apply them as timeline markers or export to share.</p>
            </section>
          </div>

          <section className="timeline-shell pro">
            <div className="timeline-toolbar">
              <div className="toolbar-group">
                <div className="pill">Zoom</div>
                <input
                  type="range"
                  min={0.6}
                  max={3}
                  step={0.1}
                  value={zoom}
                  onChange={(e) => setZoom(parseFloat(e.target.value))}
                />
                <div className="pill">Snap</div>
                <input
                  type="range"
                  min={0.04}
                  max={0.22}
                  step={0.01}
                  value={snapStrength}
                  onChange={(e) => setSnapStrength(parseFloat(e.target.value))}
                  title="Snap threshold (seconds)"
                />
                <button className="ghost" onClick={zoomToSelection}>Zoom to selection</button>
                <div className="pill ghosty">Master meter <span className="meter"><span style={{ width: `${Math.min(1, masterLevel * 2) * 100}%` }} /></span></div>
                <div className="pill">Playhead</div>
                <input
                  type="range"
                  min={0}
                  max={TOTAL_DURATION}
                  step={0.1}
                  value={playhead}
                  onChange={(e) => handleScrub(parseFloat(e.target.value))}
                  data-testid="playhead-scrub"
                  onPointerDown={() => setIsScrubbing(true)}
                />
              </div>
            </div>

            <div
              className="timeline-wrapper"
              onWheel={(e) => {
                if (e.ctrlKey || e.metaKey || e.altKey) {
                  e.preventDefault()
                  const delta = e.deltaY > 0 ? -0.12 : 0.12
                  setZoom(z => clamp(z + delta, ZOOM_MIN, ZOOM_MAX))
                } else if (e.shiftKey && timelineRef.current?.parentElement) {
                  e.preventDefault()
                  const scroller = timelineRef.current.parentElement
                  scroller.scrollLeft += e.deltaY
                }
              }}
            >
              <div className="ruler" style={{ width: timelineWidth }} onClick={handleRulerClick}>
                {timeTicks.map((t) => (
                  <div key={t} className="tick" style={{ left: t * pxPerSec }}>
                    <span>{t % 1 === 0 ? t.toFixed(0) : ''}</span>
                  </div>
                ))}
                <div className="playhead" style={{ left: playhead * pxPerSec }} />
              </div>

              <div className="tracks" style={{ width: timelineWidth }} ref={timelineRef} onMouseDown={startMarquee}>
                {selection.marquee && (
                  <div
                    className="marquee"
                    style={{
                      left: Math.min(selection.marquee.x1, selection.marquee.x2),
                      top: Math.min(selection.marquee.y1, selection.marquee.y2),
                      width: Math.abs(selection.marquee.x2 - selection.marquee.x1),
                      height: Math.abs(selection.marquee.y2 - selection.marquee.y1)
                    }}
                  />
                )}
                {loopEnabled && (
                  <div
                    className="loop-range"
                    style={{
                      left: loopRange.start * pxPerSec,
                      width: (loopRange.end - loopRange.start) * pxPerSec
                    }}
                    data-label={`Loop ${formatTime(loopRange.start)} → ${formatTime(loopRange.end)}`}
                  />
                )}
                {snap.position !== null && (
                  <div className="snap-ghost" style={{ left: snap.position * pxPerSec }} data-label={snap.label || ''} />
                )}
                <div className="playhead-line" style={{ left: playhead * pxPerSec }} />
                {tracks.map((track, tIndex) => (
                  <div
                    key={track.id}
                    className="track-row"
                    style={{ ['--track-height' as string]: `${(track.height === 'compact' ? 48 : 70) * trackHeightScale}px` }}
                  >
                    <div className="track-label">
                      <span className="badge">{track.type === 'video' ? 'V' : 'A'}</span>
                      <span className="track-name">{track.name}</span>
                      <div className="track-buttons">
                        <button
                          className={`ghost tiny ${track.mute ? 'active' : ''}`}
                          onClick={() => setProject(prev => ({
                            ...prev,
                            tracks: prev.tracks.map(t => t.id === track.id ? { ...t, mute: !t.mute, solo: t.solo && t.mute ? false : t.solo } : t)
                          }))}
                        >M</button>
                        <button
                          className={`ghost tiny ${track.solo ? 'active' : ''}`}
                          onClick={() => setProject(prev => ({
                            ...prev,
                            tracks: prev.tracks.map(t => t.id === track.id ? { ...t, solo: !t.solo, mute: t.mute && !t.solo ? false : t.mute } : t)
                          }))}
                        >S</button>
                        <button
                          className={`ghost tiny ${track.locked ? 'active' : ''}`}
                          onClick={() => setProject(prev => ({
                            ...prev,
                            tracks: prev.tracks.map(t => t.id === track.id ? { ...t, locked: !t.locked } : t)
                          }))}
                        >🔒</button>
                        <button
                          className="ghost tiny"
                          onClick={() => setProject(prev => ({
                            ...prev,
                            tracks: prev.tracks.map(t => t.id === track.id ? { ...t, height: t.height === 'compact' ? 'normal' : 'compact' } : t)
                          }))}
                        >↕</button>
                      </div>
                    </div>
                    <div
                      className={`track-lane ${track.height === 'compact' ? 'compact' : ''} ${track.locked ? 'locked' : ''}`}
                      data-track-index={tIndex}
                      onDragOver={(e) => {
                        if (track.locked) return
                        if (e.dataTransfer.types.includes('text/asset-id')) {
                          e.preventDefault()
                        }
                      }}
                      onDrop={(e) => {
                        if (track.locked) return
                        e.preventDefault()
                        const assetId = e.dataTransfer.getData('text/asset-id')
                        if (!assetId) return
                        const asset = assets.find(a => a.id === assetId)
                        if (!asset) return
                        if (!isTrackCompatible(asset.type, track.type)) return
                        const rect = e.currentTarget.getBoundingClientRect()
                        const x = e.clientX - rect.left + (timelineRef.current?.parentElement?.scrollLeft || 0)
                        const seconds = clampTime(x / pxPerSec)
                        const color = CLIP_COLORS[(clips.length) % CLIP_COLORS.length]
                        setProject(prev => ({
                          ...prev,
                          clips: [...prev.clips, {
                            id: `${asset.id}-drop-${Date.now()}`,
                            assetId: asset.id,
                            title: asset.name,
                            track: track.id,
                            color,
                            start: seconds,
                            duration: asset.duration,
                            url: asset.url,
                            assetType: asset.type,
                            waveform: asset.waveform ?? null,
                            thumb: asset.thumb ?? null,
                            mediaDuration: asset.duration,
                            mediaOffset: 0
                          }]
                        }))
                        pushCheckpoint()
                      }}
                    >
                      {clips.filter(c => c.track === track.id).map((clip) => {
                        const isAudio = (clip.assetType || '').startsWith('audio') || track.type === 'audio'
                        if (track.locked) return (
                          <div
                            key={clip.id}
                            className={`clip locked ${selection.clipIds.includes(clip.id) ? 'selected' : ''} ${isAudio ? 'audio' : ''}`}
                            style={{
                              left: clip.start * pxPerSec,
                              width: clip.duration * pxPerSec,
                              background: clip.color
                            }}
                            title={`${clip.title} (${formatTime(clip.start)} - ${formatTime(clip.start + clip.duration)})`}
                            data-testid={`clip-${clip.id}`}
                          >
                            {clip.thumb && (
                              <span className="clip-thumb" style={{ backgroundImage: `url(${clip.thumb})` }} />
                            )}
                            <span className="clip-kind">{isAudio ? 'A' : 'V'}</span>
                            <span className="clip-title">{clip.title}</span>
                            <div className="clip-tooltip" data-testid={`clip-tooltip-${clip.id}`}>
                              <div className="clip-tooltip-title">{clip.title}</div>
                              <div>{formatTime(clip.start)} → {formatTime(clip.start + clip.duration)} · Dur {formatTime(clip.duration)}</div>
                              <div className="muted small">Drag edges to trim · Alt+edge slip/slide</div>
                            </div>
                            {isAudio && clip.waveform && (
                              <div className="clip-wave">
                                {clip.waveform.map((v, idx) => (
                                  <span key={idx} style={{ height: `${Math.max(6, v * 28)}px` }} />
                                ))}
                              </div>
                            )}
                          </div>
                        )
                        return (
                        <div
                          key={clip.id}
                          className={`clip ${selection.clipIds.includes(clip.id) ? 'selected' : ''} ${isAudio ? 'audio' : ''}`}
                          style={{
                            left: clip.start * pxPerSec,
                            width: clip.duration * pxPerSec,
                            background: clip.color
                          }}
                          title={`${clip.title} (${formatTime(clip.start)} - ${formatTime(clip.start + clip.duration)})`}
                          data-testid={`clip-${clip.id}`}
                          onMouseDown={(e) => {
                            e.stopPropagation()
                            startClipDrag(e, clip)
                          }}
                          onDoubleClick={(e) => {
                            e.stopPropagation()
                            setSelection({ clipIds: [clip.id], marquee: null })
                            zoomToClip(clip)
                          }}
                          onClick={(e) => {
                            e.stopPropagation()
                            if (!(e.metaKey || e.ctrlKey || e.shiftKey)) {
                              setSelection({ clipIds: [clip.id], marquee: null })
                            }
                          }}
                        >
                          {clip.thumb && (
                            <span className="clip-thumb" style={{ backgroundImage: `url(${clip.thumb})` }} />
                          )}
                          <span className="clip-kind">{isAudio ? 'A' : 'V'}</span>
                          <span className="clip-title">{clip.title}</span>
                          <div className="clip-tooltip" data-testid={`clip-tooltip-${clip.id}`}>
                            <div className="clip-tooltip-title">{clip.title}</div>
                            <div>{formatTime(clip.start)} → {formatTime(clip.start + clip.duration)} · Dur {formatTime(clip.duration)}</div>
                            <div className="muted small">Drag edges to trim · Alt+edge slip/slide</div>
                          </div>
                          {(clip.fadeIn ?? 0) > 0 && (
                            <span
                              className="fade-handle fade-in"
                              style={{ width: Math.min(50, (clip.fadeIn ?? 0) * pxPerSec) }}
                              title={`Fade in ${(clip.fadeIn ?? 0).toFixed(2)}s`}
                            />
                          )}
                          {(clip.fadeOut ?? 0) > 0 && (
                            <span
                              className="fade-handle fade-out"
                              style={{ width: Math.min(50, (clip.fadeOut ?? 0) * pxPerSec) }}
                              title={`Fade out ${(clip.fadeOut ?? 0).toFixed(2)}s`}
                            />
                          )}
                          {(clip.gain ?? 1) !== 1 && (
                            <span className="pill tiny ghosty gain-chip">{Math.round((clip.gain ?? 1) * 100)}%</span>
                          )}
                          {isAudio && clip.waveform && (
                            <div className="clip-wave">
                              {(() => {
                                const zoomBoost = Math.max(1, Math.pow(zoom, 1.6))
                                const bars = Math.max(16, Math.min(clip.waveform.length, Math.floor(clip.waveform.length * zoomBoost / 1.2)))
                                const step = Math.max(1, Math.floor(clip.waveform.length / bars))
                                return clip.waveform
                                  .filter((_, i) => i % step === 0)
                                  .slice(0, bars)
                                  .map((v, idx) => (
                                    <span key={idx} style={{ height: `${Math.max(6, v * 28)}px` }} />
                                  ))
                              })()}
                            </div>
                          )}
                        </div>
                        )
                      })}
                    </div>
                  </div>
                ))}
              </div>
              <div className="minimap" onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect()
                const ratio = (e.clientX - rect.left) / rect.width
                const target = clampTime(ratio * TOTAL_DURATION)
                setPlayhead(target)
                if (timelineRef.current) {
                  const viewport = timelineRef.current.parentElement?.clientWidth || 0
                  const scrollTarget = Math.max(0, target * pxPerSec - viewport / 2)
                  timelineRef.current.parentElement?.scrollTo({ left: scrollTarget, behavior: 'smooth' })
                }
              }} ref={minimapRef}>
                <div className="minimap-track">
                  {clips.map(c => (
                    <div
                      key={c.id}
                      className="mini-clip"
                      style={{ left: `${(c.start / TOTAL_DURATION) * 100}%`, width: `${(c.duration / TOTAL_DURATION) * 100}%`, background: c.color }}
                    />
                  ))}
                  <div
                    className="loop-handle start"
                    style={{ left: `${(loopRange.start / TOTAL_DURATION) * 100}%` }}
                    onMouseDown={startLoopHandle('loop-start')}
                  />
                  <div
                    className="loop-handle end"
                    style={{ left: `${(loopRange.end / TOTAL_DURATION) * 100}%` }}
                    onMouseDown={startLoopHandle('loop-end')}
                  />
                  <div
                    className="mini-view"
                    style={{
                      left: `${(viewWindow.start / TOTAL_DURATION) * 100}%`,
                      width: `${(viewWindow.duration / TOTAL_DURATION) * 100}%`
                    }}
                  />
                  <div className="mini-playhead" style={{ left: `${(playhead / TOTAL_DURATION) * 100}%` }} />
                </div>
              </div>
            </div>
          </section>
        </>
      )}

      {activeTab === 'assets' && (
        <div className="placeholder">
          <p><strong>Asset bin</strong></p>
          <label className="ghost">
            Import files
            <input type="file" multiple hidden onChange={handleAssetUpload} />
          </label>
          <ul className="asset-list">
            {assets.length === 0 && <li>No assets yet</li>}
            {assets.map(a => (
              <li
                key={a.id}
                className="asset-row"
                draggable
                data-testid={`asset-row-${a.id}`}
                onDragStart={(e) => {
                  e.dataTransfer.setData('text/asset-id', a.id)
                  e.dataTransfer.effectAllowed = 'copy'
                }}
                onContextMenu={(e) => handleAssetContextMenu(e, a)}
              >
                <div className="asset-meta">
                  <strong className="asset-title">{a.name}</strong>
                  <div className="muted small">{a.type || 'file'} · ~{a.duration.toFixed(1)}s</div>
                  {analysisStatuses[a.id] && (
                    <div className={`pill ghosty ${analysisStatuses[a.id] === 'error' ? 'error' : ''}`}>
                      {analysisStatuses[a.id] === 'pending' && 'Queued for analysis'}
                      {analysisStatuses[a.id] === 'processing' && 'Analyzing…'}
                      {analysisStatuses[a.id] === 'cached' && 'Loaded from cache'}
                      {analysisStatuses[a.id] === 'done' && 'Analyzed'}
                      {analysisStatuses[a.id] === 'error' && 'Analysis failed'}
                    </div>
                  )}
                  {a.thumb && <img src={a.thumb} alt="thumb" className="asset-thumb" />}
                  {a.waveform && (
                    <div className="waveform">
                      {a.waveform.map((v, idx) => (
                        <span key={idx} style={{ height: `${Math.max(4, v * 26)}px` }} />
                      ))}
                    </div>
                  )}
                </div>
                <div className="asset-actions">
                  {tracks.filter(t => isTrackCompatible(a.type, t.type)).map(t => (
                    <button key={t.id} className="ghost" onClick={() => placeAssetOnTrack(a, t.id)}>Send to {t.name}</button>
                  ))}
                </div>
              </li>
            ))}
          </ul>
          <p className="muted">Waveform/thumb extraction ready; drag assets straight onto a track or use the send buttons.</p>
        </div>
      )}
      {activeTab === 'export' && (
        <div className="placeholder export-box">
          <div className="export-actions">
            <select className="ghost" value={exportPreset} onChange={(e) => setExportPreset(e.target.value as typeof exportPreset)}>
              <option value="json">JSON bundle</option>
              <option value="mp4">MP4 · 1080p</option>
              <option value="webm">WebM · 720p</option>
            </select>
            <button className="ghost" onClick={renderPreset}>Render preset</button>
            <button className="ghost danger" onClick={() => { exportAbortRef.current.cancelled = true; }}>Cancel</button>
            <label className="ghost">
              Import JSON
              <input type="file" accept="application/json" hidden onChange={importJson} />
            </label>
            <button className="ghost" onClick={addMarker}>Add marker @ playhead</button>
          </div>
          <div className="pill ghosty">Status: {exportStatus || 'Idle'}</div>
          {exportProgress > 0 && (
            <div className="progress">
              <div className="progress-bar" style={{ width: `${Math.min(100, exportProgress)}%` }} />
            </div>
          )}
          <p>Preset renderer uses a lightweight ffmpeg.wasm path when available; cancel anytime.</p>
        </div>
      )}
      {assetMenu && assetMenuAsset && (
        <div
          className="asset-context-menu"
          role="menu"
          ref={assetMenuRef}
          style={{ left: assetMenu.x, top: assetMenu.y }}
          data-testid="asset-context-menu"
        >
          <div className="asset-context-label">{assetMenuAsset.name}</div>
          <button
            type="button"
            className="asset-context-action"
            role="menuitem"
            disabled={assetMenuClipCount === 0}
            onPointerDown={(event) => {
              if (event.button !== 0) return
              triggerAssetUnassign(assetMenu.assetId)
            }}
            onClick={() => {
              triggerAssetUnassign(assetMenu.assetId)
            }}
            data-testid="asset-unassign"
          >
            Unassign from timeline{assetMenuClipCount ? ` (${assetMenuClipCount})` : ''}
          </button>
          {assetMenuClipCount === 0 && (
            <div className="asset-context-hint muted small">No timeline clips to remove</div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
