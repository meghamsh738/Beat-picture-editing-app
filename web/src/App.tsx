import { useEffect, useMemo, useRef, useState } from 'react'
import type React from 'react'
import './App.css'

type Clip = {
  id: string
  title: string
  track: string
  color: string
  start: number // seconds
  duration: number // seconds
  url?: string
  assetType?: string
  waveform?: number[] | null
}

type Marker = { time: number; label: string; color: string }

type Track = { id: string; name: string; type: 'video' | 'audio' }

type ProjectState = {
  tracks: Track[]
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

type Asset = {
  id: string
  name: string
  type: string
  duration: number
  url?: string
  waveform?: number[]
  thumb?: string
}

const DEFAULT_TRACKS: Track[] = [
  { id: 'v1', name: 'V1 · Motion', type: 'video' },
  { id: 'v2', name: 'V2 · Titles', type: 'video' },
  { id: 'a1', name: 'A1 · Music', type: 'audio' },
  { id: 'a2', name: 'A2 · Foley', type: 'audio' }
]

const DEFAULT_CLIPS: Clip[] = [
  { id: 'c1', title: 'Intro', track: 'v1', color: '#4ade80', start: 0, duration: 6 },
  { id: 'c2', title: 'Scene A', track: 'v1', color: '#60a5fa', start: 6.5, duration: 8 },
  { id: 'c3', title: 'Lower Third', track: 'v2', color: '#f472b6', start: 6.7, duration: 4 },
  { id: 'c4', title: 'Chorus', track: 'a1', color: '#fbbf24', start: 2, duration: 12 },
  { id: 'c5', title: 'Steps', track: 'a2', color: '#a78bfa', start: 4, duration: 5.5 }
]

const DEFAULT_MARKERS: Marker[] = [
  { time: 2, label: 'Beat drop', color: '#22d3ee' },
  { time: 8.5, label: 'Cut to B-roll', color: '#f97316' },
  { time: 12, label: 'Logo hit', color: '#e11d48' }
]

const CLIP_COLORS = ['#4ade80', '#60a5fa', '#f472b6', '#fbbf24', '#a78bfa', '#22d3ee']

const TOTAL_DURATION = 18
const GRID_STEP = 0.25
const SNAP_THRESHOLD_SEC = 0.12
const MIN_CLIP = 0.2
const ZOOM_MIN = 0.6
const ZOOM_MAX = 3
const VIEW_PADDING_PX = 60

const clamp = (v: number, min: number, max: number) => Math.min(max, Math.max(min, v))

const ensureAudioContext = async (ref: React.MutableRefObject<AudioContext | null>) => {
  if (!ref.current) ref.current = new AudioContext()
  if (ref.current.state === 'suspended') await ref.current.resume()
  return ref.current
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
  } catch (err) {
    resolve(5)
  }
})

const decodeWaveform = async (file: File): Promise<number[] | null> => {
  if (typeof window === 'undefined' || typeof AudioContext === 'undefined') return null
  try {
    const buffer = await file.arrayBuffer()
    const ctx = new AudioContext({ sampleRate: 48000 })
    const audio = await ctx.decodeAudioData(buffer)
    const channel = audio.getChannelData(0)
    const buckets = 64
    const step = Math.floor(channel.length / buckets)
    const samples: number[] = []
    for (let i = 0; i < buckets; i++) {
      let sum = 0
      for (let j = 0; j < step; j++) {
        const v = channel[i * step + j]
        sum += Math.abs(v)
      }
      samples.push(sum / step)
    }
    const max = Math.max(...samples, 0.001)
    return samples.map(s => s / max)
  } catch (err) {
    console.warn('waveform decode failed', err)
    return null
  }
}

const loadThumb = (file: File): Promise<string | null> => new Promise(resolve => {
  if (!file.type.startsWith('image')) return resolve(null)
  const reader = new FileReader()
  reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : null)
  reader.onerror = () => resolve(null)
  reader.readAsDataURL(file)
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
  const [loopEnabled, setLoopEnabled] = useState(false)
  const [loopRange, setLoopRange] = useState<{ start: number; end: number }>({ start: 0, end: 8 })
  const [assets, setAssets] = useState<Asset[]>([])
  const [exportPreset, setExportPreset] = useState<'json' | 'mp4' | 'webm'>('json')
  const [isRendering, setIsRendering] = useState(false)
  const { state: project, set: setProject, undo, redo, canUndo, canRedo, pushCheckpoint } = useHistoryState<ProjectState>(
    { tracks: DEFAULT_TRACKS, clips: DEFAULT_CLIPS, markers: DEFAULT_MARKERS }
  )
  const { tracks, clips, markers } = project
  const [selection, setSelection] = useState<SelectionState>({ clipIds: [], marquee: null })
  const [snap, setSnap] = useState<SnapState>({ position: null, label: null })
  const loopHandleRef = useRef<DragHandle>(null)

  const audioCtxRef = useRef<AudioContext | null>(null)
  const bufferCacheRef = useRef<Map<string, AudioBuffer>>(new Map())
  const sourcesRef = useRef<AudioBufferSourceNode[]>([])
  const playheadStartRef = useRef<number>(0)
  const playStartTimeRef = useRef<number>(0)

  const timelineRef = useRef<HTMLDivElement | null>(null)
  const [viewWindow, setViewWindow] = useState({ start: 0, duration: 8 })
  const rafRef = useRef<number | null>(null)
  const minimapRef = useRef<HTMLDivElement | null>(null)

  type DragInfo = {
    id: string
    mode: 'move' | 'trim-start' | 'trim-end'
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
      } catch (err) {
        console.warn('Failed to parse saved project', err)
      }
    }
  }, [])

  // Persist project
  useEffect(() => {
    const payload = JSON.stringify(project)
    localStorage.setItem(STORAGE_KEY, payload)
  }, [project])

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

  // Keyboard: nudge, playback toggles, undo/redo, delete/duplicate
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const step = e.shiftKey ? 0.5 : 0.1
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        setProject(prev => ({
          ...prev,
          clips: prev.clips.map(c => selection.clipIds.includes(c.id)
            ? { ...c, start: Math.max(0, c.start + (e.key === 'ArrowLeft' ? -step : step)) }
            : c)
        }), { push: false })
        pushCheckpoint()
      }
      if (e.code === 'Space') {
        e.preventDefault()
        setPlaying(p => !p)
      }
      if (e.key.toLowerCase() === 'l' && !e.metaKey && !e.ctrlKey) setPlayhead(p => clampTime(p + 0.5))
      if (e.key.toLowerCase() === 'j' && !e.metaKey && !e.ctrlKey) setPlayhead(p => clampTime(p - 0.5))
      if (e.key.toLowerCase() === 'k' && !e.metaKey && !e.ctrlKey) setPlaying(false)
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
  }, [selection.clipIds, setProject, pushCheckpoint, undo, redo])

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
      try { s.stop() } catch (_) { /* ignore */ }
    })
    sourcesRef.current = []
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
    rafRef.current = null
  }

  const fetchBuffer = async (ctx: AudioContext, url?: string) => {
    if (!url) return null
    const cache = bufferCacheRef.current
    if (cache.has(url)) return cache.get(url) || null
    const res = await fetch(url)
    const arr = await res.arrayBuffer()
    const buf = await ctx.decodeAudioData(arr)
    cache.set(url, buf)
    return buf
  }

  const startPlayback = async () => {
    const ctx = await ensureAudioContext(audioCtxRef)
    stopPlayback()
    playheadStartRef.current = playhead
    playStartTimeRef.current = ctx.currentTime
    const startAt = ctx.currentTime + 0.05
    const audioClips = clips.filter(c => (c.assetType || '').startsWith('audio'))
    const resolved = await Promise.all(audioClips.map(async clip => {
      const buf = await fetchBuffer(ctx, clip.url)
      if (!buf) return null
      const offset = Math.max(0, playheadStartRef.current - clip.start)
      const dur = Math.max(0, clip.duration - offset)
      if (dur <= 0) return null
      const when = startAt + Math.max(0, clip.start - playheadStartRef.current)
      const src = ctx.createBufferSource()
      src.buffer = buf
      src.connect(ctx.destination)
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
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
  }

  const collectSnapPoints = (trackId: string, excludeId?: string): SnapPoint[] => {
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
  }

  const snapTime = (candidate: number, snaps: SnapPoint[]) => {
    let best = candidate
    let label: string | null = null
    let minDelta = SNAP_THRESHOLD_SEC
    for (const s of snaps) {
      const d = Math.abs(candidate - s.time)
      if (d < minDelta) {
        minDelta = d
        best = s.time
        label = s.label
      }
    }
    return { time: best, label }
  }

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

    // Enrich metadata async (duration, thumb, waveform)
    for (const stub of stubAssets) {
      const file = files.find(f => stub.id.startsWith(String(now)) && stub.name === f.name)
      if (!file) continue
      const [duration, waveform, thumb] = await Promise.all([
        file.type.startsWith('audio') || file.type.startsWith('video') ? loadMediaDuration(file) : Promise.resolve(stub.duration),
        file.type.startsWith('audio') ? decodeWaveform(file) : Promise.resolve(null),
        loadThumb(file)
      ])
      setAssets(prev => prev.map(a => a.id === stub.id ? { ...a, duration, waveform: waveform ?? a.waveform, thumb: thumb ?? a.thumb } : a))
    }

    e.target.value = ''
  }

  const placeAssetOnTrack = (asset: Asset, trackId: string) => {
    setProject(prev => {
      const trackClips = prev.clips.filter(c => c.track === trackId)
      const end = trackClips.length ? Math.max(...trackClips.map(c => c.start + c.duration)) : 0
      const color = CLIP_COLORS[(prev.clips.length) % CLIP_COLORS.length]
      const newClip: Clip = {
        id: `${asset.id}-clip`,
        title: asset.name,
        track: trackId,
        color,
        start: end + 0.1,
        duration: asset.duration,
        url: asset.url,
        assetType: asset.type,
        waveform: asset.waveform ?? null
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
    if (offsetX < handleZone) mode = 'trim-start'
    else if (offsetX > rect.width - handleZone) mode = 'trim-end'

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
        let updatedClips = prev.clips.map(c => {
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
          if (mode === 'trim-start') {
            const newStart = clampTime(origStart + deltaSec)
            const snappedStart = snapTime(newStart, snaps).time
            const newDur = Math.max(minDur, origDuration - (snappedStart - origStart))
            const prevSibling = siblings.filter(s => s.start + s.duration <= origStart).at(-1)
            const boundedStart = !allowOverlap && prevSibling ? Math.max(snappedStart, prevSibling.start + prevSibling.duration + 0.01) : snappedStart
            return { ...c, start: clampTime(boundedStart), duration: Math.max(minDur, newDur) }
          }
          let newDur = Math.max(minDur, origDuration + deltaSec)
          const nextSibling = siblings.find(s => s.start >= origStart)
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
          updatedClips = updatedClips.map(c => {
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
  }, [pxPerSec, setProject, pushCheckpoint, project.clips, allowOverlap, selection.marquee, clips, tracks, rippleEdit])

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <p className="eyebrow">Timeline Builder · fastfx</p>
          <h1>Premiere-style timeline prototype</h1>
        </div>
          <div className="controls-row">
            <label className="pill">
              FPS
            <select defaultValue="30">
              <option value="24">24</option>
              <option value="30">30</option>
              <option value="60">60</option>
            </select>
          </label>
          <div className="pill">Timecode {formatTime(playhead)}</div>
          <div className="transport">
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
          <div className="pill">
            <button disabled={!canUndo} onClick={undo}>⌘Z Undo</button>
            <button disabled={!canRedo} onClick={redo}>⇧⌘Z Redo</button>
          </div>
          <label className="pill">
            <input type="checkbox" checked={allowOverlap} onChange={(e) => setAllowOverlap(e.target.checked)} />
            Allow overlap
          </label>
          <label className="pill">
            <input type="checkbox" checked={rippleEdit} onChange={(e) => setRippleEdit(e.target.checked)} />
            Ripple move/trim
          </label>
          <label className="pill">
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
        </div>
      </header>

      <div className="tabs">
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
        <div className="layout">
          <aside className="sidepanel">
            <div className="panel-head">Assets</div>
            <ul className="asset-list">
              {['intro.mp4', 'broll.mov', 'main.wav', 'foley.wav'].map((a) => (
                <li key={a}>{a}</li>
              ))}
            </ul>
            <div className="panel-head">Markers</div>
            <ul className="marker-list">
              {markers.map((m) => (
                <li key={m.label} style={{ borderLeftColor: m.color }} onClick={() => jumpToMarker(m)}>
                  <span>{formatTime(m.time)}</span>
                  <strong>{m.label}</strong>
                </li>
              ))}
            </ul>
          </aside>

          <section className="timeline-shell">
            <div className="timeline-toolbar">
              <div className="pill">Zoom</div>
            <input
              type="range"
              min={0.6}
              max={3}
              step={0.1}
              value={zoom}
              onChange={(e) => setZoom(parseFloat(e.target.value))}
            />
            <button className="ghost" onClick={() => {
              if (!selection.clipIds.length) return
              const picked = clips.filter(c => selection.clipIds.includes(c.id))
              if (!picked.length || !timelineRef.current) return
              const start = Math.min(...picked.map(c => c.start))
              const end = Math.max(...picked.map(c => c.start + c.duration))
              const padding = 0.5
              const span = Math.max(MIN_CLIP, end - start + padding)
              const viewPx = timelineRef.current.parentElement?.clientWidth || 800
              const targetZoom = clamp(viewPx / (span * 80), ZOOM_MIN, ZOOM_MAX)
              setZoom(targetZoom)
              const scrollTarget = Math.max(0, (start - padding) * (80 * targetZoom))
              timelineRef.current.parentElement?.scrollTo({ left: scrollTarget, behavior: 'smooth' })
            }}>Zoom to selection</button>
              <div className="pill">Playhead</div>
              <input
                type="range"
                min={0}
                max={TOTAL_DURATION}
                step={0.1}
                value={playhead}
                onChange={(e) => handleScrub(parseFloat(e.target.value))}
              />
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
                {tracks.map((track, tIndex) => (
                  <div key={track.id} className="track-row">
                    <div className="track-label">
                      <span className="badge">{track.type === 'video' ? 'V' : 'A'}</span>
                      {track.name}
                    </div>
                    <div
                      className="track-lane"
                      data-track-index={tIndex}
                      onDragOver={(e) => {
                        if (e.dataTransfer.types.includes('text/asset-id')) {
                          e.preventDefault()
                        }
                      }}
                      onDrop={(e) => {
                        e.preventDefault()
                        const assetId = e.dataTransfer.getData('text/asset-id')
                        if (!assetId) return
                        const asset = assets.find(a => a.id === assetId)
                        if (!asset) return
                        const rect = e.currentTarget.getBoundingClientRect()
                        const x = e.clientX - rect.left + (timelineRef.current?.parentElement?.scrollLeft || 0)
                        const seconds = clampTime(x / pxPerSec)
                        const color = CLIP_COLORS[(clips.length) % CLIP_COLORS.length]
                        setProject(prev => ({
                          ...prev,
                          clips: [...prev.clips, {
                            id: `${asset.id}-drop-${Date.now()}`,
                            title: asset.name,
                            track: track.id,
                            color,
                            start: seconds,
                            duration: asset.duration,
                            url: asset.url,
                            assetType: asset.type,
                            waveform: asset.waveform ?? null
                          }]
                        }))
                        pushCheckpoint()
                      }}
                    >
                      {clips.filter(c => c.track === track.id).map((clip) => {
                        const isAudio = (clip.assetType || '').startsWith('audio') || track.type === 'audio'
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
                          onMouseDown={(e) => {
                            e.stopPropagation()
                            startClipDrag(e, clip)
                          }}
                          onClick={(e) => {
                            e.stopPropagation()
                            if (!(e.metaKey || e.ctrlKey || e.shiftKey)) {
                              setSelection({ clipIds: [clip.id], marquee: null })
                            }
                          }}
                        >
                          <span>{clip.title}</span>
                          {isAudio && clip.waveform && (
                            <div className="clip-wave">
                              {clip.waveform.map((v, idx) => (
                                <span key={idx} style={{ height: `${Math.max(6, v * 28)}px` }} />
                              ))}
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
        </div>
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
                onDragStart={(e) => {
                  e.dataTransfer.setData('text/asset-id', a.id)
                  e.dataTransfer.effectAllowed = 'copy'
                }}
              >
                <div className="asset-meta">
                  <strong>{a.name}</strong>
                  <div className="muted small">{a.type || 'file'} · ~{a.duration.toFixed(1)}s</div>
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
                  {tracks.map(t => (
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
            <button className="ghost" disabled={isRendering} onClick={async () => {
              if (exportPreset === 'json') {
                exportJson()
                return
              }
              setIsRendering(true)
              await new Promise(res => setTimeout(res, 450))
              const payload = `Rendered ${exportPreset.toUpperCase()} preview with ${tracks.length} tracks, ${clips.length} clips, ${markers.length} markers.`
              const blob = new Blob([payload], { type: exportPreset === 'mp4' ? 'video/mp4' : 'video/webm' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = exportPreset === 'mp4' ? 'preview.mp4' : 'preview.webm'
              a.click()
              URL.revokeObjectURL(url)
              setIsRendering(false)
            }}>{isRendering ? 'Rendering…' : 'Render preset'}</button>
            <label className="ghost">
              Import JSON
              <input type="file" accept="application/json" hidden onChange={importJson} />
            </label>
            <button className="ghost" onClick={addMarker}>Add marker @ playhead</button>
          </div>
          <p>Preset renderer mocks final output; swap in ffmpeg/wasm backend later for real encodes.</p>
        </div>
      )}
    </div>
  )
}

export default App
