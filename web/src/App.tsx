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

const TOTAL_DURATION = 18
const GRID_STEP = 0.25
const SNAP_THRESHOLD_SEC = 0.12
const MIN_CLIP = 0.2

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
  const [loopEnabled, setLoopEnabled] = useState(false)
  const [loopRange, setLoopRange] = useState<{ start: number; end: number }>({ start: 0, end: 8 })
  const { state: project, set: setProject, undo, redo, canUndo, canRedo, pushCheckpoint } = useHistoryState<ProjectState>(
    { tracks: DEFAULT_TRACKS, clips: DEFAULT_CLIPS, markers: DEFAULT_MARKERS }
  )
  const { tracks, clips, markers } = project
  const [selection, setSelection] = useState<SelectionState>({ clipIds: [] })

  const timelineRef = useRef<HTMLDivElement | null>(null)
  const [viewWindow, setViewWindow] = useState({ start: 0, duration: 8 })
  const rafRef = useRef<number | null>(null)

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

  // playback loop
  useEffect(() => {
    if (!playing) {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      return
    }
    let lastTs: number | null = null
    const tick = (ts: number) => {
      if (lastTs === null) lastTs = ts
      const delta = (ts - lastTs) / 1000
      lastTs = ts
      setPlayhead(prev => {
        const next = clampTime(prev + delta)
        if (next >= TOTAL_DURATION) {
          setPlaying(false)
          return TOTAL_DURATION
        }
        return next
      })
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [playing])

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
          setSelection({ clipIds: [] })
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

  const collectSnapPoints = (trackId: string, excludeId?: string) => {
    const pts = new Set<number>()
    markers.forEach(m => pts.add(m.time))
    clips.filter(c => c.track === trackId && c.id !== excludeId).forEach(c => {
      pts.add(c.start)
      pts.add(c.start + c.duration)
    })
    for (let t = 0; t <= TOTAL_DURATION; t += GRID_STEP) pts.add(Number(t.toFixed(3)))
    return Array.from(pts.values())
  }

  const snapTime = (candidate: number, snaps: number[]) => {
    let best = candidate
    let minDelta = SNAP_THRESHOLD_SEC
    for (const s of snaps) {
      const d = Math.abs(candidate - s)
      if (d < minDelta) {
        minDelta = d
        best = s
      }
    }
    return best
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

  const startDrag = (e: React.MouseEvent<HTMLDivElement>, clip: Clip) => {
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
    dragStartedState.current = project
    if (e.metaKey || e.ctrlKey || e.shiftKey) {
      setSelection((prev) => {
        const has = prev.clipIds.includes(clip.id)
        return { clipIds: has ? prev.clipIds.filter(id => id !== clip.id) : [...prev.clipIds, clip.id] }
      })
    } else {
      setSelection({ clipIds: [clip.id] })
    }
  }

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
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
      setProject(prev => ({
        ...prev,
        clips: prev.clips.map(c => {
          if (c.id !== id) return c
          if (mode === 'move') {
            let candidate = clampTime(origStart + deltaSec)
            candidate = snapTime(candidate, snaps)
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
            const snappedStart = snapTime(newStart, snaps)
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
          const snappedEnd = snapTime(origStart + newDur, snaps)
          newDur = Math.max(minDur, snappedEnd - origStart)
          return { ...c, duration: newDur }
        })
      }), { push: false })
    }

    const onUp = () => {
      if (dragRef.current) {
        dragRef.current = null
        pushCheckpoint()
      }
    }

    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
  }, [pxPerSec, setProject, pushCheckpoint, project.clips])

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
            <button onClick={() => setPlaying(p => !p)}>{playing ? '⏸' : '▶'}</button>
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
                <li key={m.label} style={{ borderLeftColor: m.color }}>
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

            <div className="timeline-wrapper">
              <div className="ruler" style={{ width: timelineWidth }} onClick={handleRulerClick}>
                {timeTicks.map((t) => (
                  <div key={t} className="tick" style={{ left: t * pxPerSec }}>
                    <span>{t % 1 === 0 ? t.toFixed(0) : ''}</span>
                  </div>
                ))}
                <div className="playhead" style={{ left: playhead * pxPerSec }} />
              </div>

              <div className="tracks" style={{ width: timelineWidth }} ref={timelineRef}>
                {tracks.map((track) => (
                  <div key={track.id} className="track-row">
                    <div className="track-label">
                      <span className="badge">{track.type === 'video' ? 'V' : 'A'}</span>
                      {track.name}
                    </div>
                    <div className="track-lane">
                      {clips.filter(c => c.track === track.id).map((clip) => (
                        <div
                          key={clip.id}
                          className={`clip ${selection.clipIds.includes(clip.id) ? 'selected' : ''}`}
                          style={{
                            left: clip.start * pxPerSec,
                            width: clip.duration * pxPerSec,
                            background: clip.color
                          }}
                          title={`${clip.title} (${formatTime(clip.start)} - ${formatTime(clip.start + clip.duration)})`}
                          onMouseDown={(e) => startDrag(e, clip)}
                          onClick={(e) => {
                            if (!(e.metaKey || e.ctrlKey || e.shiftKey)) {
                              setSelection({ clipIds: [clip.id] })
                            }
                          }}
                        >
                          <span>{clip.title}</span>
                        </div>
                      ))}
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
              }}>
                <div className="minimap-track">
                  {clips.map(c => (
                    <div
                      key={c.id}
                      className="mini-clip"
                      style={{ left: `${(c.start / TOTAL_DURATION) * 100}%`, width: `${(c.duration / TOTAL_DURATION) * 100}%`, background: c.color }}
                    />
                  ))}
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
          <p>Asset bin will live here (imports, waveform/thumb extraction).</p>
          <p>Planned: drag to timeline, search, filter by type.</p>
        </div>
      )}
      {activeTab === 'export' && (
        <div className="placeholder export-box">
          <div className="export-actions">
            <button className="ghost" onClick={exportJson}>Export JSON</button>
            <label className="ghost">
              Import JSON
              <input type="file" accept="application/json" hidden onChange={importJson} />
            </label>
            <button className="ghost" onClick={addMarker}>Add marker @ playhead</button>
          </div>
          <p>Later: choose render target (mp4/webm), codec, range, burn-in markers.</p>
        </div>
      )}
    </div>
  )
}

export default App
