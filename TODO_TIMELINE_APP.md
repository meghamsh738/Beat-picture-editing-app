# Timeline Editing App TODO (persistent checklist)

## Legend
- [x] done   - [ ] pending   - [>] in progress / partial

## Recently completed
- [x] Audio playback via Web Audio, playhead sync, loop awareness.
- [x] Audio clip waveforms displayed on timeline clips.
- [x] Video preview panel that syncs to playhead and active video clip.
- [x] Free sample media (tone + NASA image) bundled; Playwright screenshots refreshed.

## Core playback & media engine
- [ ] Real multi-track AV engine: simultaneous video+audio decode, smooth scrubbing, frame-accurate playhead.
- [ ] GPU-accelerated decoding (NVDEC/VAAPI/DXVA) or efficient wasm fallback.
- [ ] Continuous preroll/cache for scrubbing; smart proxy/offline handling.
- [ ] Robust timing: drop-frame/ND timecode, fps setting applied to timeline math.
  - [>] Track-aware playback: mute/solo/lock respected for audio/video; still needs multi-stream decode.

## Timeline operations
- [>] Trim modes: ripple/roll/slip/slide implemented; need media-offset clamp UI and edge-case gap handling polish.
- [ ] Stronger trim affordance: clip-edge hover handles + clearer hit areas.
- [>] Clip controls: per-track mute/solo/lock landed; linking/grouping/disable still pending.
- [ ] Nested sequences/subclips.
- [ ] Snapping refinements: edges/markers/gaps with configurable strength.
- [ ] Marquee multi-select respects track targeting; duplicate/paste maintains relative offsets.

## Waveforms & visuals
- [>] Waveforms on timeline: basic bars done; add high-res zoomed renders and per-channel colors.
- [>] Generate/carry thumbnails for video clips on timeline (thumb capture + display shipped; high-res/per-clip cache still TODO).
- [ ] Track height resizing; collapsed/expanded audio view.

## Media ingest/bin
- [ ] Bin system with folders, metadata columns, search/filter, watch folders.
- [ ] Proxy generation & relink; conform frame rate/timecode; LUT attachment.

## Effects & color
- [ ] Per-clip/track effects pipeline with keyframes.
- [ ] Basic color tools (Lift/Gamma/Gain), LUT apply; scopes (waveform/vectorscope/RGB parade).
- [ ] Transitions (crossfade/dip/film dissolve) and simple titles.

## Audio mixing
- [ ] Track/bus mixer UI with meters (peak/LUFS), pan, sends.
- [ ] Inline per-track mini meters in timeline during playback.
- [ ] Clip gain + keyframe envelopes; VST/AU host optional.
- [ ] Ducking/sidechain presets.

## Export/render
- [ ] Replace mock export with real ffmpeg/wasm pipeline; presets (H.264/HEVC/ProRes/WebM/Audio-only).
- [ ] Render queue, background render; smart render for mezzanine.
- [ ] Burn-in overlays and slate; per-preset loudness normalization.

## UX & monitors
- [ ] Program/source monitors with split-view, safe guides, rulers; fullscreen playback.
- [ ] JKL shuttle with variable rates; jog wheel support.
- [ ] Keyboard presets; command palette.
- [ ] Edit mode status pill near playhead (ripple/roll/overlap).

## Project/test infrastructure
- [ ] Autosave/backup/versions; portable project export (media collect).
- [ ] E2E coverage for playback (audio+video), trims, ripple edits, export.
- [ ] Perf/load tests on long timelines and large media sets.
- [ ] Cross-platform packaging (Electron/Tauri) with ffmpeg bundling and codec fallbacks.

## Original .sh feature parity
- [ ] Markers CSV ingest/export.
- [ ] Pre-normalize images, JPEG pre-save/quality controls.
- [ ] FX glow/impact toggles; shortest-mode; NVENC flag; ffmpeg/ffprobe path overrides.
- [ ] Quick/full render paths preserved.

## How to use this list
- Update this file whenever a task is started/completed.
- Keep checkboxes in sync with code changes; add dated notes inline if needed.
- If Codex restarts, reload this file to know current progress.
