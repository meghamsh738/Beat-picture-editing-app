# Timeline Builder (React + Vite)

Premiere-style timeline prototype with snapping, loop ranges, ripple edits, asset ingest, and export presets. Ships with free sample media and an automated Playwright smoke that records a video + screenshot.

## Quick start
1) Install deps: `npm install`
2) Dev server: `npm run dev -- --host --port 4178`
3) Open http://localhost:4178

## Sample media
Free-to-use files are bundled in `public/samples/`:
- `sample-5s.mp4` – video preview (Samplelib)
- `sample-3s.wav` – audio clip (Samplelib)
- `sample-photo.jpg` – still image (Samplelib)
Use the **Assets** tab file picker or drag these into the timeline tracks.

## Features exercised in the UI
- Tracks with draggable clips, trim handles, marquee multi-select, undo/redo.
- Snap refinement: grid/marker/gap/edge targets with a labeled snap ghost.
- Ripple edits toggle to shift downstream clips on move/trim.
- Loop range with minimap handles and main-timeline overlay; playhead looping.
- Zoom & pan: wheel+modifier zoom, Shift+wheel horizontal pan, zoom-to-selection.
- Asset bin: duration/metadata, waveform (audio) + image thumb, drag or “Send to track”.
- Markers list is clickable to jump the playhead.
- Export presets: JSON bundle plus mock MP4/WebM render download.

## Keyboard bits
- Space play/pause; J/K/L shuttle; Arrow left/right nudge selected clips (Shift = bigger step).
- Cmd/Ctrl+Z / Shift+Cmd/Ctrl+Z undo/redo; Cmd/Ctrl+D duplicate selected clips.

## Testing & captures
- E2E smoke (records video + screenshot): `npm run test:e2e`
  - Outputs: `screenshots/timeline.png` and `screenshots/timeline-run.webm`.
- Production build: `npm run build`

## File map
- `src/App.tsx` – main timeline UI & interactions
- `src/App.css` – styling
- `playwright.config.ts`, `tests/smoke.spec.ts` – smoke coverage (snap, loop, assets, export)
- `screenshots/` – latest screenshot & test video
- `public/samples/` – bundled demo media

## Notes
- Export rendering is mocked; swap the placeholder encoder with ffmpeg/wasm when ready.
- Waveform decoding uses Web Audio; falls back gracefully if unavailable.
