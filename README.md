# Beat Picture Editing App (Timeline Builder)

A desktop timeline builder built with PySide6. Features include snapping (markers/edges/gaps), ripple moves/trims, loop ranges, asset ingest with waveforms/thumbs, zoom/pan controls, beat detection, text-on-beat overlays, and export presets.

Quick start (desktop app)
- `bash run_timeline.sh`
  - Creates `.venv` + installs `requirements.txt`
  - Launches the timeline builder UI

Sample media
- You can point the app at any folder of images/audio.
- The app also supports marker CSVs and timeline export via ffmpeg (see in-app help/tooltips).

Current feature set
- Audio + video playback synced to playhead; track-aware mute/solo/lock; loop ranges.
- Timeline trims: ripple, roll, slip, slide (alt+trim handles); snapping to markers/edges.
- Clip visuals: audio waveforms on clips; video thumbnails captured on import.
- Individual track resizing and quick asset assignment tools.

More details
See `web/README.md` if you want the separate web prototype notes.
