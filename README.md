# Killfeed-Detection
Detect killfeed events directly from CS2 video recordings without requiring demo files.

## Why I Built This

I started with a simple idea: the gameplay clips available through FACEIT Premium could become vertical 9:16 Shorts automatically. What attracted me most was the **asymmetric payoff**—spend the time building the system once, then let it turn future clips into shareable content without me having to edit and publish each one manually.

The editing and automation parts were manageable. The real problem was deciding **which kills in a video were actually mine**, and showing only those kills.

A normal video recording contains pixels, not a structured list of game events. The CS2 killfeed changes over time: entries appear, move as other kills happen, overlap with HUD elements, and disappear. Simply finding text or detecting a killfeed-shaped rectangle is not enough. The system has to recognize separate events, track them across frames, determine their timing, and distinguish my kills from everyone else's.

I looked into existing open-source projects and tools. Demo-based analyzers can access structured match events, but that doesn't help when the input is **only an MP4**. General-purpose OCR can read parts of the HUD, but reading text in a frame is not the same as reliably tracking individual killfeed events over time. I couldn't find a solution that fit this exact workflow out of the box, so I built one.

I'm sharing the detection part in case someone else runs into the same problem.

## What Makes This Project Different?

The goal isn't just to detect that a killfeed is visible. It's to extract **the right events from ordinary gameplay footage without requiring a CS2 demo file**.

This project combines visual killfeed detection, tracking of physical HUD rows across frames, event timing refinement, and player-name checks to help isolate kills belonging to the specified player. OCR assists with identifying the attacker; it is **not** the sole source of kill event detection.

Other approaches can solve parts of this problem. Demo parsers are useful when demo files exist, and OCR or object detection can find visual elements. The gap I wanted to fill was the combination of **video-only input + individual event tracking + own-kill filtering** for an unattended workflow. I'm not claiming this is the only possible solution—it's the one I needed and couldn't find ready-made.

## Intended Use and Current Scope

The original use case was an automated pipeline that takes FACEIT gameplay clips, identifies my kills, and uses those events to build vertical 9:16 Shorts. **This repository contains the killfeed detection engine, not the full video editor, clip acquisition system, or social media publisher.** It can be integrated into those kinds of pipelines.

The detector was developed primarily around **1080p, 60 FPS CS2 recordings**. Other resolutions, frame rates, HUD layouts, and compression levels may need further tuning and validation. Contributions that improve compatibility or detection accuracy are welcome.

The engine is still experimental: its checks are not a guarantee of perfectly identified kills, and not all stress-test scenarios have been completed.

## Integrating It Into Your Own System

**What you need**

- **Python 3.12** and the Python packages in `requirements.txt` (OpenCV and NumPy).
- **FFmpeg and ffprobe** installed and available on your system's PATH.
- **Tesseract OCR** with English language data installed and available on PATH.
- A **CS2 gameplay MP4** with the in-game killfeed visible, and the **player name** you want to track. No CS2 demo file, game connection, or API key is required.

**Run the detector**

```bash
python -m pip install -r requirements.txt
python detect.py --input gameplay.mp4 --output results --owner "PlayerName"
```

`--input` can also be a directory of MP4 files. The optional `--timeout` argument controls the per-video processing timeout (350 seconds by default).

**What happens to the video?**

1. FFmpeg decodes frames from the recorded gameplay.
2. OpenCV looks for the killfeed's visual elements and tracks individual rows across frames.
3. The engine separates kill events and refines their timestamps using additional source frames.
4. Tesseract helps check the attacker name so events belonging to other players can be excluded.
5. The program saves event metadata and the relevant killfeed image crops for another application to use.

**What your application receives**

- `results/<video-name>/rev12_final.json` — per-video status, event data, and detected onset timestamps (seconds into the original MP4).
- `results/<video-name>/summary.json` — structured event information, including ownership checks and paths to saved images.
- `results/<video-name>/runs/` — PNG killfeed crops, TSV event records, and diagnostic logs.
- `results/rev12_batch.json` — summary when processing one or more videos.

Your application can read these files and use the timestamps and image paths to drive its own editor, highlight generator, or other workflow. **This repository detects events; it does not cut videos into 9:16 Shorts or upload them anywhere.**

**Handling results safely:** Check the per-video `status` before consuming events. `COMPLETE` means the detector's internal checks passed, not that every kill is guaranteed correct. `SAFE_NO_OVERLAY` means detection was incomplete, and `RETRY_LOCKED` means another process holds the analysis lock. The public `detect.py` command uses strict mode and returns a nonzero exit code for incomplete results.
