# Keyframe HDR

Bracketed RAWs in, finished luxury real estate images out.

Give it a folder of 3-shot brackets (50+ scenes is normal) and it produces, for every
scene, a full-resolution JPEG and a web JPEG in the Keyframe signature style: warm,
true to tone, contrasty, deep and sharp, with straight verticals, held window views and
natural skies. Nothing about the property is changed. The only content edits are the ones
you ask for: sky replacement, and removing photographers/people from reflections.

```
python -m keyframe_hdr <input_folder> <output_folder> --style day --sky clouds --max-mb 10
python -m keyframe_hdr <input_folder> <output_folder> --style twilight --look purple --sky clear
```

Output:

```
output/
  01_DSC_9935.jpg        full resolution (~45 MP), sRGB, q95, 4:4:4
  web/01_DSC_9935.jpg    2560 px long edge, sharpened for screens/portals
  report.json            per-image log: exposure, WB, upright, merge stats, people found
  edits.json             (optional, you write it) per-image overrides - see below
```

## Photo edits in the Keyframe portal (the everyday way)

Shoots are sent for editing from the portal: **book.keyframemedia.co.nz → Admin → Deliveries →
Photo edits**. It needs Deliveries access.

1. Press **New shoot** and fill in:
   - the property;
   - the Dropbox folder link of the bracketed RAWs (Canon, Nikon or DJI);
   - **Daytime** or **Twilight**, and the options:
     - **Daytime** sky: original, blue with clouds, or clear blue;
     - **Twilight**: natural or purple dusk, the lights enhancement on or off, and the sky
       (original, clear dusk or dusk with clouds).
2. The portal's `photo-edit-start` function fires the **Keyframe editing worker** Routine.
   - That's a fresh Claude Code cloud session per shoot, so several shoots edit at once.
   - It follows [`worker/WORKER.md`](worker/WORKER.md):
     - installs the tools (`scripts/setup_worker.sh`) and downloads the RAWs;
     - edits two brackets at a time;
     - checks every frame for people and photographers in reflections and removes them.
   - `worker/run_job.py` reports to the portal's `photo-edit-worker` function with a one-time
     token per job:
     - stage, progress and ETA every few minutes;
     - a check for the Stop button;
     - uploads of the finished JPEGs and previews to the private `photo-edits` bucket.
3. The shoot moves along **Uploaded → Editing → Edit complete**. Filters hide old jobs, and
   **Archive** tidies finished ones.
4. **Download all** saves every full-resolution JPEG (each under 10 MB) as one folder (.zip).

A 50-scene shoot takes about 2 hours, plus 10-15 minutes of setup.

Portal side (repo `Keyframemedia/NEW-keyframe-portal`):
- `src/components/admin/PhotoEdits.tsx`;
- migrations `20261016000001_photo_edits.sql` and `…02_photo_edits_worker_token.sql`;
- edge functions `photo-edit-start` and `photo-edit-worker`;
- the one secret: `CLAUDE_ROUTINE_TOKEN`, the routine's API trigger token.

## Working with Claude directly

1. Share a Dropbox folder link of the bracketed RAWs. Say **day** or **twilight**, and whether any
   skies should be replaced.
2. Claude downloads the folder, runs the pipeline, then reviews every frame itself. It checks
   reflections (glass, mirrors, TVs) and shadows for the photographer, and anyone else, and
   adds the removals to `jobs/<date>_<place>/edits.json`.
3. You get a private review gallery (before/after slider). Mark images Approve or Needs changes
   and leave a note.
4. Claude reads the notes back, re-edits only those images, and delivers the full-res and web JPEGs.

## What happens to each bracket

| Stage | What it does |
|---|---|
| Ingest | Groups frames into brackets by capture time + settings. Nikon **High Efficiency / HE\*** NEFs (TicoRAW, unreadable by open-source decoders) are converted with Adobe DNG Converter first. |
| Decode | Linear 16-bit decode (LibRaw, DHT demosaic, no clipping from white balance). Lens distortion and lateral CA from the DNG's own warp opcode (Adobe's built-in Nikon/Tamron profiles, DJI's own). Vignetting from Adobe LCP profiles, or Nikon's maker-note data at 75% strength. |
| Merge | Aligns frames (ECC, sub-pixel; homography for drone), refines exposure ratios from the data, merges in linear light with smooth, noise-optimal weights, suppresses ghosts (moving trees, clouds, people), renders highlights clipped in every frame (sun, glare on water) as clean white. |
| Clean | Edge-aware chroma and luminance noise reduction, purple-fringe suppression. |
| Upright | Detects vertical lines, estimates camera pitch/roll with the real focal length and re-projects to a level camera (two-point perspective), then crops to the largest clean 3:2 frame. No upsampling. Skipped for drone shots. |
| Tone | Hat-weighted exposure fusion of virtual exposures (each area gets the brightest rendering that is not clipped), so rooms are bright and walls clean, not the grey "HDR look". Skies and views blend toward one global exposure, so clouds stay white and natural instead of crunchy, then get their own fine local contrast (crisp hills and cloud edges). Auto black point, gentle white stretch, S-curve. |
| Grade | White balance measured off the room's own neutral surfaces and set to a warm-clean white (Planckian 5600 K). Sky-lit exteriors are only warmed a little, so blue skies stay blue. Clarity is weighted to midtones and switched off in skies. Oklab vibrance, foliage pulled warm, timber protected from oversaturation. Last, the house LUT (learned from Keyframe's delivered edits) sets the final colour rendering. |
| Output | Gamut-mapped to sRGB (soft, no clipped colours), output-sharpened per size, sRGB ICC + camera EXIF embedded. |

Everything is in `keyframe_hdr/presets.py` (`DAY`, `TWILIGHT`, `NIGHT`).

## Styles

- **day**: the signature daylight look, fitted to Keyframe's own delivered edits (see
  "How the house look was fitted" below).
- **twilight**: Keyframe's delivered dusk look, fitted the same way on Water Lily (dusk):
  bright facades, warm timber, glowing windows.
  - White balance is fixed rather than auto. Exteriors are rendered warm (8400 K source to a
    5200 K white). Interiors are found automatically (almost no sky outside the windows) and
    rendered tungsten-warm cream at 4200 K, with the dusk blue left in the windows.
  - With a replaced sky (`--sky clear|clouds`) the sky is drawn to the house twilight gradient,
    measured on the delivered dusk images: peach on the horizon, pink through the middle,
    lavender blue overhead. It is the same on every frame of a set whatever the exposure, and
    has soft, thin wisps placed by real azimuth and elevation, so they move as the camera turns.
  - `--look purple` uses a more violet gradient (or, with the original sky, turns that sky
    toward violet).
  - Lights enhancement (on by default; `--no-lights` turns it off) finds every lamp, downlight,
    wall light and lit window, adds a soft glow in its own colour, and lifts the pools of light
    they throw.
- **night**: the earlier, moodier dusk look (darker key, deep blue sky), kept for reference.

## How the house look was fitted

Each bracket of a delivered shoot was merged, rendered by the pipeline and aligned (ORB features
and a homography) to Keyframe's delivered JPEG of the same frame.
1. The tone and colour parameters were searched to minimise the Oklab difference over all pairs.
   Half the pairs were held out to check it generalises. Image statistics (brightness
   percentiles, local contrast, chroma) are matched as well, because a pixel-wise difference on
   slightly misaligned pairs on its own rewards a flat image.
2. For day, a 3D LUT (`keyframe_hdr/luts/day.npy`) carries the remaining colour rendering:
   creamy whites, timber, greens and sky blue. It is fitted on a coarse 13^3 lattice with a
   smoothness prior, then baked through a cubic spline into a 65^3 table, so smooth ceilings
   and skies never band. A least-squares fit on misaligned pairs also flattens tone, so the
   LUT's lightness is then matched to the delivered images' brightness distribution
   (histogram matching, which needs no alignment), and pure white is pinned to white.
3. For twilight the sky was masked out of the comparison, because the delivered skies are
   replaced. A LUT didn't generalise on the four pairs available, so twilight uses fitted
   parameters only.

## Sky replacement (`--sky clouds|clear`)

Skies come from 360-degree HDR sky domes: CC0 "pure skies" from Poly Haven, free for commercial
use with no attribution. Every exterior of a shoot gets the **same** sky, seen from that shot's
own direction, so the clouds change naturally from frame to frame instead of repeating.

- **Drone shots:** the gimbal's compass heading turns the dome.
- **Canon/Nikon (no compass):** shots are spread around it with a golden-angle sequence, keeping
  the sun behind the photographer.
- **Horizon:** the camera model from the upright step puts the dome's horizon on the real
  horizon.
- **Compositing:**
  - the sky goes in in linear light, before the grade;
  - edges are decontaminated, so no halo of the old sky survives around rooflines and posts;
  - a colour check learned from the image's own sky keeps soffits and pale walls from being
    mistaken for sky.

| Option | Daytime dome | Twilight dome |
|---|---|---|
| clouds | Kloofendal 48d partly cloudy | Belfast sunset (its clouds, in the house gradient) |
| clear | Syferfontein 18d clear / Kloofendal 43d clear | Rosendal park sunset (its shape, in the house gradient) |

The domes download on first use (~250 MB each at 16k) into `~/.cache/keyframe_skies`.

## Output size

`--max-mb 10` caps every full-resolution JPEG:
- quality steps down from 95 (4:4:4) only as far as an image needs;
- busy 45 MP frames land at about q90-91, which is visually lossless;
- simpler frames stay at q95.

## Per-image edits (`edits.json`)

Put an `edits.json` in the output folder and re-run. Keys are the first file of the bracket.
Coordinates are fractions of the finished image (0-1, x then y).

```json
{
  "DSC_9998": {"retouch": [{"box": [0.895, 0.40, 0.965, 0.60], "mode": "person"}]},
  "DSC_9974": {"retouch": [{"poly": [[0.31, 0.62], [0.47, 0.60], [0.50, 0.99], [0.30, 0.99]]}]},
  "DSC_0050": {"sky": "skies/day/blue_cumulus.jpg"},
  "DSC_0047": {"exposure_bias": -0.3},
  "DSC_0017": {"skip": true}
}
```

| Key | Meaning |
|---|---|
| `retouch` | List of regions to remove. `mode: "person"` finds the person/photographer inside the box (Mask R-CNN) and removes just them. `mode: "fill"` removes the whole box. `poly` removes a polygon (shadows, tripods, odd shapes). Filled with LaMa inpainting plus matched grain. |
| `sky` | Replace the sky: a path to a sky photo, or `"clear"` / `"dusk"` for a generated gradient. The sky is matted at full resolution (fine branches, mullions, including sky through windows), exposure-matched and composited **before** the grade, so it is graded like the rest of the image. |
| `exposure_bias` | ± stops on top of the automatic exposure. |
| any preset key | e.g. `"warmth_b": 0.02`, `"upright": false`, `"contrast": 0.4`. |
| `skip` | Leave this bracket out. |

Every image is also scanned for people automatically. Candidates (with boxes) are listed in
`report.json` under `people_candidates`, so reflections can be checked and confirmed rather
than auto-erased. That way a person in a framed print on the wall is never wiped.

## Client review gallery

```bash
python tools/make_gallery.py output/ raw/ gallery/ --thumbs previews/ --heading "Wānaka" --meta "1 Sept 2026 · …"
```

This builds a before/after contact sheet: each finished frame against the camera's own JPEG,
with a compare slider. It's published as a private Artifact. Each image can be marked
**Approve** or **Needs changes** with a note. The notes are stored with the page, so Claude can
read them back, turn them into `edits.json` entries and re-run only those images.

## Tests

`python -m pytest tests/ -q` runs fast smoke tests on synthetic data: merge accuracy, both
presets, in-place rendering and bracket grouping.

## Setup (Linux)

`bash scripts/setup_worker.sh` does all of this on a fresh Ubuntu container and skips whatever is
already installed. The studio's worker sessions run it first, which takes about 10-15 minutes.
By hand:

```bash
pip install -r requirements.txt
sudo apt install libimage-exiftool-perl wine wine64 wine32:i386 xvfb
# Adobe DNG Converter (only needed for Nikon HE / HE* NEFs)
xvfb-run wine AdobeDNGConverter_x64.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
# its C2PA module needs DirectML.dll (Microsoft redistributable, NuGet "Microsoft.AI.DirectML"):
cp DirectML.dll ~/.wine/drive_c/Program\ Files/Adobe/Adobe\ DNG\ Converter/
# LaMa inpainting model (Apache-2.0)
curl -L -o ~/.cache/keyframe_models/lama_fp32.onnx \
  https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx
```

The sky (UperNet-ConvNeXt, MIT) and person (torchvision Mask R-CNN, BSD) models download
on first use. All models are licensed for commercial use.

**Shooting tip:** set the Z8 to **Lossless compressed** NEF instead of High Efficiency\*.
The files are a bit bigger, but they decode directly, with no Adobe/Wine conversion step
(about 4 s per frame) and no dependency on Adobe's converter at all.

## Speed

About 4 minutes per bracket at full 45 MP. Peak memory is about 7 GB per bracket, and about
1.5 GB more for brackets with retouching. On a 16 GB machine, run two processes on the same
output folder, one plain and one with `--reverse`. They share the work through lock files,
and a 50-scene shoot takes about 1.5-2 hours. `--half` gives a quick quarter-size proof run.

## Known limits

- **Inpainted areas** are a little softer than their surroundings at 100%, because LaMa works
  at 512 px. They're invisible at listing sizes. Reflections in glass hide it best.
- **Twilight** is fitted on one dusk shoot (Water Lily) and checked against a second
  (Foxglove). Interiors at dusk had no matched pairs, so their white balance was set by eye
  against Foxglove's delivered interiors.
- **Canon CR3/CR2** go through the same Adobe DNG Converter step as Nikon, which gives Adobe's
  lens corrections. This hasn't been run on a real Canon shoot yet.
- **Sky replacement** works best from a curated library of clean, ungraded sky photos. Skies cut
  from finished images carry their grade with them.
- **Nikon HE\*** files need the Adobe DNG Converter step (Windows app under Wine).
