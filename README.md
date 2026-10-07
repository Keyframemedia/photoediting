# Keyframe HDR

Bracketed RAWs in, finished luxury real estate images out.

Give it a folder of 3-shot brackets (50+ scenes is normal) and it produces, for every
scene, a full-resolution JPEG and a web JPEG in the Keyframe signature style: warm,
true to tone, contrasty, deep and sharp, with straight verticals, held window views and
natural skies. Nothing about the property is changed. The only content edits are the ones
you ask for: sky replacement, and removing photographers/people from reflections.

```
python -m keyframe_hdr <input_folder> <output_folder> --style day     # or --style night
```

Output:

```
output/
  01_DSC_9935.jpg        full resolution (~45 MP), sRGB, q95, 4:4:4
  web/01_DSC_9935.jpg    2560 px long edge, sharpened for screens/portals
  report.json            per-image log: exposure, WB, upright, merge stats, people found
  edits.json             (optional, you write it) per-image overrides - see below
```

## Working with Claude

1. Share a Dropbox folder link of the bracketed RAWs. Say **day** or **night**, and whether any
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
| Tone | Hat-weighted exposure fusion of virtual exposures (each area gets the brightest rendering that is not clipped), so rooms are bright and walls clean, not the grey "HDR look". Skies and views blend toward one global exposure, so clouds stay white and natural instead of crunchy. Auto black point, gentle white stretch, S-curve. |
| Grade | White balance measured off the room's own neutral surfaces and set to a warm-clean white (Planckian 5200 K). Sky-lit exteriors are only warmed a little, so blue skies stay blue. Clarity is weighted to midtones and switched off in skies. Oklab vibrance, richer sky blue, foliage pulled warm, timber protected from oversaturation. |
| Output | Gamut-mapped to sRGB (soft, no clipped colours), output-sharpened per size, sRGB ICC + camera EXIF embedded. |

Everything is in `keyframe_hdr/presets.py` (`DAY`, `NIGHT`).

## Styles

- **day**: the signature daylight look described above. Tuned on a real 50-scene shoot.
- **night**: twilight/dusk. White balance is fixed rather than auto, so interiors glow warm
  against a deep blue sky. It has a darker key, a wider exposure range for light fittings,
  stronger noise reduction and a richer sky. **Tuned blind.** It needs one real twilight set
  to finalise.

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
- **Night preset** hasn't been tuned on real twilight brackets yet.
- **Sky replacement** works best from a curated library of clean, ungraded sky photos. Skies cut
  from finished images carry their grade with them.
- **Nikon HE\*** files need the Adobe DNG Converter step (Windows app under Wine).
