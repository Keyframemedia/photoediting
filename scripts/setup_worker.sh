#!/usr/bin/env bash
# Prepare a fresh Linux container (Ubuntu, Python 3.11, root) to run Keyframe HDR jobs.
# Idempotent: steps that are already done are skipped. Takes ~10-15 min from scratch.
#
#   bash scripts/setup_worker.sh            # everything
#   KF_NO_WINE=1 bash scripts/setup_worker.sh  # skip Adobe DNG Converter (DJI-only shoots)
set -uo pipefail
T0=$(date +%s)
log() { echo "[setup +$(( $(date +%s) - T0 ))s] $*"; }
export DEBIAN_FRONTEND=noninteractive WINEDEBUG=-all WINEPREFIX=/root/.wine
CACHE=/root/.cache/keyframe_setup
mkdir -p "$CACHE" /root/.cache/keyframe_models

py_deps() {
  if python3 -c "import rawpy, cv2, torch, torchvision, transformers, onnxruntime, numexpr, scipy" 2>/dev/null; then
    log "python packages already installed"; return 0; fi
  log "installing python packages"
  pip install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision 2>&1 | tail -2
  pip install -q "numpy>=2.0" "opencv-python-headless>=4.10" "rawpy>=0.22" scipy numexpr pillow transformers onnxruntime 2>&1 | tail -2
  python3 -c "import rawpy, cv2, torch, transformers, onnxruntime" && log "python packages ok"
}

models() {
  local lama=/root/.cache/keyframe_models/lama_fp32.onnx
  if [ ! -s "$lama" ]; then
    log "downloading LaMa inpainting model"
    curl -sSL --retry 4 -o "$lama.part" https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx && mv "$lama.part" "$lama"
  fi
  log "prefetching sky + person models"
  python3 - <<'EOF' 2>&1 | grep -v -i warn | tail -2
import torchvision
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-small")
UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-small")
torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
print("models ok")
EOF
}

apt_base() {
  if command -v exiftool >/dev/null && command -v xvfb-run >/dev/null && command -v unzip >/dev/null; then
    log "exiftool/xvfb/unzip present"; return 0; fi
  log "apt: exiftool, xvfb, unzip"
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq --no-install-recommends libimage-exiftool-perl xvfb xauth unzip >/dev/null 2>&1
  command -v exiftool >/dev/null && log "exiftool ok"
}

dng_converter() {
  local exe="/root/.wine/drive_c/Program Files/Adobe/Adobe DNG Converter/Adobe DNG Converter.exe"
  if [ -f "$exe" ] && [ -f "$(dirname "$exe")/DirectML.dll" ]; then log "Adobe DNG Converter present"; return 0; fi
  if ! command -v wine >/dev/null || ! dpkg -s wine32:i386 >/dev/null 2>&1; then
    log "apt: wine (64 + 32 bit)"
    dpkg --add-architecture i386
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq --no-install-recommends wine wine64 wine32:i386 >/dev/null 2>&1
  fi
  [ -d /root/.wine/drive_c ] || { log "wineboot"; xvfb-run -a wineboot --init >/dev/null 2>&1; }
  if [ ! -s "$CACHE/AdobeDNGConverter.exe" ]; then
    log "downloading Adobe DNG Converter"
    curl -sSL --retry 4 -o "$CACHE/AdobeDNGConverter.exe.part" \
      https://download.adobe.com/pub/adobe/dng/win/AdobeDNGConverter_x64_18_7.exe \
      && mv "$CACHE/AdobeDNGConverter.exe.part" "$CACHE/AdobeDNGConverter.exe"
  fi
  log "installing Adobe DNG Converter (silent)"
  (cd "$CACHE" && timeout 1500 xvfb-run -a wine AdobeDNGConverter.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /SP- >/dev/null 2>&1)
  # its C2PA module needs DirectML.dll (Microsoft redistributable, NuGet Microsoft.AI.DirectML)
  if [ ! -s "$CACHE/dml/bin/x64-win/DirectML.dll" ]; then
    log "downloading DirectML.dll"
    curl -sSL --retry 4 -o "$CACHE/dml.nupkg" https://www.nuget.org/api/v2/package/Microsoft.AI.DirectML
    mkdir -p "$CACHE/dml" && (cd "$CACHE/dml" && unzip -qo ../dml.nupkg)
  fi
  cp "$CACHE/dml/bin/x64-win/DirectML.dll" "$(dirname "$exe")/" 2>/dev/null
  [ -f "$exe" ] && log "Adobe DNG Converter ok" || log "WARNING: Adobe DNG Converter not installed"
}

# python + models in parallel with the apt work (they don't touch the same files)
( py_deps && models ) > "$CACHE/py.log" 2>&1 &
PY=$!
apt_base
if [ "${KF_NO_WINE:-0}" != "1" ]; then dng_converter; fi
wait $PY; cat "$CACHE/py.log"
python3 -c "import rawpy, cv2, torch" 2>/dev/null && command -v exiftool >/dev/null \
  && { log "SETUP OK"; exit 0; } || { log "SETUP FAILED"; exit 1; }
