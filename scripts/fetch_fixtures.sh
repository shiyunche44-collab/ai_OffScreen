#!/usr/bin/env bash
# Test footage for development and golden regression runs. Output goes to data/fixtures/ (gitignored).
#
#   scripts/fetch_fixtures.sh download   # Sintel + Tears of Steel (Blender Foundation, CC-BY 3.0)
#   scripts/fetch_fixtures.sh slice      # 30 s slices of the downloaded films (fast unit/e2e inputs)
#   scripts/fetch_fixtures.sh synthetic  # offline: 30 s clip with a beep + white flash every 5 s
#   scripts/fetch_fixtures.sh all        # download + slice + synthetic
#
# Attribution: "Sintel" (c) Blender Foundation | durian.blender.org ; "Tears of Steel"
# (c) Blender Foundation | mango.blender.org ; both licensed CC-BY 3.0.
set -euo pipefail

OUT="$(cd "$(dirname "$0")/.." && pwd)/data/fixtures"
mkdir -p "$OUT"

SINTEL_URL="https://download.blender.org/durian/movies/Sintel.2010.720p.mkv.zip"
TOS_URL="https://download.blender.org/demo/movies/ToS/tears_of_steel_720p.mov"

download() {
  curl -fL --retry 3 -C - -o "$OUT/sintel.zip" "$SINTEL_URL"
  unzip -o -q "$OUT/sintel.zip" -d "$OUT" && mv "$OUT/Sintel.2010.720p.mkv" "$OUT/sintel.mkv" && rm "$OUT/sintel.zip"
  curl -fL --retry 3 -C - -o "$OUT/tears_of_steel.mov" "$TOS_URL"
}

slice() {
  for pair in "sintel.mkv:sintel_30s.mp4" "tears_of_steel.mov:tears_of_steel_30s.mp4"; do
    src="$OUT/${pair%%:*}"; dst="$OUT/${pair##*:}"
    [ -f "$src" ] || { echo "missing $src (run: $0 download)" >&2; exit 1; }
    # start at 60 s to skip titles; re-encode so the cut is frame-accurate
    ffmpeg -loglevel error -y -ss 60 -i "$src" -t 30 -c:v libx264 -preset veryfast -crf 23 -c:a aac "$dst"
    echo "wrote $dst"
  done
}

synthetic() {
  # Beep (1 kHz, 100 ms) and a full-white frame at t = 0, 5, 10, ... so audio/video sync
  # can be measured automatically (used by the M7-05 sync test).
  ffmpeg -loglevel error -y \
    -f lavfi -i "color=c=black:size=640x360:rate=24:duration=30" \
    -f lavfi -i "sine=frequency=1000:sample_rate=48000:duration=30" \
    -filter_complex "[0:v]drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='lt(mod(t\\,5)\\,0.04)'[v];\
[1:a]volume='if(lt(mod(t\\,5)\\,0.1)\\,0.5\\,0)':eval=frame,aformat=channel_layouts=stereo[a]" \
    -map "[v]" -map "[a]" -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest "$OUT/sync_30s.mp4"
  echo "wrote $OUT/sync_30s.mp4"
}

case "${1:-}" in
  download) download ;;
  slice) slice ;;
  synthetic) synthetic ;;
  all) download; slice; synthetic ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
