#!/usr/bin/env bash
# =============================================================================
# fetch_onnxruntime.sh - download the prebuilt ONNX Runtime for this platform
# into lib/. The binaries are not committed; CI runs this via setup_script.
# =============================================================================
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.27.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_DIR="$HERE/lib"
mkdir -p "$LIB_DIR"

uname_s="$(uname -s)"
uname_m="$(uname -m)"

case "$uname_s" in
  Darwin)
    case "$uname_m" in
      arm64) pkg="onnxruntime-osx-arm64-${ORT_VERSION}" ;;
      x86_64) pkg="onnxruntime-osx-x86_64-${ORT_VERSION}" ;;
      *) echo "unsupported macOS arch: $uname_m" >&2; exit 1 ;;
    esac
    ext="tgz" ;;
  Linux)
    pkg="onnxruntime-linux-x64-${ORT_VERSION}"
    ext="tgz" ;;
  MINGW*|MSYS*|CYGWIN*)
    pkg="onnxruntime-win-x64-${ORT_VERSION}"
    ext="zip" ;;
  *)
    echo "unsupported OS: $uname_s" >&2; exit 1 ;;
esac

dest="$LIB_DIR/$pkg"
if [ -d "$dest" ]; then
  echo "tcxOnnx: ONNX Runtime already present: $dest"
  exit 0
fi

url="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${pkg}.${ext}"
echo "tcxOnnx: downloading $url"
tmp="$LIB_DIR/_ort_download.${ext}"
curl -fsSL -o "$tmp" "$url"

cd "$LIB_DIR"
if [ "$ext" = "tgz" ]; then
  tar xzf "$tmp"
elif command -v unzip >/dev/null 2>&1; then
  unzip -q "$tmp"
elif command -v 7z >/dev/null 2>&1; then
  7z x "$tmp" >/dev/null            # Windows CI runners ship 7z, not always unzip
else
  echo "tcxOnnx: need 'unzip' or '7z' to extract $tmp" >&2; exit 1
fi
rm -f "$tmp"
echo "tcxOnnx: extracted $dest"
