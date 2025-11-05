#!/usr/bin/env bash
set -euo pipefail

# Tiny launcher for scripts/tools/mix_wavs.py
#
# Examples:
#   # 3 路对齐混合，设置相对 SNR（相对第 1 路）
#   ./mix_wavs.sh --out mix.wav --sr 16000 --snr 0,5,10 a.wav b.wav c.wav
#
#   # 加起始偏移（秒）
#   ./mix_wavs.sh --out mix.wav --sr 16000 --snr 0,5,10 --offsets 0,1.2,2.5 a.wav b.wav c.wav
#
#   # 使用绝对增益（dB）而不是 SNR
#   ./mix_wavs.sh --out mix.wav --sr 16000 --gains-db 0,-6 s0.wav s1.wav

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PY=${SCRIPT_DIR}/mix_wavs.py

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] <input1.wav> [input2.wav ...]

Options:
  --out PATH         输出混合 WAV 路径 (必填)
  --sr INT           目标采样率，默认 16000
  --offsets LIST     每路起始偏移（秒），逗号分隔，如 0,1.2,2.5（长度=输入数；给一个值则复用）
  --snr LIST         相对 SNR（dB，相对第 1 路），逗号分隔，如 0,5,10（正数=更安静）
  --gains-db LIST    绝对增益（dB），逗号分隔，如 0,-6,-12（与 --snr 互斥）
  --peak FLOAT       峰值限制（0..1），默认 0.98，避免削顶
  -h, --help         显示帮助

示例：
  $(basename "$0") --out mix.wav --sr 16000 --snr 0,5,10 a.wav b.wav c.wav
  $(basename "$0") --out mix.wav --sr 16000 --snr 0,5,10 --offsets 0,1.2,2.5 a.wav b.wav c.wav
  $(basename "$0") --out mix.wav --sr 16000 --gains-db 0,-6 s0.wav s1.wav
EOF
}

OUT=""
SR=16000
OFFSETS=""
SNR=""
GAINS_DB=""
PEAK=0.98

# Parse options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT=${2:-}; shift 2 ;;
    --sr) SR=${2:-16000}; shift 2 ;;
    --offsets) OFFSETS=${2:-}; shift 2 ;;
    --snr) SNR=${2:-}; shift 2 ;;
    --gains-db) GAINS_DB=${2:-}; shift 2 ;;
    --peak) PEAK=${2:-0.98}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "[mix_wavs.sh] Unknown option: $1" >&2; usage; exit 2 ;;
    *) break ;;
  esac
done

INPUTS=("$@")

if [[ -z "${OUT}" ]]; then
  echo "[mix_wavs.sh] --out 是必填参数" >&2
  usage
  exit 2
fi

if [[ ${#INPUTS[@]} -lt 1 ]]; then
  echo "[mix_wavs.sh] 需要至少一个输入 WAV" >&2
  usage
  exit 2
fi

if [[ -n "${SNR}" && -n "${GAINS_DB}" ]]; then
  echo "[mix_wavs.sh] --snr 与 --gains-db 互斥，请二选一" >&2
  exit 2
fi

CMD=( python3 "${PY}" )
for f in "${INPUTS[@]}"; do CMD+=("${f}"); done
CMD+=("--out" "${OUT}" "--sr" "${SR}" "--peak" "${PEAK}")

if [[ -n "${OFFSETS}" ]]; then CMD+=("--offsets" "${OFFSETS}"); fi
if [[ -n "${SNR}" ]]; then CMD+=("--snr" "${SNR}"); fi
if [[ -n "${GAINS_DB}" ]]; then CMD+=("--gains-db" "${GAINS_DB}"); fi

echo "[mix_wavs.sh] Running: ${CMD[*]}"
"${CMD[@]}"
