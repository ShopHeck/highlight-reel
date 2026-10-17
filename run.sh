#!/usr/bin/env bash
# run.sh <work_dir> <script.py> [args]: runs a skill script with the skill venv, HR_WORK=<work_dir>, HR_MODELS=<skill>/models.
# Paths in args are resolved against the caller's cwd first. Works from any cwd; no PYTHONPATH needed.
set -euo pipefail
SK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; VENV="${HR_VENV:-$(cat "$SK/.venv_path" 2>/dev/null || echo "$SK/.venv")}"
[ -x "$VENV/bin/python" ] || { echo "run $SK/setup.sh first" >&2; exit 2; }
W="$(realpath -m "$1")"; S="$2"; shift 2; mkdir -p "$W/out" "$W/flow" "$W/proxy"
args=(); for a in "$@"; do if [[ "$a" != -* && ( -e "$a" || "$a" == */* || "$a" =~ \.(json|mp4|mov|wav|mp3|png)$ ) ]]; then args+=("$(realpath -m "$a")"); else args+=("$a"); fi; done
export HR_WORK="$W" HR_MODELS="$SK/models" PYTHONPATH="$SK/scripts" PYTHONDONTWRITEBYTECODE=1
cd "$W" && case "$S" in /*) ;; *) S="$SK/scripts/$S";; esac; exec "$VENV/bin/python" "$S" "${args[@]}"
