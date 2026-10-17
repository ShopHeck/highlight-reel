#!/usr/bin/env bash
# pipeline.sh <job.json> <out_dir>: footage -> proxies -> sequence -> render -> mux (licensed beds + SFX) -> QA. Exit 0 = QA pass.
# job.json adds to prepare.py's keys: "name", "account", "beds":[A.wav,B.wav], "licenses":{trackA:{title,artist,url,page},...}, "never_list"
set -euo pipefail
SK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; J="$(realpath "$1")"; OD="$(realpath -m "$2")"; W="$OD/work"; mkdir -p "$W"
N=$(python3 -c "import json;print(json.load(open('$J')).get('name','reel_9x16'))"); R="$SK/run.sh"
"$R" "$W" prepare.py "$J"
"$R" "$W" sequence.py "$W/in.json" "$W/$N.spec.json" > "$W/seq.log" 2> "$W/seq.err"
"$R" "$W" render_flow.py "$W/$N.spec.json" > "$W/render.log" 2> "$W/render.err"
# loop seam retry: if last-to-first SSIM < median consecutive SSIM, extend the morph to 15 frames and render again
if python3 -c "import json,sys;r=json.loads(open('$W/render.log').read().strip().splitlines()[-1]);sys.exit(0 if r['seam_ssim']<r['median_consecutive_ssim'] else 1)"; then
  echo "[pipeline] seam below median -> retry with a 15-frame morph" >&2
  python3 -c "import json;p='$W/$N.spec.json';d=json.load(open(p));d['loop_frames']=15;json.dump(d,open(p,'w'),indent=1)"
  "$R" "$W" render_flow.py "$W/$N.spec.json" > "$W/render.log" 2> "$W/render.err"
fi
RAW=$(python3 -c "import json;print(json.loads(open('$W/render.log').read().strip().splitlines()[-1])['file'])")
mv "$W/$RAW" "$W/${N}_render_sfx.mp4"; mv "$W/${RAW%.mp4}.ass" "$OD/"
# beds: given ready-made ("beds"), or built from full licensed tracks ("tracks":[{"file","bpm_hint","drop_beat"}]) with build_bed.py
BPMJ=$(python3 -c "import json;print(json.load(open('$J')).get('bpm',140))")
mapfile -t TRK < <(python3 -c "import json;[print(t['file'],t.get('bpm_hint',''),t.get('drop_beat','')) for t in json.load(open('$J')).get('tracks',[])]")
BEDS=(); i=0
for line in "${TRK[@]}"; do read -r tf bh dbt <<<"$line"; o="$W/bed_${i}.wav"
  "$R" "$W" build_bed.py "$tf" "$o" --out-bpm "$BPMJ" ${bh:+--bpm-hint $bh} ${dbt:+--drop-beat $dbt} > "$W/bed_${i}.log"; BEDS+=("$o"); i=$((i+1)); done
if [ ${#BEDS[@]} -eq 0 ]; then mapfile -t BEDS < <(python3 -c "import json;[print(b) for b in json.load(open('$J')).get('beds',[])]"); fi
"$R" "$W" mux.py "$W/${N}_render_sfx.mp4" "$OD" "$N" "${BEDS[@]}" > "$W/mux.log"
python3 - "$J" "$OD" "$W" "$N" > "$W/qa_job.json" <<'PY'
import json,sys;j=json.load(open(sys.argv[1]));od,w,n=sys.argv[2:5]
print(json.dumps(dict(out_dir=od,name=n,style=j.get('style','hype'),account=j.get('account','creator'),licenses=j.get('licenses',{}),never_list=j.get('never_list',[]),
  seq_spec=f'{w}/{n}.spec.json',render_json=f'{w}/render.log',mux_json=f'{od}/{n}.mux.json')))
PY
"$R" "$W" qa.py "$W/qa_job.json" 2> "$W/qa.err" | tee "$OD/qa.log"
