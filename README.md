# highlight-reel

CPU-only render pipeline behind the **Highlight Reel** bot: it turns raw training/fight footage into looping
9:16 (1080×1920) hype montages with two licensed music versions and a no-music version, then runs an automatic QA gate.

What it does: proxies → narrative sequencing (beam search on an energy arc, match-on-action, session continuity) →
render from the original files (speed ramps, impact zooms, kinetic captions placed clear of faces, seamless loop) →
music mux (licensed bed mixed with the SFX layer) → loudness to −14 LUFS / ≤ −1 dBTP on the final AAC → QA
(format, loudness, loop-seam SSIM, flow, licenses, no text over faces, no text in the bottom 20%).

## Install (one line)
```bash
git clone https://github.com/ShopHeck/highlight-reel ~/highlight-reel && bash ~/highlight-reel/setup.sh
```
No git? `mkdir -p ~/highlight-reel && curl -L https://github.com/ShopHeck/highlight-reel/archive/refs/heads/main.tar.gz | tar xz --strip-components=1 -C ~/highlight-reel && bash ~/highlight-reel/setup.sh`

`setup.sh` is idempotent: apt-installs missing system libraries (libegl1, libgl1, libgles2, ffmpeg, fonts; needs sudo
only if something is missing), builds a venv (`~/highlight-reel/.venv`, or `$HR_VENV`), installs `requirements.txt`
and downloads the MediaPipe pose and face models into `models/`.

## Run
```bash
bash ~/highlight-reel/pipeline.sh my_job.json out_dir     # exit 0 = QA passed
bash ~/highlight-reel/run.sh <work_dir> license_check.py track_page.html --business
```
See `examples/job.example.json`: `footage` {name: path to the ORIGINAL file}, `hook`, `words`, `end_line`,
`style` (hype|fight|clean), `account` (creator|business), `tracks` [{file, bpm_hint, drop_beat}] (full licensed tracks,
beds are built by `build_bed.py`) or ready `beds`, and `licenses` {trackA: {title, artist, url, page}}.
Outputs: `<name>_trackA.mp4`, `_trackB.mp4`, `_nomusic.mp4`, `.qa.json`, `_contact.png`.

## Music licensing notes
- No music is included in this repo. You supply the tracks, and you are responsible for their licenses.
- Use only tracks you are licensed to use: a library license you hold (e.g. Epidemic, Artlist), CC0, or CC BY with the credit line.
  **Never NC or ND** for business/brand accounts, or for owners posting from both creator and business accounts.
- `license_check.py` reads the license from the track's own license field or the text next to its download link,
  and ignores page-wide badges (site footers can carry a different license than the track). Save the track page and
  pass it as `licenses.*.page`; QA fails if a license can't be verified.
- Never use commercial copyrighted songs. Trending sounds are added in-app (creator accounts only); business accounts
  use TikTok's Commercial Music Library or Meta's Sound Collection in-app.
- The renderer never generates a music bed; the no-music version is video + SFX only.

## License
MIT (see `LICENSE`). Third-party models (MediaPipe) and Python packages keep their own licenses.
