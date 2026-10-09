# CryWatch — ASUS UGen AI League 2026 entry

**Theme:** Everyday AI · **Battlefield:** Lightning (lightweight edge computing)
**Finals track:** Global Online Track (judged remotely)

An on-device AI baby monitor that doesn't just hear crying — it classifies
*why* the baby is crying (hungry, tired, belly pain, discomfort, burping,
too cold/hot), and never lets nursery audio leave the room.

## Why this wins

- **Real problem, personal stake.** Built by a dad of a 5-month-old. Misreading
  cries is the top stressor for new parents; today's monitors stream nursery
  audio to the cloud.
- **Real working prototype, honest metrics.** 70k-parameter neural net trained
  on 1,197 labeled clips, evaluated on a held-out test set. Numbers in
  `metrics.json` — no cherry-picking.
- **Built for the UGen300.** The model is tiny (<300 KB fp32, ~70 KB int8),
  mel-spectrogram front-end runs on a microcontroller-class CPU, and the
  forward pass is pure matrix math — a textbook Hailo-10H (40 TOPS INT4)
  INT8 deployment via ONNX in Stage II.
- **Privacy is the feature.** 2.5 W USB stick, no cloud, no subscription,
  no latency.

## Repo layout

- `crywatch/features.py` — wav → log-mel → pooled feature vector (numpy/scipy)
- `crywatch/model.py` — tiny MLP in pure numpy (480→128→64→9)
- `crywatch/train.py` — stratified split, class-weighted training, test metrics
- `crywatch/predict.py` — CLI demo: classify any wav clip
- `test_flow.py` — end-to-end verification (`ALL CHECKS PASSED`)
- `weights.npz`, `labels.json`, `metrics.json` — trained artifacts
- `pitch-deck.pdf` — Stage I proposal deck (≤20 pages)
- `VIDEO-SCRIPT.md` — 3-minute demo video script + shot plan

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install numpy scipy
.venv/bin/python crywatch/train.py     # trains, writes weights.npz + metrics.json
.venv/bin/python crywatch/predict.py path/to/cry.wav
.venv/bin/python test_flow.py
```

## Stage I submission checklist (deadline: Oct 14, 2026, 5:00 AM EDT)

- [x] Proposal deck ≤ 20 pages (`pitch-deck.pdf`)
- [x] Demo video ≤ 3 min — script + shot plan in `VIDEO-SCRIPT.md`
- [ ] **Record the video** (screen recording + voiceover, ~2.5 min) — needs Dakota or parent agent
- [ ] **Upload video to YouTube as Unlisted** — needs Dakota (his account)
- [x] GitHub code documentation: https://github.com/midnightmichigan/crywatch
- [ ] **Register + submit** at the official event page (contest platform) — needs Dakota's click
- [ ] Submission form fields: deck PDF upload, YouTube URL, GitHub URL, theme=Everyday AI, battlefield=Lightning, track=Global Online

## Needs from Dakota

1. **Video voiceover + YouTube upload** (or approve a TTS voiceover — his call).
2. **Registration/submission click** on the official event page (account signup is under his standing auth, but the final submit is his).
3. Nothing else — no spending, no ID, no travel (Global Online Track).

## Stage II plan (if shortlisted, announced Oct 26)

1. Quantize model to INT8, export ONNX, compile for Hailo-10H via Hailo SDK.
2. Deploy on the shipped UGen300 (USB stick the team keeps).
3. Build the nursery companion app (phone alert + guidance + cry diary).
4. 10-min final demo video + updated deck by Dec 4.
