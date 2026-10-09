# CryWatch — Stage I Demo Video: Script & Shot Plan

**Format:** screen recording + voiceover, no human on camera.
**Length:** ≤ 3:00 (target 2:30). **Upload:** YouTube, visibility "Anyone with the link".
**Tools:** any screen recorder (OBS); slides from `pitch-deck.pdf` for the intro/outro cards.

---

## Shot 0 — Title card (0:00–0:10)
**Visual:** pitch-deck title slide full-screen.
**Voiceover:** "CryWatch — an on-device AI baby monitor that doesn't just hear your baby crying. It understands *why*. Built for the ASUS UGen AI League, Everyday AI theme, Battlefield Lightning."

## Shot 1 — The problem (0:10–0:35)
**Visual:** two quick slides: (1) "Every new parent knows this sound" + stat card — parents of newborns lose ~100+ hours of sleep in year one; misreading cries is the #1 stressor in surveys of first-time parents. (2) "Cloud baby monitors send your nursery audio to someone else's server."
**Voiceover:** "Every parent knows the 3 AM cry. Is she hungry? Tired? In pain? Guess wrong and everyone stays up. Today's smart monitors stream that audio to the cloud — your baby's most vulnerable moments, on someone else's server. CryWatch fixes both: it classifies the *reason* for the cry, and it never lets audio leave the room."

## Shot 2 — Live demo, cry vs. non-cry (0:35–1:20)
**Visual:** terminal. Run:
```
.venv/bin/python crywatch/predict.py data/baby-cry/baby_crying_sound/belly\ pain/421c.wav
```
Show output: `crying: True`, belly pain 84%, guidance line, latency "x ms on CPU".
Then run `data/baby-cry/baby_crying_sound/silence/silence.wav_17.wav` → `crying: False`, silence 97%.
**Voiceover:** "Here's the working prototype. Feed it a nursery recording — it tells you in milliseconds whether that's a cry, and if so, what the baby most likely needs, plus what to try first. This runs on an ordinary CPU today. The whole model is 70,000 parameters — small enough to live entirely on the UGen300's neural accelerator."

## Shot 3 — How well does it work (1:20–1:50)
**Visual:** slide with the real test metrics (from `metrics.json`): test accuracy, cry-detection accuracy, per-class bars. "Trained on 1,197 labeled clips across 9 classes. Held-out test set. No cherry-picking."
**Voiceover:** "Honest numbers, measured on clips the model never saw in training: [X]% overall accuracy across nine cry reasons and background sounds, [Y]% on the core job — telling a cry from background noise. The full confusion matrix and training code are on our GitHub."

## Shot 4 — How it works / Stage II plan (1:50–2:20)
**Visual:** architecture diagram slide: mic → log-mel features → 70k-param MLP → ONNX → Hailo-10H NPU on UGen300 (USB stick, 2.5 W) → phone app alert. Then roadmap slide: quantize to INT8, deploy on the UGen300, build the nursery app.
**Voiceover:** "Audio becomes a mel spectrogram, a tiny neural net classifies it, and in Stage II that net gets quantized and deployed to the UGen300 — a USB stick that adds 40 trillion operations per second at two and a half watts. No cloud, no subscription, no latency. Then we build the nursery app around it."

## Shot 5 — Close (2:20–2:30)
**Visual:** title slide again + GitHub URL + "Global Online Track".
**Voiceover:** "CryWatch — every cry understood, every byte stays home. Code, metrics, and proposal are linked below."

---

## Recording checklist
- [ ] Terminal font ≥ 18pt, high contrast
- [ ] Pre-run the demo commands once (warm the feature cache)
- [ ] Pick one strong hungry clip + one silence clip; note exact filenames here before recording
- [ ] Record at 1080p, export ≤ 3:00
- [ ] Upload to YouTube as Unlisted, paste URL into the submission form
- [ ] Captions: YouTube auto-captions, quick skim for errors
