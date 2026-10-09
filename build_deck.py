"""Build pitch-deck.pdf for the ASUS UGen AI League Stage I submission.

Reads metrics.json for real numbers when available.
Run: .venv/bin/python build_deck.py
"""
import json
import os

from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, white, black
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGE_W, PAGE_H = 13.33 * inch, 7.5 * inch  # 16:9

BG = HexColor('#0E1526')
BG2 = HexColor('#16213B')
ACCENT = HexColor('#4CC9F0')
ACCENT2 = HexColor('#F72585')
TEXT = HexColor('#EDEDED')
MUTED = HexColor('#9AA5C4')
CARD = HexColor('#1B2547')

try:
    METRICS = json.load(open(os.path.join(ROOT, 'metrics.json')))
except Exception:
    METRICS = {}


def new_page(c, title, subtitle=None):
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # accent bar
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - 0.08 * inch, PAGE_W, 0.08 * inch, fill=1, stroke=0)
    c.setFillColor(TEXT)
    c.setFont('Helvetica-Bold', 30)
    c.drawString(0.7 * inch, PAGE_H - 0.95 * inch, title)
    if subtitle:
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 15)
        c.drawString(0.7 * inch, PAGE_H - 1.3 * inch, subtitle)
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 9)
    c.drawRightString(PAGE_W - 0.5 * inch, 0.35 * inch,
                      'CryWatch  |  ASUS UGen AI League 2026')


def bullet(c, x, y, text, size=17, color=TEXT, bold_prefix=None):
    c.setFillColor(ACCENT)
    c.circle(x, y - 5, 4, fill=1, stroke=0)
    c.setFillColor(color)
    if bold_prefix:
        c.setFont('Helvetica-Bold', size)
        c.drawString(x + 18, y - 7, bold_prefix)
        w = c.stringWidth(bold_prefix, 'Helvetica-Bold', size)
        c.setFont('Helvetica', size)
        c.drawString(x + 18 + w, y - 7, text)
    else:
        c.setFont('Helvetica', size)
        c.drawString(x + 18, y - 7, text)
    return y - 38


def card(c, x, y, w, h, title, value, sub=None):
    c.setFillColor(CARD)
    c.roundRect(x, y - h, w, h, 12, fill=1, stroke=0)
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 12)
    c.drawString(x + 20, y - 32, title)
    c.setFillColor(ACCENT)
    c.setFont('Helvetica-Bold', 34)
    c.drawString(x + 20, y - 78, value)
    if sub:
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 11)
        c.drawString(x + 20, y - h + 22, sub)


def build():
    path = os.path.join(ROOT, 'pitch-deck.pdf')
    c = canvas.Canvas(path, pagesize=(PAGE_W, PAGE_H))
    n_classes = len(METRICS.get('classes', [])) or 9
    acc = METRICS.get('test_accuracy')
    det = METRICS.get('cry_detection_accuracy')
    acc_s = f'{acc * 100:.1f}%' if acc else 'TBD'
    det_s = f'{det * 100:.1f}%' if det else 'TBD'

    # 1 — title
    c.setFillColor(BG)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(ACCENT)
    c.rect(0, PAGE_H - 0.08 * inch, PAGE_W, 0.08 * inch, fill=1, stroke=0)
    c.setFillColor(TEXT)
    c.setFont('Helvetica-Bold', 64)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 2.6 * inch, 'CryWatch')
    c.setFillColor(ACCENT)
    c.setFont('Helvetica', 24)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 3.4 * inch,
                        'On-device AI that understands why your baby is crying')
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 16)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 4.3 * inch,
                        'ASUS UGen AI League 2026  —  Everyday AI  ·  Battlefield Lightning  ·  Global Online Track')
    c.setFont('Helvetica', 13)
    c.drawCentredString(PAGE_W / 2, 1.1 * inch,
                        'Dakota McCall  ·  github.com/midnightmichigan/crywatch')
    c.showPage()

    # 2 — problem
    new_page(c, 'The 3 AM question', 'Every parent knows this sound. Nobody knows what it means.')
    y = PAGE_H - 2.1 * inch
    y = bullet(c, 0.9 * inch, y, 'Is she hungry? Tired? In pain? Too cold? Guess wrong and nobody sleeps.')
    y = bullet(c, 0.9 * inch, y, 'Not understanding cries is the most-cited stressor for first-time parents.')
    y = bullet(c, 0.9 * inch, y, "Today's “smart” monitors stream nursery audio to someone else's cloud servers —")
    y = bullet(c, 0.9 * inch, y, 'your baby\'s most vulnerable moments, stored who-knows-where, behind a subscription.')
    c.setFillColor(ACCENT2)
    c.setFont('Helvetica-Bold', 20)
    c.drawString(0.9 * inch, y - 30, 'Parents need answers, not audio streams. And the audio should never leave the room.')
    c.showPage()

    # 3 — why now
    new_page(c, 'Why now', 'Edge AI finally fits the nursery.')
    y = PAGE_H - 2.1 * inch
    y = bullet(c, 0.9 * inch, y, 'The ASUS UGen300 puts 40 TOPS (INT4) of neural compute on a 2.5 W USB stick —')
    y = bullet(c, 0.9 * inch, y, 'enough to run audio AI continuously, on-device, for pennies of electricity.')
    y = bullet(c, 0.9 * inch, y, 'Parents are rejecting cloud baby-tech over privacy fears — on-device is the differentiator.')
    y = bullet(c, 0.9 * inch, y, 'Cry-reason classification is an unsolved consumer problem: detection exists, understanding doesn\'t.')
    c.showPage()

    # 4 — solution
    new_page(c, 'CryWatch', 'Hear the cry. Know the reason. Keep it private.')
    y = PAGE_H - 2.1 * inch
    y = bullet(c, 0.9 * inch, y, 'A nursery microphone listens locally — nothing is recorded to the cloud, ever.')
    y = bullet(c, 0.9 * inch, y, f'A tiny neural net classifies each cry into {n_classes} reasons: hungry, tired, belly pain, discomfort,')
    y = bullet(c, 0.9 * inch, y, 'burping, too cold/hot — and tells cry apart from laughter, background noise, and silence.')
    y = bullet(c, 0.9 * inch, y, 'Each prediction ships with parent-friendly guidance: “likely hunger — check when the last feed was.”')
    y = bullet(c, 0.9 * inch, y, 'Runs in milliseconds on CPU today; Stage II target: the full pipeline on the UGen300.')
    c.showPage()

    # 5 — results (real metrics)
    new_page(c, 'It works — measured, not promised', 'Held-out test set. Clips the model never saw in training.')
    card(c, 0.9 * inch, PAGE_H - 2.2 * inch, 3.4 * inch, 1.7 * inch,
         'TEST ACCURACY', acc_s, f'{n_classes} cry reasons + background')
    card(c, 4.7 * inch, PAGE_H - 2.2 * inch, 3.4 * inch, 1.7 * inch,
         'CRY DETECTION', det_s, 'cry vs laugh / noise / silence')
    card(c, 8.5 * inch, PAGE_H - 2.2 * inch, 3.4 * inch, 1.7 * inch,
         'MODEL SIZE', '~70K params', '<300 KB fp32 · ~70 KB int8')
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 13)
    c.drawString(0.9 * inch, 1.5 * inch,
                 'Training: 1,197 labeled clips · stratified 70/15/15 split · class-weighted loss · full code on GitHub.')
    c.setFillColor(TEXT)
    c.setFont('Helvetica', 13)
    per = METRICS.get('per_class_accuracy', {})
    if per:
        c.drawString(0.9 * inch, 1.05 * inch,
                     'Per-class test accuracy: ' + ' · '.join(
                         f'{k} {v * 100:.0f}%' for k, v in sorted(per.items())))
    c.showPage()

    # 6 — how it works
    new_page(c, 'How it works', 'Classical signal processing + a tiny learned classifier.')
    steps = [('MIC', 'nursery audio\n8 kHz'), ('FEATURES', 'log-mel\nspectrogram\n40 bands'),
             ('POOL', 'segmented\nmean+std\n480-dim vector'), ('MLP', '480→128→64→9\nReLU · softmax'),
             ('OUTPUT', 'cry reason\n+ confidence\n+ guidance')]
    x0, bw, gap = 0.9 * inch, 1.9 * inch, 0.45 * inch
    cy = PAGE_H / 2 + 0.4 * inch
    for i, (t, sub) in enumerate(steps):
        x = x0 + i * (bw + gap)
        c.setFillColor(CARD)
        c.roundRect(x, cy - 0.9 * inch, bw, 1.8 * inch, 12, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.setFont('Helvetica-Bold', 18)
        c.drawCentredString(x + bw / 2, cy + 0.45 * inch, t)
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 12)
        for j, line in enumerate(sub.split('\n')):
            c.drawCentredString(x + bw / 2, cy - 0.05 * inch - j * 20, line)
        if i < len(steps) - 1:
            c.setFillColor(ACCENT)
            c.setFont('Helvetica-Bold', 22)
            c.drawString(x + bw + 8, cy - 8, '→')
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 13)
    c.drawString(0.9 * inch, 1.4 * inch,
                 'The front-end is fixed math (no learned weights); only the 70K-param MLP is learned —')
    c.drawString(0.9 * inch, 1.05 * inch,
                 'which is exactly the part that quantizes cleanly to INT8 for the Hailo-10H NPU.')
    c.showPage()

    # 7 — model card
    new_page(c, 'The model', 'Small on purpose: built for the edge, not the datacenter.')
    y = PAGE_H - 2.1 * inch
    y = bullet(c, 0.9 * inch, y, 'Architecture: 480 → 128 → 64 → 9, ReLU, softmax. ~70,000 parameters.')
    y = bullet(c, 0.9 * inch, y, 'Footprint: ~280 KB fp32, ~70 KB after INT8 quantization — fits in NPU SRAM.')
    y = bullet(c, 0.9 * inch, y, 'Training: pure NumPy + SciPy, CPU-only, minutes — no GPU cluster, no API bills.')
    y = bullet(c, 0.9 * inch, y, 'Inference: milliseconds per 7-second clip on an ordinary laptop CPU.')
    y = bullet(c, 0.9 * inch, y, 'Imbalance handled: inverse-frequency class weights (hungry dominates the dataset).')
    y = bullet(c, 0.9 * inch, y, 'Reproducible: fixed seeds, stratified splits, every number regenerable from this repo.')
    c.showPage()

    # 8 — HW/SW architecture
    new_page(c, 'Hardware / software architecture', 'Stage II target: the entire loop on the UGen300.')
    boxes = [
        ('Nursery mic\narray', 'always-on\n2.5 W budget'),
        ('Host CPU\n(x86/ARM)', 'log-mel features\nvoice activity gate'),
        ('UGen300\nHailo-10H', 'INT8 MLP inference\n20 TOPS headroom'),
        ('Phone app', 'alert + reason\n+ guidance + diary'),
    ]
    x0, bw, gap = 0.7 * inch, 2.5 * inch, 0.5 * inch
    cy = PAGE_H / 2 + 0.5 * inch
    for i, (t, sub) in enumerate(boxes):
        x = x0 + i * (bw + gap)
        c.setFillColor(CARD if i != 2 else HexColor('#243B6B'))
        c.roundRect(x, cy - 1.0 * inch, bw, 2.0 * inch, 12, fill=1, stroke=0)
        c.setFillColor(ACCENT if i == 2 else TEXT)
        c.setFont('Helvetica-Bold', 17)
        for j, line in enumerate(t.split('\n')):
            c.drawCentredString(x + bw / 2, cy + 0.5 * inch - j * 24, line)
        c.setFillColor(MUTED)
        c.setFont('Helvetica', 12)
        for j, line in enumerate(sub.split('\n')):
            c.drawCentredString(x + bw / 2, cy - 0.25 * inch - j * 20, line)
        if i < len(boxes) - 1:
            c.setFillColor(ACCENT)
            c.setFont('Helvetica-Bold', 22)
            c.drawString(x + bw + 10, cy - 8, '→')
    c.setFillColor(ACCENT2)
    c.setFont('Helvetica-Bold', 15)
    c.drawCentredString(PAGE_W / 2, 1.35 * inch,
                        'No cloud in the loop. Audio never leaves the home.')
    c.setFillColor(MUTED)
    c.setFont('Helvetica', 12)
    c.drawCentredString(PAGE_W / 2, 0.95 * inch,
                        'Interfaces: USB-C (UGen300) · Linux/Windows/Android hosts · PyTorch → ONNX → Hailo SDK')
    c.showPage()

    # 9 — expected outcomes
    new_page(c, 'Expected outcomes', 'What winning looks like.')
    y = PAGE_H - 2.1 * inch
    y = bullet(c, 0.9 * inch, y, 'Stage I: shortlist — working prototype, honest metrics, credible edge deployment plan.')
    y = bullet(c, 0.9 * inch, y, 'Stage II: INT8 model running live on the UGen300; nursery app demo; ≥90% cry-detection accuracy.')
    y = bullet(c, 0.9 * inch, y, 'Product: a $99-class nursery device or USB add-on for existing monitors — no subscription.')
    y = bullet(c, 0.9 * inch, y, 'Impact: fewer 3 AM guessing games; nursery audio stays in the nursery.')
    y = bullet(c, 0.9 * inch, y, 'Commercial path: license to baby-monitor makers; direct device via the UGen300 USB form factor.')
    c.showPage()

    # 10 — roadmap
    new_page(c, 'Stage II roadmap', 'Nov 6 – Dec 4 development sprint.')
    y = PAGE_H - 2.1 * inch
    for label, text in [
        ('Wk 1–2', 'Quantize to INT8, export ONNX, compile with the Hailo SDK; validate numerics vs fp32.'),
        ('Wk 2–3', 'Deploy on the shipped UGen300; measure latency + power; tune the voice-activity gate.'),
        ('Wk 3–4', 'Build the nursery companion app: alerts, reason + guidance, cry diary.'),
        ('Wk 4', '10-minute final demo video, updated deck, full repo — final submission Dec 4.'),
    ]:
        y = bullet(c, 0.9 * inch, y, text, bold_prefix=label + '  ')
    c.showPage()

    # 11 — team
    new_page(c, 'Team', 'One builder, personal stake, AI-assisted engineering.')
    y = PAGE_H - 2.1 * inch
    y = bullet(c, 0.9 * inch, y, 'Dakota McCall — builder. Dad of a 5-month-old; this product exists because his nursery needs it.')
    y = bullet(c, 0.9 * inch, y, 'Engineering: AI-assisted development (Muse) — every line reviewed, every metric reproduced from the repo.')
    y = bullet(c, 0.9 * inch, y, 'Track: Global Online — judged remotely, no travel required.')
    c.showPage()

    # 12 — references
    new_page(c, 'References', '')
    y = PAGE_H - 2.1 * inch
    refs = [
        'ASUS UGen300 M2/USB AI Module — Hailo-10H, 40 TOPS INT4, 2.5 W (asus.com).',
        'ASUS UGen AI League Innovation Hackathon — official rules (contest platform).',
        'Hailo-10H overview — Hailo developer zone; HailoRT SDK + ONNX support.',
        'Baby cry sound dataset — 1,197 labeled clips, 9 classes (Hugging Face, mahmudulhasan01).',
        'Infant cry acoustics — fundamentals 300–600 Hz; reason-from-cry classification literature.',
    ]
    for r in refs:
        y = bullet(c, 0.9 * inch, y, r, size=14)
    c.setFillColor(ACCENT)
    c.setFont('Helvetica-Bold', 16)
    c.drawString(0.9 * inch, 1.2 * inch, 'Code & docs: github.com/midnightmichigan/crywatch')
    c.showPage()

    c.save()
    print('wrote', path)


if __name__ == '__main__':
    build()
