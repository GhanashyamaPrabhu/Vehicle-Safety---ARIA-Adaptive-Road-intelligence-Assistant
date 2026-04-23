#!/usr/bin/env python3
"""
Tech Summit — Vehicle Safety AI
Generates:  qr_code.png  +  poster.png  (A4 print-ready)
Usage:  python3 make_poster.py "https://forms.gle/2bQTya2cWLJGBFKJ7"
"""

import sys
import qrcode
from PIL import Image, ImageDraw, ImageFont
import os

FORM_URL   = sys.argv[1] if len(sys.argv) > 1 else "https://forms.gle/2bQTya2cWLJGBFKJ7"
OUT_QR     = "qr_code.png"
OUT_POSTER = "poster.png"
W, H = 874, 1240

# ── CHEERFUL PALETTE ─────────────────────────────────────────
BG      = (255, 255, 255)       # clean white canvas
ORANGE  = (255, 111,  0)        # warm orange — headline, CTA
PURPLE  = (111,  48, 210)       # vivid purple — prize
CORAL   = (255,  75,  75)       # coral red — accents
YELLOW  = (255, 213,   0)       # sunshine yellow — badge
GREEN   = ( 16, 185, 129)       # fresh mint — features
BLUE    = ( 59, 130, 246)       # sky blue — QR border
DARK    = ( 25,  20,  50)       # near-black for body text
MID     = (100,  90, 130)       # muted purple-grey
CONFETTI = [
    (255, 111,   0),
    (111,  48, 210),
    (255, 213,   0),
    ( 16, 185, 129),
    ( 59, 130, 246),
    (255,  75,  75),
]

# ── QR CODE ──────────────────────────────────────────────────
print(f"[1/2] Generating QR code for: {FORM_URL}")
qr = qrcode.QRCode(version=3,
                   error_correction=qrcode.constants.ERROR_CORRECT_H,
                   box_size=10, border=2)
qr.add_data(FORM_URL)
qr.make(fit=True)
qr_img = qr.make_image(fill_color=DARK, back_color=BG).convert("RGB")
qr_img = qr_img.resize((300, 300), Image.LANCZOS)
qr_img.save(OUT_QR)
print(f"    Saved → {OUT_QR}")

# ── POSTER ───────────────────────────────────────────────────
print("[2/2] Building poster …")
poster = Image.new("RGB", (W, H), BG)
draw   = ImageDraw.Draw(poster)

def font(size, bold=False):
    for path in [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/liberation/LiberationSans{'-Bold' if bold else '-Regular'}.ttf",
        f"/usr/share/fonts/truetype/freefont/FreeSans{'Bold' if bold else ''}.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def cx(y, text, f, colour):
    b = draw.textbbox((0,0), text, font=f)
    draw.text(((W - (b[2]-b[0])) // 2, y), text, font=f, fill=colour)
    return b[3] - b[1]

# ── CONFETTI DOTS (fixed seed pattern) ───────────────────────
dots = [
    (44,  55, 14), (810, 42, 10), (130, 120, 8), (760, 110, 12),
    (30, 220, 10), (840, 200, 8), (60, 380, 12), (820, 360, 9),
    (25, 500, 8),  (860, 490, 11),(70, 650, 10), (800, 640, 8),
    (35, 780, 12), (845, 760, 9), (55, 920, 8),  (815, 900, 10),
    (40,1080, 11), (830,1060, 8), (70,1180, 9),  (790,1190, 12),
    (200, 30, 9),  (650,  20, 8), (180, 80, 7),  (680,  70, 10),
    (300, 15, 11), (560,  25, 8),
]
for i, (dx, dy, r) in enumerate(dots):
    col = CONFETTI[i % len(CONFETTI)]
    draw.ellipse([(dx-r, dy-r), (dx+r, dy+r)], fill=col)

# ── ORANGE HEADER BLOCK ──────────────────────────────────────
draw.rounded_rectangle([(30, 28), (W-30, 148)], radius=22, fill=ORANGE)

cx(58, "Vehicle Safety AI  ·  ARIA", font(32, bold=True), BG)
cx(106, "Come interact with us!", font(22), (255, 230, 180))

# ── WIN HEADLINE ─────────────────────────────────────────────
cx(180, "Fill our quick survey and", font(28), DARK)
cx(220, "WIN a gift worth", font(32, bold=True), DARK)

# ── GIANT $50 ────────────────────────────────────────────────
big = font(140, bold=True)
prize_text = "$50!"
pb  = draw.textbbox((0,0), prize_text, font=big)
pw  = pb[2] - pb[0]
px  = (W - pw) // 2
# shadow
draw.text((px+4, 268), prize_text, font=big, fill=(200, 160, 0))
# main
draw.text((px,   264), prize_text, font=big, fill=PURPLE)

# ── STARS AROUND PRIZE ───────────────────────────────────────
star_positions = [(52, 310), (82, 260), (790, 300), (820, 258), (60, 380), (800, 370)]
star_cols      = [ORANGE, CORAL, ORANGE, YELLOW, CORAL, YELLOW]
for (sx, sy), sc in zip(star_positions, star_cols):
    r = 9
    draw.polygon([
        (sx, sy-r), (sx+3, sy-3), (sx+r, sy-3),
        (sx+4, sy+2), (sx+6, sy+r),
        (sx,   sy+5), (sx-6, sy+r),
        (sx-4, sy+2), (sx-r, sy-3), (sx-3, sy-3),
    ], fill=sc)

# ── 2 LUCKY WINNERS TAG ──────────────────────────────────────
wf = font(26, bold=True)
wt = "🎉  2 Lucky Winners  🎉"
wb = draw.textbbox((0,0), wt, font=wf)
ww = wb[2]-wb[0]+32
wx = (W-ww)//2
draw.rounded_rectangle([(wx, 428), (wx+ww, 472)], radius=20, fill=CORAL)
draw.text((wx+16, 432), wt, font=wf, fill=BG)

# ── WAVY DIVIDER (zig-zag dots) ───────────────────────────────
for i in range(18):
    dx2 = 60 + i * 43
    dy2 = 494 + (6 if i % 2 == 0 else -6)
    col = CONFETTI[i % len(CONFETTI)]
    draw.ellipse([(dx2-4, dy2-4), (dx2+4, dy2+4)], fill=col)

# ── SCAN TO WIN LABEL ────────────────────────────────────────
cx(512, "Scan to Enter the Draw", font(30, bold=True), DARK)
cx(554, "Takes less than 60 seconds!", font(21), MID)

# ── QR CARD ──────────────────────────────────────────────────
card_w, card_h = 370, 370
card_x = (W - card_w) // 2
card_y = 590
# thick colourful border (layered)
for layer, col in enumerate([ORANGE, BLUE, GREEN, CORAL]):
    off = 14 - layer * 4
    draw.rounded_rectangle(
        [(card_x-off, card_y-off), (card_x+card_w+off, card_y+card_h+off)],
        radius=24-layer*3, fill=col
    )
draw.rounded_rectangle(
    [(card_x, card_y), (card_x+card_w, card_y+card_h)],
    radius=16, fill=BG
)
qr_big = qr_img.resize((334, 334), Image.LANCZOS)
poster.paste(qr_big, (card_x+18, card_y+18))

# ── ARROW POINTING UP ────────────────────────────────────────
cx(978, "▲  Point your phone camera here  ▲", font(18, bold=True), BLUE)

# ── HORIZONTAL RULE ──────────────────────────────────────────
draw.rectangle([(50, 1010), (W-50, 1012)], fill=(220, 215, 240))

# ── FEATURES (colourful chips) ───────────────────────────────
cx(1022, "What We Built", font(23, bold=True), DARK)

chip_data = [
    ("🎯 Person & Vehicle Detection", ORANGE),
    ("📏 Depth & Distance Alerts",    PURPLE),
    ("🗣 Offline AI Voice (ARIA)",     GREEN),
    ("📐 Motion & Velocity Tracking",  BLUE),
]
cy2 = 1060
chip_font = font(18, bold=True)
row = []
for label, col in chip_data:
    row.append((label, col))
    if len(row) == 2:
        total = sum(draw.textbbox((0,0),t,font=chip_font)[2]+32 for t,_ in row) + 20
        rx = (W - total) // 2
        for t, c in row:
            tb = draw.textbbox((0,0), t, font=chip_font)
            tw2 = tb[2]-tb[0]+32
            draw.rounded_rectangle([(rx, cy2), (rx+tw2, cy2+34)], radius=17, fill=c)
            draw.text((rx+16, cy2+7), t, font=chip_font, fill=BG)
            rx += tw2 + 20
        cy2 += 46
        row = []

# ── FOOTER ───────────────────────────────────────────────────
draw.rounded_rectangle([(0, H-52), (W, H)], radius=0, fill=DARK)
cx(H-36, "Built on NVIDIA Jetson  ·  Fully Offline AI Pipeline", font(16), (180, 170, 220))

# ── SAVE ─────────────────────────────────────────────────────
poster.save(OUT_POSTER, dpi=(150, 150))
print(f"    Saved → {OUT_POSTER}")
print(f"\n  QR  : {os.path.abspath(OUT_QR)}")
print(f"  Poster: {os.path.abspath(OUT_POSTER)}")
print("\nPrint at A4 / 100% scale.")
