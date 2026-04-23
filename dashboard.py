#!/usr/bin/env python3
"""Vehicle Safety AI — Interactive Driver Dashboard (Fully Offline Pipeline)
   STT: Vosk  |  LLM: Ollama (llama3.2:1b)  |  TTS: Piper
"""

import os, sys, time, queue, random, io, wave, subprocess, threading, math, json
import numpy as np
import cv2  # cv2 overwrites QT_QPA_PLATFORM_PLUGIN_PATH on import

# Override AFTER cv2 import so the correct system Qt plugins are used
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = '/usr/lib/aarch64-linux-gnu/qt5/plugins'

from PyQt5.QtWidgets import *
from PyQt5.QtCore    import *
from PyQt5.QtGui     import *

import pyorbbecsdk as ob
from ultralytics import YOLO
from piper import PiperVoice

try:
    import requests
    REQUESTS_OK = True
except ImportError:
    REQUESTS_OK = False

try:
    import pyaudio
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False

try:
    from vosk import Model as VoskModel, KaldiRecognizer
    VOSK_LIB_OK = True
except ImportError:
    VOSK_LIB_OK = False

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
YOLO_ENGINE      = '/home/yahboom/models/yolo/yolov8n.engine'
PIPER_MODEL      = '/home/yahboom/models/piper/en_US-amy-medium.onnx'
VOSK_MODEL_PATH  = '/home/yahboom/models/vosk/vosk-model-en-us-0.22'
OLLAMA_MODEL     = 'llama3.2:1b'
OLLAMA_URL       = 'http://localhost:11434/api/generate'
MIC_DEVICE_INDEX = 0

PERSON_STOP_DIST    = 1.0
PERSON_WARNING_DIST = 5.0   # detect people up to 5 m away
CAR_DIST_THRESHOLD  = 8.0   # detect vehicles up to 8 m away
SPEAK_COOLDOWN      = 7.0
MOTION_KMH_THRESH   = 1.5   # min km/h to trigger "approaching/moving away" speech
MOTION_COOLDOWN     = 15.0  # min seconds between motion-change updates per track
ANY_SPEAK_COOLDOWN  = 8.0   # global: no speech within this many seconds of the last one
PERSON_CONF         = 0.55  # person-specific confidence (stricter than vehicles)
PERSON_MIN_H        = 90    # min bounding-box height in pixels (filters jackets/objects)
PERSON_MIN_RATIO    = 0.55  # min height/width ratio (people are taller than wide)
BEEP_COOLDOWN       = 2.0
BEEP_DANGER_DIST_P  = 1.2
BEEP_WARN_DIST_P    = 2.2
BEEP_DANGER_DIST_V  = 1.5
BEEP_WARN_DIST_V    = 3.0

# ─────────────────────────────────────────────────────────────
# STYLESHEET
# ─────────────────────────────────────────────────────────────
STYLE = """
* { font-family: 'Courier New', monospace; }
QMainWindow, QWidget#root { background: #070b14; }
QWidget#header, QWidget#footer { background: #0b1120; border: none; }
QWidget#sidebar { background: #0b1120; border-left: 1px solid #1a2e4a; }
QLabel#appTitle { color: #38bdf8; font-size: 17px; font-weight: bold; letter-spacing: 2px; }
QLabel#sysStatus { color: #475569; font-size: 10px; letter-spacing: 1px; }
QLabel#clockLabel { color: #64748b; font-size: 13px; }
QLabel#sectionHead { color: #334155; font-size: 9px; letter-spacing: 3px; padding: 2px 0px; }
QLabel#infoText { color: #64748b; font-size: 11px; padding: 2px 6px; }
QLabel#transcriptBox {
    color: #38bdf8; font-size: 11px; font-style: italic;
    background: #0a1628; border: 1px solid #1a3a5c; border-radius: 5px;
    padding: 4px 10px; min-width: 220px; max-width: 360px;
}
QLabel#ariaBox {
    color: #a78bfa; font-size: 11px;
    background: #0d0a1f; border: 1px solid #3b1f6a; border-radius: 5px;
    padding: 4px 10px; min-width: 220px; max-width: 360px;
}
QListWidget#alertLog {
    background: #070b14; color: #475569; font-size: 10px;
    border: 1px solid #1a2e4a; border-radius: 5px;
}
QListWidget#alertLog::item { padding: 2px 6px; }
QPushButton {
    background: #0f2240; color: #38bdf8; border: 1px solid #1e4a7a;
    border-radius: 7px; font-size: 12px; font-weight: bold;
    padding: 7px 16px; min-width: 120px;
}
QPushButton:hover  { background: #38bdf8; color: #070b14; }
QPushButton:pressed{ background: #0284c7; color: #070b14; }
QPushButton#micBtn { background: #052e16; color: #4ade80; border-color: #22c55e; }
QPushButton#micBtn:hover { background: #22c55e; color: #070b14; }
QPushButton#micBtn[state="listening"] {
    background: #0d3d2a; color: #6ee7b7; border-color: #10b981;
}
QPushButton#micBtn[state="processing"] {
    background: #713f12; color: #fde68a; border-color: #f59e0b;
}
QPushButton#muteBtn { background: #072818; color: #4ade80; border-color: #16a34a; min-width: 130px; }
QPushButton#muteBtn:hover { background: #16a34a; color: #070b14; }
QPushButton#muteBtn[muted="true"] { background: #450a0a; color: #fca5a5; border-color: #ef4444; }
QPushButton#muteBtn[muted="true"]:hover { background: #ef4444; color: white; }
QPushButton#exitBtn { background: #1a0808; color: #f87171; border-color: #b91c1c; min-width: 70px; }
QPushButton#exitBtn:hover { background: #ef4444; color: white; }
"""

# ─────────────────────────────────────────────────────────────
# TTS — Piper via aplay, with subprocess tracking for mute
# ─────────────────────────────────────────────────────────────
_speech_q     = queue.Queue()
_current_proc = None
_proc_lock    = threading.Lock()

def _dist_volume(dist, vehicle=False):
    if vehicle:
        if dist < 1.0: return 1.0
        elif dist < 2.0: return 0.85
        elif dist < 3.0: return 0.60
        else:            return 0.30
    else:
        if dist < 1.0: return 1.0
        elif dist < 2.0: return 0.50
        else:            return 0.10

def stop_speaking():
    global _current_proc
    while not _speech_q.empty():
        try: _speech_q.get_nowait()
        except: break
    with _proc_lock:
        if _current_proc and _current_proc.poll() is None:
            try: _current_proc.terminate()
            except: pass

def _tts_worker():
    global _current_proc
    voice = PiperVoice.load(PIPER_MODEL)
    rate  = voice.config.sample_rate
    while True:
        item = _speech_q.get()
        if item is None: break
        text, vol = item
        try:
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
                for c in voice.synthesize(text):
                    pcm = (c.audio_float_array * 32767 * vol).clip(-32768, 32767).astype(np.int16)
                    wf.writeframes(pcm.tobytes())
            proc = subprocess.Popen(
                ['aplay', '-q', '-f', 'cd', '-r', str(rate), '-c', '1', '-t', 'wav', '-'],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            with _proc_lock:
                _current_proc = proc
            try:
                proc.stdin.write(buf.getvalue())
                proc.stdin.close()
                proc.wait()
            except (BrokenPipeError, OSError):
                pass
        except Exception as e:
            print(f"[TTS] {e}")

threading.Thread(target=_tts_worker, daemon=True).start()

def speak(text, vol=1.0):
    while not _speech_q.empty():
        try: _speech_q.get_nowait()
        except: break
    _speech_q.put((text, vol))

# ─────────────────────────────────────────────────────────────
# BEEP — short tone generator, non-blocking daemon thread
# ─────────────────────────────────────────────────────────────
_beep_last_t = 0.0

def _do_beep_thread(danger: bool):
    rate    = 22050
    freq    = 1300 if danger else 880
    dur     = 0.10
    gap     = 0.07
    repeats = 3 if danger else 2
    vol     = 0.70
    out = io.BytesIO()
    with wave.open(out, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        for i in range(repeats):
            n    = int(rate * dur)
            t    = np.linspace(0, dur, n, False)
            tone = np.sin(2 * math.pi * freq * t)
            fade = max(1, int(rate * 0.012))
            tone[:fade]  *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
            wf.writeframes((tone * vol * 32767).astype(np.int16).tobytes())
            if i < repeats - 1:
                wf.writeframes(np.zeros(int(rate * gap), dtype=np.int16).tobytes())
    subprocess.run(['aplay', '-q', '-'], input=out.getvalue(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

def play_beep(danger=False):
    global _beep_last_t
    now = time.time()
    if now - _beep_last_t < BEEP_COOLDOWN: return
    _beep_last_t = now
    threading.Thread(target=_do_beep_thread, args=(danger,), daemon=True).start()

# ─────────────────────────────────────────────────────────────
# SRK LAUGH — "Ha Ha Ha Ha!" played on first person detection
# ─────────────────────────────────────────────────────────────
_laugh_last_t = 0.0
_LAUGH_COOLDOWN = 12.0   # don't laugh more than once every 12 s

def _srk_laugh_thread():
    """Generate a Shahrukh Khan-style 'Ha ha ha ha!' and play via aplay."""
    sr     = 22050
    chunks = []
    # Rising pitch across 4 "Ha"s — SRK's laugh gets more excited each beat
    pitches = [310, 370, 430, 400]
    durs    = [0.22, 0.20, 0.18, 0.28]   # last one lingers a bit
    gaps    = [0.07, 0.06, 0.06, 0.0]

    for pitch, dur, gap in zip(pitches, durs, gaps):
        t   = np.linspace(0, dur, int(dur * sr), endpoint=False)
        # Fundamental + 2nd harmonic for vocal warmth
        sig = (np.sin(2 * np.pi * pitch * t) * 0.55 +
               np.sin(2 * np.pi * pitch * 2 * t) * 0.28 +
               np.sin(2 * np.pi * pitch * 3 * t) * 0.10)
        # Sharp attack, smooth decay — mimics a laugh syllable
        env = np.clip(t * 80, 0, 1) * np.exp(-t * 9)
        chunks.append((sig * env * 0.80).astype(np.float32))
        if gap > 0:
            chunks.append(np.zeros(int(gap * sr), dtype=np.float32))

    audio = np.concatenate(chunks)
    out   = io.BytesIO()
    with wave.open(out, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    subprocess.run(['aplay', '-q', '-'], input=out.getvalue(),
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)

def play_srk_laugh():
    global _laugh_last_t
    now = time.time()
    if now - _laugh_last_t < _LAUGH_COOLDOWN: return
    _laugh_last_t = now
    threading.Thread(target=_srk_laugh_thread, daemon=True).start()

# ─────────────────────────────────────────────────────────────
# VOSK — offline speech-to-text (large model for best accuracy)
# ─────────────────────────────────────────────────────────────
_vosk_model   = None
_vosk_ready   = False
_vosk_loading = False

def _load_vosk_bg():
    global _vosk_model, _vosk_ready, _vosk_loading
    _vosk_loading = True
    if not VOSK_LIB_OK:
        print("[Vosk] Library not installed — run: pip install vosk")
        _vosk_loading = False
        return
    if not os.path.isdir(VOSK_MODEL_PATH):
        print(f"[Vosk] Model not found at {VOSK_MODEL_PATH}")
        _vosk_loading = False
        return
    try:
        import logging; logging.getLogger("vosk").setLevel(logging.ERROR)
        _vosk_model = VoskModel(VOSK_MODEL_PATH)
        _vosk_ready = True
        print("[Vosk] Large model loaded — offline STT ready.")
    except Exception as e:
        print(f"[Vosk] Failed to load: {e}")
    _vosk_loading = False

threading.Thread(target=_load_vosk_bg, daemon=True).start()

def _record_vosk(timeout=6.0, phrase_limit=10.0) -> str:
    """Record from mic, auto-resample to 16 kHz, return Vosk transcript."""
    if not _vosk_ready or not PYAUDIO_OK:
        return ""
    import audioop

    TARGET_RATE = 16000
    pa = pyaudio.PyAudio()

    # Find the first sample rate the mic actually accepts
    rec_rate = None
    for rate in [16000, 44100, 48000, 32000, 22050]:
        try:
            if pa.is_format_supported(rate, input_device=MIC_DEVICE_INDEX,
                                      input_channels=1,
                                      input_format=pyaudio.paInt16):
                rec_rate = rate
                break
        except Exception:
            continue

    if rec_rate is None:
        pa.terminate()
        print("[Vosk] No supported sample rate found")
        return ""

    rec    = KaldiRecognizer(_vosk_model, TARGET_RATE)
    stream = pa.open(
        format=pyaudio.paInt16, channels=1, rate=rec_rate,
        input=True, input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=4096,
    )
    stream.start_stream()
    start_t        = time.time()
    speech_start   = None
    text           = ""
    resample_state = None

    try:
        while True:
            data = stream.read(4096, exception_on_overflow=False)
            now  = time.time()

            if rec_rate != TARGET_RATE:
                data, resample_state = audioop.ratecv(
                    data, 2, 1, rec_rate, TARGET_RATE, resample_state)

            if rec.AcceptWaveform(data):
                res  = json.loads(rec.Result())
                text = res.get("text", "").strip()
                if text:
                    break
            else:
                part = json.loads(rec.PartialResult()).get("partial", "").strip()
                if part and speech_start is None:
                    speech_start = now

            if speech_start is None and now - start_t > timeout:
                break
            if speech_start and now - speech_start > phrase_limit:
                res  = json.loads(rec.FinalResult())
                text = res.get("text", "").strip()
                break
    except Exception as e:
        print(f"[Vosk record] {e}")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
    return text

# ─────────────────────────────────────────────────────────────
# OLLAMA — offline LLM (ARIA persona)
# ─────────────────────────────────────────────────────────────
ARIA_SYSTEM = """\
You are ARIA, the friendly AI co-pilot of a semi-autonomous vehicle safety system.
You are calm, warm, natural, and slightly witty — like a smart friend riding shotgun.

Your personality rules:
- Talk like a real person, never robotic or stiff
- Keep every reply SHORT: 1-2 spoken sentences only (it will be read aloud)
- Greetings: respond warmly and naturally (e.g. "Hey! How's the drive going?")
- If asked what you see: describe detections naturally using the sensor data provided
  e.g. "I can see a person about 2 metres ahead on the left, and a vehicle further back."
  If nothing detected: "All clear around you right now, looking good!"
- If asked how you are: respond positively and briefly
- Safety questions: give direct, calm, reassuring answers
- Don't start responses with "I" every time — vary your sentence starters
- Never say "As an AI" or "I am ARIA" — just be natural
- If you don't understand something, say so briefly and warmly
"""

# Shared detection context — updated every frame by CameraWorker
_det_context = "nothing detected around the vehicle"

def ask_ollama(user_text: str) -> str:
    if not REQUESTS_OK:
        return "Sorry, I can't think right now."
    # Always inject live sensor state so ARIA can answer "what do you see"
    prompt = (
        f"Live sensor data: {_det_context}\n"
        f"Driver says: {user_text}"
    )
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "prompt":  prompt,
                "system":  ARIA_SYSTEM,
                "stream":  False,
                "options": {
                    "num_predict": 90,
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=15,
        )
        r.raise_for_status()
        reply = r.json().get("response", "").strip()
        # Strip any stray asterisks/markdown that won't read well aloud
        reply = reply.replace("*", "").replace("#", "").strip()
        return reply
    except requests.exceptions.ConnectionError:
        return "Ollama isn't running — start it with ollama serve."
    except Exception:
        return "Something went wrong on my end, sorry."

# ─────────────────────────────────────────────────────────────
# Hard voice commands — these bypass the LLM for instant response
# ─────────────────────────────────────────────────────────────
HARD_CMDS = {
    "mute":      ["mute", "silence", "stop talking", "stop alerts", "be quiet"],
    "unmute":    ["unmute", "alerts on", "resume alerts", "turn alerts on"],
    "emergency": ["emergency", "mayday", "call for help", "sos"],
    "exit":      ["exit system", "shut down", "shutdown system", "close app"],
}

def classify_command(text: str):
    """Return hard command key if matched, else None (→ LLM handles it)."""
    t = text.lower()
    for cmd, keywords in HARD_CMDS.items():
        if any(k in t for k in keywords):
            return cmd
    return None

# ─────────────────────────────────────────────────────────────
# CAMERA / DETECTION HELPERS
# ─────────────────────────────────────────────────────────────
def decode_color(frame):
    raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
    h, w = frame.get_height(), frame.get_width()
    if len(raw) == h * w * 3:
        return raw.reshape(h, w, 3)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)

def scale_box(x1, y1, x2, y2, cw, ch, dw, dh):
    sx, sy = dw/cw, dh/ch
    return int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)

def get_direction(cx, width):
    if cx < width * 0.33: return "on the left"
    elif cx > width * 0.66: return "on the right"
    return "ahead"

def dist_color_cv(dist):
    if dist < 1.0:   return (0,  30, 220)
    elif dist < 2.0: return (0, 110, 255)
    elif dist < 3.0: return (0, 210, 255)
    else:            return (50, 210,  50)

def status_meta(event):
    return {
        "STOP_PERSON":    ("#dc2626", "⛔  STOP"),
        "STOP_CAR":       ("#dc2626", "⛔  STOP"),
        "CAR_PERSON":     ("#9333ea", "⚡  HIGH RISK"),
        "PERSON_WARNING": ("#d97706", "⚠   CAUTION"),
        "CAR_WARNING":    ("#ea580c", "⚠   CAUTION"),
    }.get(event, ("#16a34a", "✓   ALL CLEAR"))

def make_speech(event, p_word, d_per, d_car,
                p_vel=0.0, p_motion="", v_vel=0.0, v_motion=""):
    # Build natural motion suffix — both directions
    if   p_motion == "approaching"  and p_vel > 0.5: pa = f", approaching at {p_vel:.0f} km/h"
    elif p_motion == "moving away"  and p_vel > 0.5: pa = f", moving away at {p_vel:.0f} km/h"
    else:                                              pa = ""
    if   v_motion == "approaching"  and v_vel > 0.5: va = f", approaching at {v_vel:.0f} km/h"
    elif v_motion == "moving away"  and v_vel > 0.5: va = f", moving away at {v_vel:.0f} km/h"
    else:                                              va = ""
    P = {
        "STOP_PERSON":    [f"Stop stop stop! {p_word} is {d_per}{pa}, stop now!",
                           f"Whoa whoa! {p_word} extremely close {d_per}{pa}, stop!",
                           f"Watch out! {p_word} is {d_per}{pa}, don't move!"],
        "CAR_PERSON":     [f"Whoa! Car and {p_word} ahead{pa} — slow down now!",
                           f"Watch out! {p_word} and a vehicle ahead, high risk!"],
        "PERSON_WARNING": [f"Hey, there's {p_word} {d_per}{pa}, slow down.",
                           f"Heads up! {p_word} {d_per}{pa}, ease off.",
                           f"Watch it! {p_word} {d_per}{pa}, be careful."],
        "STOP_CAR":       [f"Stop! Vehicle is {d_car}{va}, don't move!",
                           f"Whoa whoa! Vehicle super close {d_car}{va}, stop now!"],
        "CAR_WARNING":    [f"Hey, there's a vehicle {d_car}{va}, watch it.",
                           f"Easy, vehicle coming up {d_car}{va}, slow down."],
    }
    return random.choice(P.get(event, ["Stay alert."]))


def _iou_nms(detections, iou_thresh=0.40):
    """Remove duplicate bounding boxes of the same class via IoU suppression.
    detections: list of (cx,cy,dist,x1,y1,x2,y2,...) — keeps the closer one."""
    if len(detections) <= 1:
        return detections
    dets = sorted(detections, key=lambda d: d[2])   # sort by distance (closest first)
    keep = []
    for det in dets:
        x1, y1, x2, y2 = det[3], det[4], det[5], det[6]
        dup = False
        for k in keep:
            kx1, ky1, kx2, ky2 = k[3], k[4], k[5], k[6]
            ix = max(0, min(x2, kx2) - max(x1, kx1))
            iy = max(0, min(y2, ky2) - max(y1, ky1))
            inter = ix * iy
            union = (x2-x1)*(y2-y1) + (kx2-kx1)*(ky2-ky1) - inter
            if union > 0 and inter / union > iou_thresh:
                dup = True; break
        if not dup:
            keep.append(det)
    return keep


# ─────────────────────────────────────────────────────────────
# OBJECT TRACKER — centroid-based, depth-history velocity
# ─────────────────────────────────────────────────────────────
class ObjectTracker:
    """
    Matches detections frame-to-frame by normalised centroid distance.
    Maintains a rolling depth history per track and estimates radial
    velocity (towards / away from camera) via linear regression.
    """
    MATCH_THRESH = 0.25   # normalised centroid distance threshold
    HISTORY      = 14     # max depth samples kept per track
    MIN_SAMPLES  = 5      # minimum samples before reporting velocity
    MAX_AGE      = 60     # frames without a match before dropping (survives ~2-4s YOLO misses)
    STILL_KMH    = 0.3    # below this → stationary

    def __init__(self):
        self._tracks: dict = {}
        self._nid          = 0
        # Display-number pool — reuses 1,2,3,... when tracks drop
        self._disp:      dict = {}   # {track_id: display_number}
        self._free_pool: list = []   # recycled display numbers (sorted smallest first)
        self._next_disp: int  = 1

    def _alloc_disp(self, tid: int):
        n = self._free_pool.pop(0) if self._free_pool else self._next_disp
        if n == self._next_disp:
            self._next_disp += 1
        self._disp[tid] = n

    def _release_disp(self, tid: int):
        n = self._disp.pop(tid, None)
        if n is not None:
            self._free_pool.append(n)
            self._free_pool.sort()

    def get_disp(self, tid: int) -> int:
        """Human-readable display number for a track (stable, reused after dropout)."""
        return self._disp.get(tid, 0)

    def update(self, detections, fw: int, fh: int, ts: float) -> dict:
        """
        detections : list of (cx, cy, dist_m)
        fw, fh     : frame width / height for normalisation
        ts         : current time in seconds
        Returns    : {det_index: (vel_kmh, motion_str)}
        """
        # Age every existing track
        for t in self._tracks.values():
            t["age"] += 1

        used_tracks: set = set()
        used_dets:   set = set()
        result:      dict = {}

        # Build all candidate (distance, det_idx, track_id) pairs
        candidates = []
        for di, (cx, cy, _) in enumerate(detections):
            for tid, t in self._tracks.items():
                if tid in used_tracks:
                    continue
                nd = math.sqrt(((cx - t["cx"]) / fw) ** 2 +
                               ((cy - t["cy"]) / fh) ** 2)
                if nd < self.MATCH_THRESH:
                    candidates.append((nd, di, tid))
        candidates.sort()

        # Greedy best-match assignment
        for _, di, tid in candidates:
            if di in used_dets or tid in used_tracks:
                continue
            used_dets.add(di); used_tracks.add(tid)
            cx, cy, dist = detections[di]
            t = self._tracks[tid]
            t["cx"] = cx; t["cy"] = cy
            t["dists"].append(dist); t["times"].append(ts)
            t["age"] = 0
            if len(t["dists"]) > self.HISTORY:
                t["dists"].pop(0); t["times"].pop(0)
            vel, motion = self._calc_vel(t)
            result[di] = (tid, vel, motion)

        # New tracks for unmatched detections
        for di, (cx, cy, dist) in enumerate(detections):
            if di not in used_dets:
                new_tid = self._nid
                self._tracks[new_tid] = {
                    "cx": cx, "cy": cy,
                    "dists": [dist], "times": [ts], "age": 0,
                }
                self._alloc_disp(new_tid)
                self._nid += 1
                result[di] = (new_tid, 0.0, "–")

        # Drop stale tracks — return their display numbers to the pool
        stale = [k for k, v in self._tracks.items() if v["age"] > self.MAX_AGE]
        for k in stale:
            self._release_disp(k)
            del self._tracks[k]
        return result

    def active_ids(self) -> set:
        """Return all track IDs that are still alive (not yet dropped)."""
        return set(self._tracks.keys())

    def _calc_vel(self, t) -> tuple:
        dists = t["dists"]; times = t["times"]
        if len(dists) < self.MIN_SAMPLES:
            return 0.0, "–"
        ta = np.array(times, dtype=np.float64) - times[0]
        da = np.array(dists, dtype=np.float64)
        # Rolling-median smooth (window=5) to reduce depth sensor noise
        # This is especially important for "moving away" where signal is weaker
        smoothed = np.array([float(np.median(da[max(0, k-2):k+3]))
                             for k in range(len(da))])
        slope = float(np.polyfit(ta, smoothed, 1)[0])   # m/s (neg=approaching, pos=away)
        kmh   = slope * 3.6
        if abs(kmh) < self.STILL_KMH:
            return 0.0, "stationary"
        return abs(kmh), ("approaching" if kmh < 0 else "moving away")

# ─────────────────────────────────────────────────────────────
# PULSE ORB — animated mic listening indicator
# ─────────────────────────────────────────────────────────────
class PulseOrb(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(48, 48)
        self._phase = 0.0
        self._state = "idle"
        t = QTimer(self); t.timeout.connect(self._tick); t.start(33)

    def set_state(self, state: str):
        self._state = state

    def _tick(self):
        self._phase = (self._phase + 0.10) % (2 * math.pi)
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        cx, cy = self.width() // 2, self.height() // 2
        pulse  = (math.sin(self._phase) + 1) / 2

        if self._state == "idle":
            core  = QColor(18, 36, 60);  glow = QColor(0,0,0,0); r, gr = 7, 10
        elif self._state == "listening":
            lv    = int(140 + pulse * 115)
            core  = QColor(0, lv, lv); glow = QColor(0, 230, 220, int(25 + pulse * 90))
            r     = int(9 + pulse * 6); gr = r + 12
        elif self._state == "processing":
            lv    = int(160 + pulse * 95)
            core  = QColor(lv, int(lv * 0.55), 0); glow = QColor(255, 170, 0, int(25 + pulse * 80))
            r     = int(8 + pulse * 5); gr = r + 10
        else:  # error
            lv    = int(140 + pulse * 80)
            core  = QColor(lv, 20, 20); glow = QColor(255, 40, 40, int(40 + pulse * 70))
            r, gr = 8, 16

        grad = QRadialGradient(float(cx), float(cy), float(gr))
        grad.setColorAt(0.0, glow); grad.setColorAt(1.0, QColor(0,0,0,0))
        painter.setPen(Qt.NoPen); painter.setBrush(QBrush(grad))
        painter.drawEllipse(cx-gr, cy-gr, 2*gr, 2*gr)
        painter.setBrush(QBrush(core)); painter.setPen(Qt.NoPen)
        painter.drawEllipse(cx-r, cy-r, 2*r, 2*r)


# ─────────────────────────────────────────────────────────────
# STATUS CARD — animated breathing + flash on event change
# ─────────────────────────────────────────────────────────────
class StatusCard(QWidget):
    def __init__(self):
        super().__init__()
        self.event  = None
        self._phase = 0.0
        self._flash = 1.0
        self.setMinimumHeight(82)
        t = QTimer(self); t.timeout.connect(self._tick); t.start(40)

    def set_event(self, event):
        if event != self.event: self._flash = 1.6
        self.event = event

    def _tick(self):
        self._phase = (self._phase + 0.055) % (2 * math.pi)
        if self._flash > 1.0: self._flash = max(1.0, self._flash - 0.04)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        hex_col, text = status_meta(self.event)
        col   = QColor(hex_col)
        pulse = (math.sin(self._phase) + 1) / 2
        if   self.event in ("STOP_PERSON","STOP_CAR","CAR_PERSON"): breath = 0.22
        elif self.event:                                             breath = 0.12
        else:                                                        breath = 0.05
        intensity = self._flash * (1.0 + pulse * breath)
        dark1 = int(max(100, 170 / intensity))
        dark2 = int(max(140, 230 / intensity))
        grad  = QLinearGradient(0, 0, w, h)
        grad.setColorAt(0, col.darker(dark1)); grad.setColorAt(1, col.darker(dark2))
        path  = QPainterPath(); path.addRoundedRect(0, 0, w, h, 10, 10)
        p.fillPath(path, QBrush(grad))
        p.setPen(QPen(col, 1.5 + pulse * 1.5 if self.event else 1.0)); p.drawPath(path)
        p.setPen(Qt.white); p.setFont(QFont("Courier New", 19, QFont.Bold))
        p.drawText(QRect(0, 0, w, h), Qt.AlignCenter, text)


# ─────────────────────────────────────────────────────────────
# DISTANCE METER
# ─────────────────────────────────────────────────────────────
class DistanceMeter(QWidget):
    MAX_DIST = 5.0
    def __init__(self, label):
        super().__init__(); self.label = label; self.dist = None; self.setFixedSize(138, 138)
    def set_dist(self, dist): self.dist = dist; self.update()
    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height(); cx, cy = w//2, h//2; r = min(w,h)//2-8
        p.setPen(Qt.NoPen); p.setBrush(QColor(10,18,34)); p.drawEllipse(cx-r,cy-r,2*r,2*r)
        pen = QPen(QColor(25,42,70),8); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen); p.setBrush(Qt.NoBrush)
        rect = QRectF(cx-r+5,cy-r+5,2*(r-5),2*(r-5)); p.drawArc(rect,int(225*16),int(-270*16))
        if self.dist is not None:
            ratio = max(0.0, min(1.0, 1.0 - self.dist/self.MAX_DIST)); span = ratio*270
            if   self.dist < 1.0: col = QColor(220,30,30)
            elif self.dist < 2.0: col = QColor(255,130,0)
            elif self.dist < 3.0: col = QColor(240,200,0)
            else:                 col = QColor(0,200,100)
            glow = QRadialGradient(cx,cy,r)
            glow.setColorAt(0,QColor(col.red(),col.green(),col.blue(),18)); glow.setColorAt(1,QColor(0,0,0,0))
            p.setBrush(QBrush(glow)); p.setPen(Qt.NoPen); p.drawEllipse(cx-r,cy-r,2*r,2*r)
            pen = QPen(col,8); pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen); p.setBrush(Qt.NoBrush); p.drawArc(rect,int(225*16),int(-span*16))
            p.setPen(QPen(col)); p.setFont(QFont("Courier New",15,QFont.Bold))
            p.drawText(QRectF(cx-44,cy-16,88,28),Qt.AlignCenter,f"{self.dist:.1f}m")
        p.setPen(QPen(QColor(60,80,110))); p.setFont(QFont("Courier New",8))
        p.drawText(QRectF(cx-44,cy+14,88,16),Qt.AlignCenter,self.label)


# ─────────────────────────────────────────────────────────────
# CAMERA WORKER
# ─────────────────────────────────────────────────────────────
class DetData:
    bgr=None; depth=None; closest_person=None; closest_car=None
    person_count=0; event=None; speech=None; volume=1.0
    # closest_person / closest_car dicts now carry:
    #   {"dist", "direction", "vel_kmh", "motion"}

class CameraWorker(QThread):
    ready  = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run        = True
        self._muted      = False
        self._p_tracker  = ObjectTracker()
        self._v_tracker  = ObjectTracker()
        # Per-track speech state  (None / "new" / "approaching" / "moving away" / "danger")
        self._p_announced: dict = {}
        self._v_announced: dict = {}
        # Per-track last-spoken timestamp (for motion-change cooldown)
        self._p_spoken_t: dict = {}
        self._v_spoken_t: dict = {}
        # Global gate — no speech at all within this many seconds of the last one
        self._last_any_speak = 0.0

    def set_muted(self, v): self._muted = v

    def run(self):
        global _det_context
        self.status.emit("Loading YOLO…")
        yolo = YOLO(YOLO_ENGINE, task='detect')
        self.status.emit("Connecting camera…")
        try:
            pipeline = ob.Pipeline()
            cfg = ob.Config()
            cfg.enable_stream(ob.OBStreamType.COLOR_STREAM)
            cfg.enable_stream(ob.OBStreamType.DEPTH_STREAM)
            pipeline.start(cfg)
        except Exception as e:
            self.status.emit(f"⚠ Camera failed — replug USB  ({e})")
            return
        self.status.emit("● SYSTEM ACTIVE")

        while self._run:
            try:
                frames = pipeline.wait_for_frames(1000)
            except Exception as e:
                self.status.emit(f"Camera error — {e}"); time.sleep(0.1); continue
            if not frames: continue
            cf = frames.get_color_frame(); df = frames.get_depth_frame()
            if not cf or not df: continue
            bgr = decode_color(cf)
            if bgr is None: continue
            ch, cw = bgr.shape[:2]
            dh, dw = df.get_height(), df.get_width()
            depth  = np.frombuffer(df.get_data(), dtype=np.uint16).reshape(dh, dw)
            try:
                results = yolo(bgr, verbose=False, conf=0.40, iou=0.45)[0]
            except Exception: continue

            # ── Pass 1: measure depth for every detection ─────────
            p_raw = []   # person detections: (cx, cy, dist, x1,y1,x2,y2, direction)
            v_raw = []   # vehicle detections: same
            VEHICLES = {"car","truck","bus","motorcycle","bicycle"}

            for box in results.boxes:
                label  = results.names[int(box.cls)]
                conf   = float(box.conf)
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cx_b, cy_b  = (x1+x2)//2, (y1+y2)//2
                box_h = y2 - y1
                box_w = x2 - x1

                # ── Person-specific filters: stricter conf + size/shape gate
                if label == "person":
                    if conf < PERSON_CONF: continue              # stricter threshold
                    if box_h < PERSON_MIN_H: continue            # too short (jacket/object)
                    if box_w > 0 and box_h / box_w < PERSON_MIN_RATIO: continue  # too wide

                dx1,dy1,dx2,dy2 = scale_box(x1,y1,x2,y2,cw,ch,dw,dh)
                dx1,dy1 = max(0,dx1), max(0,dy1); dx2,dy2 = min(dw,dx2), min(dh,dy2)
                my,mx   = (dy2-dy1)//5, (dx2-dx1)//5
                roi   = depth[dy1+my:dy2-my, dx1+mx:dx2-mx]
                valid = roi[roi > 0]
                if len(valid) < 5: continue
                dist      = float(np.percentile(valid, 25)) / 1000
                direction = get_direction(cx_b, cw)
                if label == "person" and dist < PERSON_WARNING_DIST:
                    p_raw.append((cx_b, cy_b, dist, x1, y1, x2, y2, direction, label))
                elif label in VEHICLES and dist < CAR_DIST_THRESHOLD:
                    v_raw.append((cx_b, cy_b, dist, x1, y1, x2, y2, direction, label))

            # Remove duplicate boxes (same person detected twice by YOLO)
            p_raw = _iou_nms(p_raw)
            v_raw = _iou_nms(v_raw)

            # ── Pass 2: track & compute velocities ────────────
            ts     = time.time()
            p_vels = self._p_tracker.update(
                [(r[0],r[1],r[2]) for r in p_raw], cw, ch, ts)
            v_vels = self._v_tracker.update(
                [(r[0],r[1],r[2]) for r in v_raw], cw, ch, ts)
            # Prune dicts for tracks that have been fully dropped by the tracker
            act_p = self._p_tracker.active_ids()
            act_v = self._v_tracker.active_ids()
            self._p_announced = {k: v for k, v in self._p_announced.items() if k in act_p}
            self._v_announced = {k: v for k, v in self._v_announced.items() if k in act_v}
            self._p_spoken_t  = {k: v for k, v in self._p_spoken_t.items()  if k in act_p}
            self._v_spoken_t  = {k: v for k, v in self._v_spoken_t.items()  if k in act_v}

            # ── Pass 3: draw overlays & populate DetData ──────
            d = DetData(); d.bgr = bgr; d.depth = depth

            def _draw_box(x1, y1, x2, y2, label, dist, vel, motion, col):
                has_vel = motion in ("approaching","moving away") and vel > 0.4
                bh = 46 if has_vel else 24
                cv2.rectangle(bgr, (x1,y1), (x2,y2), col, 2)
                cv2.rectangle(bgr, (x1, y1-bh), (x1+170, y1), col, -1)
                cv2.putText(bgr, f"{label}  {dist:.1f}m",
                            (x1+4, y1-bh+16), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1)
                if has_vel:
                    arrow = ">> " if motion == "approaching" else "<< "
                    cv2.putText(bgr, f"{arrow}{vel:.1f} km/h",
                                (x1+4, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,0,0), 1)

            for i, (cx_b, cy_b, dist, x1, y1, x2, y2, direction, label) in enumerate(p_raw):
                tid, vel, motion = p_vels.get(i, (None, 0.0, "–"))
                dn  = self._p_tracker.get_disp(tid) if tid is not None else 0
                tag = f"person-{dn}" if dn else "person"
                col = dist_color_cv(dist)
                _draw_box(x1, y1, x2, y2, tag, dist, vel, motion, col)
                d.person_count += 1
                if d.closest_person is None or dist < d.closest_person["dist"]:
                    d.closest_person = {"dist": dist, "direction": direction,
                                        "vel_kmh": vel, "motion": motion,
                                        "tid": tid}

            for i, (cx_b, cy_b, dist, x1, y1, x2, y2, direction, label) in enumerate(v_raw):
                tid, vel, motion = v_vels.get(i, (None, 0.0, "–"))
                dn    = self._v_tracker.get_disp(tid) if tid is not None else 0
                v_tag = f"{label}-{dn}" if dn else label
                col = dist_color_cv(dist)
                _draw_box(x1, y1, x2, y2, v_tag, dist, vel, motion, col)
                if d.closest_car is None or dist < d.closest_car["dist"]:
                    d.closest_car = {"dist": dist, "direction": direction,
                                     "vel_kmh": vel, "motion": motion,
                                     "tid": tid}

            # ── Event classification ───────────────────────────
            p_word = "person" if d.person_count == 1 else f"{d.person_count} people"
            d_per  = d.closest_person["direction"] if d.closest_person else ""
            d_car  = d.closest_car["direction"]    if d.closest_car    else ""
            p_vel  = d.closest_person["vel_kmh"]   if d.closest_person else 0.0
            p_mot  = d.closest_person["motion"]    if d.closest_person else ""
            v_vel  = d.closest_car["vel_kmh"]      if d.closest_car    else 0.0
            v_mot  = d.closest_car["motion"]       if d.closest_car    else ""

            # ── Update ARIA detection context ──────────────────
            ctx_parts = []
            if d.closest_person:
                mot_str = (f", {p_mot} at {p_vel:.1f} km/h"
                           if p_mot not in ("–","stationary") and p_vel > 0.4 else
                           (", stationary" if p_mot == "stationary" else ""))
                ctx_parts.append(
                    f"{d.person_count} person{'s' if d.person_count>1 else ''} "
                    f"{d_per} at {d.closest_person['dist']:.1f}m{mot_str}")
            if d.closest_car:
                mot_str = (f", {v_mot} at {v_vel:.1f} km/h"
                           if v_mot not in ("–","stationary") and v_vel > 0.4 else
                           (", stationary" if v_mot == "stationary" else ""))
                ctx_parts.append(
                    f"vehicle {d_car} at {d.closest_car['dist']:.1f}m{mot_str}")
            _det_context = ", ".join(ctx_parts) if ctx_parts else "nothing detected"

            if d.closest_person:
                d.volume = _dist_volume(d.closest_person["dist"])
                if d.closest_person["dist"] < PERSON_STOP_DIST: d.event = "STOP_PERSON"
                elif d.closest_car:                              d.event = "CAR_PERSON"
                else:                                            d.event = "PERSON_WARNING"
            elif d.closest_car:
                d.volume = _dist_volume(d.closest_car["dist"], vehicle=True)
                d.event  = "STOP_CAR" if d.closest_car["dist"] < PERSON_STOP_DIST else "CAR_WARNING"

            # ── Per-track speech — one-shot detect + motion-change only ──
            # States stored in _p/_v_announced[tid]:
            #   None → "new" → "approaching" / "moving away" / "stationary" / "danger"
            if not self._muted:
                to_say = None
                to_vol = 0.8

                now = time.time()

                # ---- Persons ----
                for i, (_, _, dist, _, _, _, _, direction, _) in enumerate(p_raw):
                    tid, vel, motion = p_vels.get(i, (None, 0.0, "–"))
                    if tid is None:
                        continue
                    prev      = self._p_announced.get(tid)
                    last_t    = self._p_spoken_t.get(tid, 0.0)
                    on_cooldown = (now - last_t) < MOTION_COOLDOWN
                    fast_enough = vel >= MOTION_KMH_THRESH

                    if prev is None:
                        # First sighting — announce exactly once
                        to_say = random.choice([
                            f"Heads up! Person {direction}, {dist:.1f} metres.",
                            f"Hey, person detected {direction}, {dist:.1f} metres away.",
                        ])
                        to_vol = _dist_volume(dist)
                        self._p_announced[tid] = "new"
                        self._p_spoken_t[tid]  = now
                        break

                    if dist < PERSON_STOP_DIST and prev != "danger":
                        # Danger overrides cooldown
                        to_say = random.choice([
                            "Stop! Person is right ahead, don't move!",
                            "Whoa! Stop now, person extremely close!",
                        ])
                        to_vol = 1.0
                        self._p_announced[tid] = "danger"
                        self._p_spoken_t[tid]  = now
                        break

                    if (motion == "approaching"
                            and prev not in ("approaching", "danger")
                            and fast_enough and not on_cooldown):
                        to_say = f"Watch out, person is approaching at {vel:.0f} km/h."
                        to_vol = _dist_volume(dist)
                        self._p_announced[tid] = "approaching"
                        self._p_spoken_t[tid]  = now
                        break

                    if (motion == "moving away"
                            and prev in ("approaching", "danger")
                            and fast_enough and not on_cooldown):
                        to_say = "Person is moving away now."
                        to_vol = 0.65
                        self._p_announced[tid] = "moving away"
                        self._p_spoken_t[tid]  = now
                        break

                # ---- Vehicles (only if no person speech queued) ----
                if to_say is None:
                    for i, (_, _, dist, _, _, _, _, direction, label) in enumerate(v_raw):
                        tid, vel, motion = v_vels.get(i, (None, 0.0, "–"))
                        if tid is None:
                            continue
                        prev      = self._v_announced.get(tid)
                        last_t    = self._v_spoken_t.get(tid, 0.0)
                        on_cooldown = (now - last_t) < MOTION_COOLDOWN
                        fast_enough = vel >= MOTION_KMH_THRESH

                        if prev is None:
                            to_say = random.choice([
                                f"Heads up! {label.capitalize()} {direction}, {dist:.1f} metres.",
                                f"Hey, {label} detected {direction}, {dist:.1f} metres away.",
                            ])
                            to_vol = _dist_volume(dist, vehicle=True)
                            self._v_announced[tid] = "new"
                            self._v_spoken_t[tid]  = now
                            break

                        if dist < PERSON_STOP_DIST and prev != "danger":
                            to_say = random.choice([
                                f"Stop! {label.capitalize()} is right there, don't move!",
                                f"Whoa! Stop now, {label} extremely close!",
                            ])
                            to_vol = 1.0
                            self._v_announced[tid] = "danger"
                            self._v_spoken_t[tid]  = now
                            break

                        if (motion == "approaching"
                                and prev not in ("approaching", "danger")
                                and fast_enough and not on_cooldown):
                            to_say = f"Watch out, {label} approaching at {vel:.0f} km/h."
                            to_vol = _dist_volume(dist, vehicle=True)
                            self._v_announced[tid] = "approaching"
                            self._v_spoken_t[tid]  = now
                            break

                        if (motion == "moving away"
                                and prev in ("approaching", "danger")
                                and fast_enough and not on_cooldown):
                            to_say = f"{label.capitalize()} is moving away now."
                            to_vol = 0.65
                            self._v_announced[tid] = "moving away"
                            self._v_spoken_t[tid]  = now
                            break

                # Global gate: suppress everything except danger if too soon
                is_danger = to_say and ("Stop!" in to_say or "Whoa!" in to_say)
                global_ok = (now - self._last_any_speak) >= ANY_SPEAK_COOLDOWN
                if to_say and (is_danger or global_ok):
                    speak(to_say, to_vol)
                    d.speech = to_say
                    self._last_any_speak = now

            if not self._muted:
                p_dist = d.closest_person["dist"] if d.closest_person else None
                v_dist = d.closest_car["dist"]    if d.closest_car    else None
                if   (p_dist and p_dist < BEEP_DANGER_DIST_P) or (v_dist and v_dist < BEEP_DANGER_DIST_V):
                    play_beep(danger=True)
                elif (p_dist and p_dist < BEEP_WARN_DIST_P)   or (v_dist and v_dist < BEEP_WARN_DIST_V):
                    play_beep(danger=False)

            self.ready.emit(d)

        try: pipeline.stop()
        except: pass

    def stop(self): self._run = False


# ─────────────────────────────────────────────────────────────
# MIC WORKER — Vosk STT → hard-command OR Ollama LLM → Piper TTS
# ─────────────────────────────────────────────────────────────
class MicWorker(QThread):
    heard   = pyqtSignal(str)   # raw transcribed text (for transcript display)
    replied = pyqtSignal(str)   # ARIA's LLM reply text (for ARIA display box)
    command = pyqtSignal(str)   # hard command key: "mute" / "unmute" / "emergency" / "exit"
    state   = pyqtSignal(str)   # "listening" | "processing" | "idle" | "error"

    def run(self):
        if not VOSK_LIB_OK or not PYAUDIO_OK:
            self.state.emit("error")
            self.heard.emit("[Vosk or PyAudio not installed]")
            return
        if _vosk_loading:
            self.state.emit("error")
            self.heard.emit("[STT model still loading — try again in a moment]")
            return
        if not _vosk_ready:
            self.state.emit("error")
            self.heard.emit("[STT model not found — see console]")
            return

        # ── 1. Record + transcribe with Vosk ──────────────────
        self.state.emit("listening")
        text = _record_vosk(timeout=6.0, phrase_limit=10.0)

        if not text:
            self.state.emit("idle")
            return

        self.heard.emit(text)

        # ── 2. Route: instant hard command OR ARIA (Ollama) ───
        self.state.emit("processing")
        cmd = classify_command(text)
        if cmd:
            self.command.emit(cmd)
        else:
            reply = ask_ollama(text)
            if reply:
                self.replied.emit(reply)
                speak(reply, 0.9)

        self.state.emit("idle")


# ─────────────────────────────────────────────────────────────
# MAIN DASHBOARD WINDOW
# ─────────────────────────────────────────────────────────────
class Dashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Safety AI — ARIA")
        self.resize(1280, 720)
        self.setStyleSheet(STYLE)
        self._muted  = False
        self._worker = CameraWorker()
        self._mic    = MicWorker()
        self._build_ui()
        self._connect_signals()
        t = QTimer(self); t.timeout.connect(self._tick_clock); t.start(1000)
        self._worker.start()

    def _build_ui(self):
        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root); vbox.setSpacing(0); vbox.setContentsMargins(0,0,0,0)
        vbox.addWidget(self._make_header())
        body = QWidget(); bh = QHBoxLayout(body)
        bh.setSpacing(0); bh.setContentsMargins(0,0,0,0)
        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setMinimumSize(880, 550)
        self.cam_label.setStyleSheet("background:#040810; color:#1a3a5c; font-size:16px;")
        self.cam_label.setText("Initialising camera…")
        bh.addWidget(self.cam_label, stretch=1)
        bh.addWidget(self._make_sidebar())
        vbox.addWidget(body, stretch=1)
        vbox.addWidget(self._make_footer())

    def _make_header(self):
        h = QWidget(); h.setObjectName("header"); h.setFixedHeight(52)
        lay = QHBoxLayout(h); lay.setContentsMargins(16,0,16,0)
        dot = QLabel("◉"); dot.setStyleSheet("color:#22c55e; font-size:14px;")
        lay.addWidget(dot)
        title = QLabel("VEHICLE SAFETY AI  ·  ARIA"); title.setObjectName("appTitle")
        lay.addWidget(title); lay.addSpacing(16)
        self.sys_status = QLabel("Initialising…"); self.sys_status.setObjectName("sysStatus")
        lay.addWidget(self.sys_status); lay.addStretch()
        self.clock_lbl = QLabel(); self.clock_lbl.setObjectName("clockLabel")
        self._tick_clock(); lay.addWidget(self.clock_lbl)
        return h

    def _make_sidebar(self):
        sb = QWidget(); sb.setObjectName("sidebar"); sb.setFixedWidth(340)
        lay = QVBoxLayout(sb); lay.setContentsMargins(10,10,10,10); lay.setSpacing(8)
        self.status_card = StatusCard(); lay.addWidget(self.status_card)
        lay.addWidget(self._sec("DETECTION"))
        meters = QWidget(); ml = QHBoxLayout(meters)
        ml.setContentsMargins(0,0,0,0); ml.setSpacing(6)
        self.person_meter  = DistanceMeter("PERSON")
        self.vehicle_meter = DistanceMeter("VEHICLE")
        ml.addWidget(self.person_meter); ml.addWidget(self.vehicle_meter)
        lay.addWidget(meters)
        self.info_lbl = QLabel("No detections"); self.info_lbl.setObjectName("infoText")
        self.info_lbl.setAlignment(Qt.AlignCenter); lay.addWidget(self.info_lbl)
        lay.addWidget(self._sec("DEPTH VIEW"))
        self.depth_lbl = QLabel(); self.depth_lbl.setAlignment(Qt.AlignCenter)
        self.depth_lbl.setStyleSheet("background:#040810; border:1px solid #1a2e4a; border-radius:6px;")
        self.depth_lbl.setMinimumHeight(110); lay.addWidget(self.depth_lbl, stretch=1)
        lay.addStretch(); return sb

    def _make_footer(self):
        f = QWidget(); f.setObjectName("footer"); f.setFixedHeight(90)
        lay = QHBoxLayout(f); lay.setContentsMargins(14,8,14,8); lay.setSpacing(10)

        self.mic_orb = PulseOrb(); lay.addWidget(self.mic_orb)

        self.mic_btn = QPushButton("🎤  SPEAK"); self.mic_btn.setObjectName("micBtn")
        self.mic_btn.setProperty("state","idle"); self.mic_btn.clicked.connect(self._on_mic)
        lay.addWidget(self.mic_btn)

        self.mute_btn = QPushButton("🔊  ALERTS ON"); self.mute_btn.setObjectName("muteBtn")
        self.mute_btn.setProperty("muted","false"); self.mute_btn.clicked.connect(self._toggle_mute)
        lay.addWidget(self.mute_btn)

        # Transcript + ARIA reply stacked vertically
        tbox = QWidget(); tv = QVBoxLayout(tbox); tv.setContentsMargins(0,0,0,0); tv.setSpacing(3)
        self.transcript = QLabel("Say something to ARIA…"); self.transcript.setObjectName("transcriptBox")
        self.aria_reply = QLabel(""); self.aria_reply.setObjectName("ariaBox")
        self.aria_reply.setVisible(False)
        tv.addWidget(self.transcript); tv.addWidget(self.aria_reply)
        lay.addWidget(tbox)

        self.alert_log = QListWidget(); self.alert_log.setObjectName("alertLog")
        self.alert_log.setMaximumHeight(74); self.alert_log.setMinimumWidth(220)
        lay.addWidget(self.alert_log, stretch=1)

        exit_btn = QPushButton("✕ EXIT"); exit_btn.setObjectName("exitBtn")
        exit_btn.clicked.connect(self.close); lay.addWidget(exit_btn)
        return f

    def _sec(self, text):
        lbl = QLabel(text); lbl.setObjectName("sectionHead"); lbl.setAlignment(Qt.AlignLeft)
        return lbl

    def _connect_signals(self):
        self._worker.ready.connect(self._on_frame)
        self._worker.status.connect(self.sys_status.setText)
        self._mic.state.connect(self._on_mic_state)
        self._mic.heard.connect(self._on_mic_heard)
        self._mic.replied.connect(self._on_mic_replied)
        self._mic.command.connect(self._on_mic_command)

    # ── SLOTS ─────────────────────────────────────────────────
    def _on_frame(self, d: DetData):
        if d.bgr is not None:
            rgb = cv2.cvtColor(d.bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (880, 550))
            h, w, ch = rgb.shape
            self.cam_label.setPixmap(QPixmap.fromImage(
                QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)))
        self.status_card.set_event(d.event)
        self.person_meter.set_dist(d.closest_person["dist"] if d.closest_person else None)
        self.vehicle_meter.set_dist(d.closest_car["dist"]   if d.closest_car    else None)
        parts = []
        if d.closest_person:
            p = d.closest_person
            if p["motion"] == "approaching" and p["vel_kmh"] > 0.4:
                v_str = f"  ↓ {p['vel_kmh']:.1f} km/h"
            elif p["motion"] == "moving away" and p["vel_kmh"] > 0.4:
                v_str = f"  ↑ {p['vel_kmh']:.1f} km/h"
            elif p["motion"] == "stationary":
                v_str = "  ● still"
            else:
                v_str = ""
            parts.append(f"{d.person_count} person{'s' if d.person_count>1 else ''}  {p['direction']}{v_str}")
        if d.closest_car:
            v = d.closest_car
            if v["motion"] == "approaching" and v["vel_kmh"] > 0.4:
                v_str = f"  ↓ {v['vel_kmh']:.1f} km/h"
            elif v["motion"] == "moving away" and v["vel_kmh"] > 0.4:
                v_str = f"  ↑ {v['vel_kmh']:.1f} km/h"
            elif v["motion"] == "stationary":
                v_str = "  ● still"
            else:
                v_str = ""
            parts.append(f"vehicle  {v['direction']}{v_str}")
        self.info_lbl.setText("  |  ".join(parts) if parts else "No detections")
        if d.depth is not None:
            vis   = cv2.normalize(d.depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
            color = cv2.cvtColor(cv2.resize(color,(318,108)), cv2.COLOR_BGR2RGB)
            self.depth_lbl.setPixmap(QPixmap.fromImage(
                QImage(color.data, 318, 108, 318*3, QImage.Format_RGB888)))
        if d.speech:
            hex_col, _ = status_meta(d.event)
            item = QListWidgetItem(f"[{time.strftime('%H:%M:%S')}]  {d.speech}")
            item.setForeground(QColor(hex_col))
            self.alert_log.insertItem(0, item)
            if self.alert_log.count() > 50:
                self.alert_log.takeItem(self.alert_log.count()-1)

    def _tick_clock(self):
        self.clock_lbl.setText(time.strftime("  %H:%M:%S  %Y-%m-%d"))

    def _toggle_mute(self):
        self._muted = not self._muted
        self._worker.set_muted(self._muted)
        if self._muted:
            stop_speaking()
            self.mute_btn.setText("🔇  ALERTS OFF")
        else:
            self.mute_btn.setText("🔊  ALERTS ON")
        self.mute_btn.setProperty("muted", "true" if self._muted else "false")
        self.mute_btn.style().unpolish(self.mute_btn)
        self.mute_btn.style().polish(self.mute_btn)

    def _on_mic(self):
        if not self._mic.isRunning():
            self._mic = MicWorker()
            self._mic.state.connect(self._on_mic_state)
            self._mic.heard.connect(self._on_mic_heard)
            self._mic.replied.connect(self._on_mic_replied)
            self._mic.command.connect(self._on_mic_command)
            self._mic.start()

    def _on_mic_state(self, state):
        labels = {"listening":"🔴  LISTENING…","processing":"⏳  ARIA THINKING…",
                  "idle":"🎤  SPEAK","error":"🚫  MIC ERROR"}
        self.mic_btn.setText(labels.get(state, "🎤  SPEAK"))
        self.mic_btn.setProperty("state", state)
        self.mic_btn.style().unpolish(self.mic_btn)
        self.mic_btn.style().polish(self.mic_btn)
        self.mic_orb.set_state(state)
        if state == "idle":
            # Hide ARIA reply after 6 seconds
            QTimer.singleShot(6000, lambda: self.aria_reply.setVisible(False))

    def _on_mic_heard(self, text):
        if text:
            self.transcript.setText(f'You: "{text}"')
            self.aria_reply.setVisible(False)

    def _on_mic_replied(self, reply):
        self.aria_reply.setText(f"ARIA: {reply}")
        self.aria_reply.setVisible(True)

    def _on_mic_command(self, cmd):
        responses = {
            "mute":      "Got it, going quiet now.",
            "unmute":    "Alerts back on, I'm watching.",
            "emergency": "Emergency acknowledged — please stop the vehicle right now!",
            "exit":      "Shutting down. Drive safe out there.",
        }
        msg = responses.get(cmd, "")
        if cmd == "mute":
            if not self._muted: self._toggle_mute()
        elif cmd == "unmute":
            if self._muted: self._toggle_mute()
        elif cmd == "emergency":
            pass  # just speak it loudly
        elif cmd == "exit":
            QTimer.singleShot(2400, self.close)
        if msg:
            speak(msg, 1.0 if cmd == "emergency" else 0.85)
            self.aria_reply.setText(f"ARIA: {msg}")
            self.aria_reply.setVisible(True)

    def closeEvent(self, e):
        self._worker.stop()
        self._worker.wait(3000)
        stop_speaking()
        _speech_q.put(None)
        cv2.destroyAllWindows()
        e.accept()


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setApplicationName("Vehicle Safety AI")
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())
