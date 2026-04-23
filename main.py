import pyorbbecsdk as ob
import numpy as np
import cv2
import threading
import queue
import time
import io
import wave
import random
import subprocess
from piper import PiperVoice
from ultralytics import YOLO

# ── CONFIG ──────────────────────────────────────────
YOLO_ENGINE = '/home/yahboom/models/yolo/yolov8n.engine'

PERSON_STOP_DIST    = 1.0
PERSON_WARNING_DIST = 3.0
CAR_DIST_THRESHOLD  = 5.0

SPEAK_COOLDOWN = 7.0

# ── PIPER TTS CONFIG ─────────────────────────────────
PIPER_MODEL = '/home/yahboom/models/piper/en_US-amy-medium.onnx'

# ── SPEAKER SETUP (thread-safe queue) ───────────────
_speech_queue = queue.Queue()

def _dist_to_volume(dist, is_vehicle=False):
    """
    Person tiers  (0-3m): <1m=1.0, 1-2m=0.50, 2-3m=0.10
    Vehicle tiers (0-5m): <1m=1.0, 1-2m=0.85, 2-3m=0.60, 3-5m=0.30
    """
    if is_vehicle:
        if dist < 1.0:
            return 1.0
        elif dist < 2.0:
            return 0.85
        elif dist < 3.0:
            return 0.60
        else:
            return 0.30
    else:
        if dist < 1.0:
            return 1.0
        elif dist < 2.0:
            return 0.50
        else:
            return 0.10

def _speech_worker():
    voice = PiperVoice.load(PIPER_MODEL)
    sr    = voice.config.sample_rate
    while True:
        item = _speech_queue.get()
        if item is None:
            break
        text, volume = item
        try:
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                for chunk in voice.synthesize(text):
                    pcm = (chunk.audio_float_array * 32767 * volume).clip(-32768, 32767).astype(np.int16)
                    wf.writeframes(pcm.tobytes())
            subprocess.run(
                ['aplay', '-q', '-f', 'cd', '-r', str(sr), '-c', '1', '-t', 'wav', '-'],
                input=buf.getvalue(),
                check=False
            )
        except Exception as e:
            print(f"[TTS error] {e}")

_speech_thread = threading.Thread(target=_speech_worker, daemon=True)
_speech_thread.start()

def speak_async(text, volume=1.0):
    # Drop queued items so only the latest alert plays
    while not _speech_queue.empty():
        try:
            _speech_queue.get_nowait()
        except queue.Empty:
            break
    _speech_queue.put((text, volume))

# ── STATE ───────────────────────────────────────────
last_spoken_time = 0
last_event = None

# ── CAMERA DECODE ───────────────────────────────────
def decode_color(frame):
    raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
    h, w = frame.get_height(), frame.get_width()

    if len(raw) == h * w * 3:
        return raw.reshape(h, w, 3)

    dec = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if dec is not None:
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)

    return None

# ── DIRECTION LOGIC ─────────────────────────────────
def get_direction(cx, width):
    if cx < width * 0.33:
        return "on the left"
    elif cx > width * 0.66:
        return "on the right"
    else:
        return "ahead"

# ── SCALE BBOX FROM COLOR TO DEPTH RESOLUTION ───────
def scale_box(x1, y1, x2, y2, color_w, color_h, depth_w, depth_h):
    sx = depth_w / color_w
    sy = depth_h / color_h
    return (
        int(x1 * sx), int(y1 * sy),
        int(x2 * sx), int(y2 * sy)
    )

# ── LOAD YOLO ───────────────────────────────────────
print("Loading YOLO...")
yolo = YOLO(YOLO_ENGINE, task='detect')
print("✅ YOLO Ready")

# ── DASHBOARD HELPERS ───────────────────────────────
def dist_color_bgr(dist):
    """Box/text color by distance (BGR)."""
    if dist < 1.0:   return (0,   0,   220)   # red
    elif dist < 2.0: return (0,   110, 255)   # orange
    elif dist < 3.0: return (0,   210, 255)   # yellow
    else:            return (50,  210,  50)   # green

def status_style(event):
    """Returns (bgr_color, label) for the current alert event."""
    if event in ("STOP_PERSON", "STOP_CAR"):
        return (0, 0, 180),    "!! STOP !!"
    elif event == "CAR_PERSON":
        return (0, 60, 210),   "HIGH RISK"
    elif event in ("PERSON_WARNING", "CAR_WARNING"):
        return (0, 130, 255),  "CAUTION"
    else:
        return (35, 130, 35),  "ALL CLEAR"

def draw_dashboard(camera_bgr, depth, closest_person, closest_car,
                   person_count, event):
    cam_h, cam_w = camera_bgr.shape[:2]
    s_color, s_text = status_style(event)

    # ── TOP BANNER ───────────────────────────────
    overlay = camera_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (cam_w, 48), s_color, -1)
    cv2.addWeighted(overlay, 0.78, camera_bgr, 0.22, 0, camera_bgr)
    cv2.putText(camera_bgr, s_text, (12, 34),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    ts = time.strftime("%H:%M:%S")
    cv2.putText(camera_bgr, ts, (cam_w - 115, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 210, 210), 1)

    # ── SIDEBAR ──────────────────────────────────
    sb = np.full((cam_h, SIDEBAR_W, 3), 18, dtype=np.uint8)

    # Status block
    cv2.rectangle(sb, (0, 0), (SIDEBAR_W, 90), s_color, -1)
    (tw, _), _ = cv2.getTextSize(s_text, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)
    cv2.putText(sb, s_text, ((SIDEBAR_W - tw) // 2, 60),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    def section(sb, y, title, detected, dist=None, extra=""):
        cv2.putText(sb, title, (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130, 130, 130), 1)
        y += 28
        if detected and dist is not None:
            col = dist_color_bgr(dist)
            cv2.putText(sb, f"{dist:.2f} m", (14, y),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, col, 2)
            y += 38
            if extra:
                cv2.putText(sb, extra, (14, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
                y += 20
        else:
            cv2.putText(sb, "Not detected", (14, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
            y += 22
        return y

    y = 108
    y = section(sb, y, "PERSON",
                closest_person is not None,
                closest_person["dist"]      if closest_person else None,
                f"Count: {person_count}   {closest_person['direction']}" if closest_person else "")
    y += 8
    cv2.line(sb, (14, y), (SIDEBAR_W - 14, y), (50, 50, 50), 1)
    y += 14

    y = section(sb, y, "VEHICLE",
                closest_car is not None,
                closest_car["dist"]         if closest_car else None,
                closest_car["direction"]    if closest_car else "")
    y += 8
    cv2.line(sb, (14, y), (SIDEBAR_W - 14, y), (50, 50, 50), 1)
    y += 14

    # Depth thumbnail
    thumb_w = SIDEBAR_W - 20
    thumb_h = int(thumb_w * depth.shape[0] / depth.shape[1])
    if y + thumb_h + 35 < cam_h:
        cv2.putText(sb, "DEPTH VIEW", (14, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (110, 110, 110), 1)
        depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        thumb     = cv2.resize(cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET),
                               (thumb_w, thumb_h))
        sb[y + 22 : y + 22 + thumb_h, 10 : 10 + thumb_w] = thumb

    return np.hstack([camera_bgr, sb])

# ── DISPLAY CONFIG ──────────────────────────────────
DISPLAY_W, DISPLAY_H = 960, 540   # camera panel
SIDEBAR_W            = 300         # right info panel
DASHBOARD_W          = DISPLAY_W + SIDEBAR_W

pipeline = ob.Pipeline()
cfg = ob.Config()
cfg.enable_stream(ob.OBStreamType.COLOR_STREAM)
cfg.enable_stream(ob.OBStreamType.DEPTH_STREAM)
pipeline.start(cfg)

print("🚗 Safety System Started")

# ── MAIN LOOP ───────────────────────────────────────
try:
    while True:
        try:
            frames = pipeline.wait_for_frames(1000)
        except Exception as e:
            print(f"[camera error] {e}, retrying...")
            time.sleep(0.1)
            continue

        if not frames:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        rgb = decode_color(color_frame)
        if rgb is None:
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        color_h, color_w = bgr.shape[:2]
        depth_h = depth_frame.get_height()
        depth_w = depth_frame.get_width()

        depth = np.frombuffer(
            depth_frame.get_data(), dtype=np.uint16
        ).reshape(depth_h, depth_w)

        try:
            results = yolo(bgr, verbose=False)[0]
        except Exception as e:
            print(f"[yolo error] {e}")
            continue

        closest_person = None
        closest_car    = None
        person_count   = 0

        for box in results.boxes:
            label = results.names[int(box.cls)]
            conf  = float(box.conf)

            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2

            # Scale to depth resolution before sampling
            dx1, dy1, dx2, dy2 = scale_box(
                x1, y1, x2, y2,
                color_w, color_h, depth_w, depth_h
            )
            dx1, dy1 = max(0, dx1), max(0, dy1)
            dx2, dy2 = min(depth_w, dx2), min(depth_h, dy2)

            # Use centre 60% of bbox to avoid background bleed at edges
            margin_y = (dy2 - dy1) // 5
            margin_x = (dx2 - dx1) // 5
            roi   = depth[dy1 + margin_y : dy2 - margin_y,
                          dx1 + margin_x : dx2 - margin_x]
            valid = roi[roi > 0]

            if len(valid) < 10:
                continue

            # 20th percentile: closest front-facing surface, ignores background
            dist = float(np.percentile(valid, 20)) / 1000
            direction = get_direction(cx, color_w)

            # DRAW COLOR-CODED BOX
            col = dist_color_bgr(dist)
            cv2.rectangle(bgr, (x1, y1), (x2, y2), col, 2)
            cv2.rectangle(bgr, (x1, y1 - 22), (x1 + 140, y1), col, -1)
            cv2.putText(bgr, f"{label}  {dist:.1f}m",
                        (x1 + 4, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 1)

            # PERSON
            if label == "person" and dist < PERSON_WARNING_DIST:
                person_count += 1
                if closest_person is None or dist < closest_person["dist"]:
                    closest_person = {"dist": dist, "direction": direction}

            # VEHICLE (any type)
            if label in ["car", "truck", "bus", "motorcycle", "bicycle"] and dist < CAR_DIST_THRESHOLD:
                if closest_car is None or dist < closest_car["dist"]:
                    closest_car = {"dist": dist, "direction": direction}

        # ── EVENT LOGIC ───────────────────────────────
        now   = time.time()
        event = None
        speech = None

        volume   = 1.0
        d_person = closest_person["direction"] if closest_person else ""
        d_car    = closest_car["direction"]    if closest_car    else ""
        p_word   = "person" if person_count == 1 else f"{person_count} people"

        if closest_person:
            volume = _dist_to_volume(closest_person["dist"])

            if closest_person["dist"] < PERSON_STOP_DIST:
                event  = "STOP_PERSON"
                speech = random.choice([
                    f"Stop stop stop! {p_word} is {d_person}, stop now!",
                    f"Whoa whoa! {p_word} extremely close {d_person}, stop!",
                    f"Hey! Stop immediately, {p_word} is {d_person}!",
                    f"Watch out! {p_word} is {d_person}, don't move!",
                ])
            elif closest_car:
                event  = "CAR_PERSON"
                speech = random.choice([
                    f"Whoa! Car and {p_word} ahead — serious danger, slow down now!",
                    f"Watch out! {p_word} and a vehicle ahead, high risk!",
                    f"Hey hey hey! Car and {p_word} detected, stop or slow down!",
                    f"Danger! Both a car and {p_word} ahead, be very careful!",
                ])
            else:
                event  = "PERSON_WARNING"
                speech = random.choice([
                    f"Hey, wait — there's {p_word} {d_person}, slow down.",
                    f"Heads up! {p_word} {d_person}, ease off.",
                    f"Easy there, {p_word} detected {d_person}, take it slow.",
                    f"Watch it! There's {p_word} {d_person}, be careful.",
                ])

        elif closest_car:
            volume = _dist_to_volume(closest_car["dist"], is_vehicle=True)

            if closest_car["dist"] < PERSON_STOP_DIST:
                event  = "STOP_CAR"
                speech = random.choice([
                    f"Stop! Vehicle is {d_car}, don't move!",
                    f"Whoa whoa! Vehicle super close {d_car}, stop now!",
                    f"Hey! Stop immediately, vehicle is {d_car}!",
                ])
            else:
                event  = "CAR_WARNING"
                speech = random.choice([
                    f"Hey, there's a vehicle {d_car}, watch it.",
                    f"Easy, car coming up {d_car}, slow down.",
                    f"Heads up! Vehicle detected {d_car}, proceed carefully.",
                ])

        # ── SPEECH CONTROL ───────────────────────────
        if event:
            if event != last_event or now - last_spoken_time > SPEAK_COOLDOWN:
                print(f"🔊 [vol={volume:.2f}] {speech}")
                speak_async(speech, volume)
                last_event       = event
                last_spoken_time = now
        else:
            last_event = None

        # ── DASHBOARD ───────────────────────────────
        display   = cv2.resize(bgr, (DISPLAY_W, DISPLAY_H))
        dashboard = draw_dashboard(display, depth,
                                   closest_person, closest_car,
                                   person_count, event)
        cv2.imshow("Vehicle Safety Dashboard", dashboard)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopping...")

finally:
    _speech_queue.put(None)  # sentinel to stop speech worker
    _speech_thread.join(timeout=2)
    try:
        pipeline.stop()
    except Exception as e:
        print(f"Warning during shutdown: {e}")
    cv2.destroyAllWindows()
    print("✅ System stopped.")
