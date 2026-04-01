import pyorbbecsdk as ob
import numpy as np
import cv2
import torch
import threading
import time
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ── Config ──────────────────────────────────────────
YOLO_ENGINE  = '/home/yahboom/models/yolo/yolov8n.engine'
PHI35_PATH   = '/home/yahboom/models/phi35'
INFERENCE_HZ = 1
SPEAKER_CONNECTED = False  # Set True when speaker available

# ── Safety Thresholds ───────────────────────────────
DANGER_ZONE_MM  = 1500
WARNING_ZONE_MM = 3000
CAUTION_ZONE_MM = 5000
SPEED_LIMIT_KMH = 30

# ── Object Classes ───────────────────────────────────
PEDESTRIAN_CLASSES = ['person', 'bicycle', 'motorcycle']
VEHICLE_CLASSES    = ['car', 'truck', 'bus', 'train']
OBSTACLE_CLASSES   = [
    'stop sign', 'traffic light',
    'fire hydrant', 'bench', 'chair',
    'dog', 'cat', 'bird'
]

# ── Load YOLO ───────────────────────────────────────
print('Loading YOLOv8n TensorRT...')
yolo = YOLO(YOLO_ENGINE, task='detect')
print('✅ YOLO ready!')

# ── Load Phi-3.5 Mini ───────────────────────────────
print('Loading Phi-3.5 Mini 4-bit...')
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
tok = AutoTokenizer.from_pretrained(PHI35_PATH)
llm = AutoModelForCausalLM.from_pretrained(
    PHI35_PATH,
    quantization_config=bnb,
    device_map='auto',
    trust_remote_code=True,
    attn_implementation='eager'
)
llm.eval()
print('✅ Phi-3.5 Mini ready!')

# ── Safety Assessment ────────────────────────────────
def assess_safety(objects, nearest_mm, depth):
    pedestrians = [o for o in objects
                   if o['label'] in PEDESTRIAN_CLASSES]
    vehicles    = [o for o in objects
                   if o['label'] in VEHICLE_CLASSES]
    obstacles   = [o for o in objects
                   if o['label'] in OBSTACLE_CLASSES]

    h, w = depth.shape
    road_roi   = depth[h//2:, w//4:3*w//4]
    road_valid = road_roi[road_roi > 0]
    road_m     = float(road_valid.mean()) / 1000 \
                 if len(road_valid) else 10.0

    if nearest_mm < DANGER_ZONE_MM:
        level = 'DANGER'
    elif nearest_mm < WARNING_ZONE_MM:
        level = 'WARNING'
    elif nearest_mm < CAUTION_ZONE_MM:
        level = 'CAUTION'
    else:
        level = 'SAFE'

    return {
        'level':       level,
        'pedestrians': pedestrians,
        'vehicles':    vehicles,
        'obstacles':   obstacles,
        'road_m':      road_m,
        'nearest_mm':  nearest_mm,
        'drivable':    nearest_mm > WARNING_ZONE_MM
    }

# ── LLM Description ──────────────────────────────────
def get_ai_instruction(safety):
    peds  = safety['pedestrians']
    vehs  = safety['vehicles']
    level = safety['level']
    road  = safety['road_m']

    ped_str = ', '.join(
        f"person at {p['dist_m']:.1f}m"
        for p in peds[:3]
    ) if peds else 'none'

    veh_str = ', '.join(
        f"{v['label']} at {v['dist_m']:.1f}m"
        for v in vehs[:3]
    ) if vehs else 'none'

    prompt = (
        f"<|system|>You are a vehicle safety AI. "
        f"Give ONE short driving instruction. "
        f"Max 20 words. No explanations.<|end|>\n"
        f"<|user|>"
        f"Alert: {level}. "
        f"Road clear: {road:.1f}m. "
        f"Pedestrians: {ped_str}. "
        f"Vehicles: {veh_str}. "
        f"Speed limit: {SPEED_LIMIT_KMH}kmh. "
        f"Give instruction.<|end|>\n"
        f"<|assistant|>"
    )

    inputs = tok(prompt, return_tensors='pt')
    with torch.inference_mode():
        out = llm.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    if '<|assistant|>' in text:
        text = text.split('<|assistant|>')[-1].strip()
    # Keep only first sentence
    text = text.split('.')[0].strip() + '.'
    return text

# ── Draw HUD ─────────────────────────────────────────
def draw_hud(frame, safety, objects, ai_text):
    h, w = frame.shape[:2]

    # Alert colors
    colors = {
        'SAFE':    (0, 200, 0),
        'CAUTION': (0, 200, 200),
        'WARNING': (0, 140, 255),
        'DANGER':  (0, 0, 255)
    }
    level = safety['level']
    color = colors.get(level, (255, 255, 255))

    # ── Top bar ──
    cv2.rectangle(frame, (0, 0), (w, 65),
                  (20, 20, 20), -1)

    # Alert level badge
    badge_colors = {
        'SAFE':    (0, 180, 0),
        'CAUTION': (0, 180, 180),
        'WARNING': (0, 120, 255),
        'DANGER':  (0, 0, 220)
    }
    bc = badge_colors.get(level, (100, 100, 100))
    cv2.rectangle(frame, (5, 5), (200, 58), bc, -1)
    cv2.putText(frame, level,
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2)

    # Nearest distance
    dist_text = f"Nearest: {safety['nearest_mm']/1000:.1f}m"
    cv2.putText(frame, dist_text,
                (210, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2)

    # Road status
    road_text = (f"Road: {safety['road_m']:.1f}m "
                 f"{'CLEAR' if safety['drivable'] else 'BLOCKED'}")
    road_color = ((0, 200, 0) if safety['drivable']
                  else (0, 0, 255))
    cv2.putText(frame, road_text,
                (210, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, road_color, 2)

    # Speed limit sign
    cv2.circle(frame, (w - 60, 40), 35,
               (0, 0, 200), -1)
    cv2.circle(frame, (w - 60, 40), 35,
               (255, 255, 255), 3)
    cv2.putText(frame, str(SPEED_LIMIT_KMH),
                (w - 82, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(frame, 'km/h',
                (w - 88, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (200, 200, 200), 1)

    # ── Detection boxes ──
    for obj in objects:
        if obj['label'] in PEDESTRIAN_CLASSES:
            box_color = (0, 0, 255)
            icon = '🚶'
        elif obj['label'] in VEHICLE_CLASSES:
            box_color = (0, 165, 255)
            icon = '🚗'
        else:
            box_color = (255, 255, 0)
            icon = '⚠'

        x1, y1, x2, y2 = obj['box']
        cv2.rectangle(frame,
                     (x1, y1), (x2, y2),
                     box_color, 2)

        # Label background
        label = (f"{obj['label']} "
                f"{obj['dist_m']:.1f}m "
                f"{obj['conf']:.0%}")
        lw = len(label) * 9
        cv2.rectangle(frame,
                     (x1, y1 - 22),
                     (x1 + lw, y1),
                     box_color, -1)
        cv2.putText(frame, label,
                   (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.55, (255, 255, 255), 1)

    # ── Depth colormap strip ──
    # (shown as small overlay top-right)

    # ── AI instruction bar ──
    if ai_text:
        bar_h = 45
        cv2.rectangle(frame,
                     (0, h - bar_h),
                     (w, h),
                     (20, 20, 20), -1)
        cv2.putText(frame,
                   f'AI: {ai_text}',
                   (10, h - 12),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.75,
                   color, 2)

    # ── Stats sidebar ──
    n_ped = len(safety['pedestrians'])
    n_veh = len(safety['vehicles'])
    n_obs = len(safety['obstacles'])

    cv2.rectangle(frame,
                 (w - 160, 80),
                 (w, 200),
                 (20, 20, 20), -1)
    cv2.putText(frame, f'Persons: {n_ped}',
                (w - 150, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if n_ped else (200, 200, 200),
                1)
    cv2.putText(frame, f'Vehicles: {n_veh}',
                (w - 150, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255) if n_veh else (200, 200, 200),
                1)
    cv2.putText(frame, f'Objects: {n_obs}',
                (w - 150, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0) if n_obs else (200, 200, 200),
                1)
    cv2.putText(frame,
                time.strftime('%H:%M:%S'),
                (w - 150, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)

    return frame

# ── Decode Frame ─────────────────────────────────────
def decode_color(frame):
    raw = np.frombuffer(frame.get_data(), dtype=np.uint8)
    h, w = frame.get_height(), frame.get_width()
    if len(raw) == h * w * 3:
        return raw.reshape(h, w, 3)
    dec = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if dec is not None:
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    if len(raw) == h * w * 2:
        yuv = raw.reshape(h, w, 2)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_YUYV)
    return None

# ── Camera Setup ─────────────────────────────────────
print('Starting Femto Bolt...')
pipeline = ob.Pipeline()
cfg = ob.Config()
cfg.enable_stream(ob.OBStreamType.COLOR_STREAM)
cfg.enable_stream(ob.OBStreamType.DEPTH_STREAM)
cfg.set_align_mode(ob.OBAlignMode.SW_MODE)
pipeline.start(cfg)
print('✅ Camera ready!')
print('\n🚗 Vehicle Safety System — OFFLINE MODE')
print('Press Q to quit\n')

# ── State ────────────────────────────────────────────
last_inference = 0
ai_instruction = 'Initializing AI...'
current_safety = {
    'level':       'SAFE',
    'pedestrians': [],
    'vehicles':    [],
    'obstacles':   [],
    'road_m':      10.0,
    'nearest_mm':  9999,
    'drivable':    True
}
all_objects = []

# ── AI Thread ────────────────────────────────────────
ai_running = False

def run_ai(rgb, depth, objects, nearest_mm):
    global current_safety, ai_instruction, ai_running
    ai_running = True
    try:
        safety = assess_safety(objects, nearest_mm, depth)
        current_safety = safety
        instruction    = get_ai_instruction(safety)
        ai_instruction = instruction
        level = safety['level']
        print(f'[{level}] {instruction}')
        print(f'  Persons:{len(safety["pedestrians"])} '
              f'Vehicles:{len(safety["vehicles"])} '
              f'Nearest:{nearest_mm}mm')
    except Exception as e:
        print('AI error:', e)
    finally:
        ai_running = False

# ── Main Loop ────────────────────────────────────────
try:
    while True:
        frames = pipeline.wait_for_frames(1000)
        if not frames:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        rgb = decode_color(color_frame)
        if rgb is None:
            continue

        depth = np.frombuffer(
            depth_frame.get_data(), dtype=np.uint16
        ).reshape(depth_frame.get_height(),
                   depth_frame.get_width())

        # Always run YOLO for display
        results = yolo(rgb, verbose=False, half=True)[0]
        all_objects = []
        for box in results.boxes:
            label = results.names[int(box.cls)]
            conf  = float(box.conf)
            if conf < 0.35:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            sx  = depth.shape[1] / rgb.shape[1]
            sy  = depth.shape[0] / rgb.shape[0]
            roi = depth[int(y1*sy):int(y2*sy),
                       int(x1*sx):int(x2*sx)]
            valid  = roi[roi > 0]
            dist_m = float(valid.mean()) / 1000 \
                     if len(valid) else -1
            all_objects.append({
                'label':  label,
                'conf':   conf,
                'dist_m': dist_m,
                'box':    (x1, y1, x2, y2)
            })

        valid_depth = depth[depth > 0]
        nearest_mm  = int(valid_depth.min()) \
                      if len(valid_depth) else 9999

        # Display
        bgr     = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        display = draw_hud(
            bgr, current_safety,
            all_objects, ai_instruction
        )
        cv2.imshow('Vehicle Safety System', display)

        # Depth view (separate window)
        depth_vis = cv2.normalize(
            depth, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_color = cv2.applyColorMap(
            depth_vis, cv2.COLORMAP_JET
        )
        cv2.imshow('Depth Map', depth_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Run LLM in background thread
        now = time.time()
        if (now - last_inference > 1.0 / INFERENCE_HZ
                and not ai_running):
            last_inference = now
            t = threading.Thread(
                target=run_ai,
                args=(rgb.copy(), depth.copy(),
                      all_objects.copy(), nearest_mm),
                daemon=True
            )
            t.start()

except KeyboardInterrupt:
    print('\nStopping...')
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print('✅ System stopped.')
