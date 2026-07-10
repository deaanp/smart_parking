import cv2
import json
import os
import numpy as np

from collections import deque
from datetime    import datetime

from ultralytics import YOLO

from tracker            import assign_vehicle_to_slot
from events             import detect_events
from vehicle_classifier import classify_vehicle, remove_from_cache, debug_dump

# ============================================================
# CONFIG
# ============================================================

PROC_W = 854
PROC_H = 480

DISPLAY_W = 1180
SCALE     = DISPLAY_W / PROC_W
DISPLAY_H = int(PROC_H * SCALE)
SIDEBAR_W = int(360 * SCALE)

WARMUP_FRAMES = 15

BRIGHTNESS_CHANGE_THRESH = 18.0
SCENE_FREEZE_FRAMES      = 45

# Debounce
DEBOUNCE_ON  = 20
DEBOUNCE_OFF = 10

# Klasifikasi
CLASSIFY_INTERVAL = 4

# Filter non-kendaraan
NON_VEHICLE_NAMES  = {"person", "pedestrian", "people", "human", "cyclist"}
FILTER_PEDESTRIAN  = True
PED_ASPECT         = 1.4
PED_MAX_AREA_RATIO = 0.030
MIN_BOX_AREA_RATIO = 0.0008

EVENT_HISTORY         = deque(maxlen=300)
_last_snapshot_second = None

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Warna (BGR)
C_OCC    = (60,  70,  255)
C_AVAIL  = (90,  220,  90)
C_TEXT   = (235, 238, 240)
C_MUTED  = (150, 165, 175)
C_ACCENT = (200, 215,  60)
C_TITLE  = (60,  220, 230)
C_BORDER = (70,   88, 105)
C_ENTRY  = (90,  210, 110)
C_EXIT   = (70,  165, 255)
C_WARN   = (30,  180, 255)
C_SIDEBG = (18,   23,  29)

# ============================================================
# HELPER SKALA
# ============================================================

def fs(base): return base * SCALE
def px(v):    return int(round(v * SCALE))
def th(base): return max(1, int(round(base * SCALE)))

def put(img, text, x, y, scale, color, thick):
    cv2.putText(img, text, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

def text_h(scale, thick):
    (_, h), b = cv2.getTextSize("Ag", FONT, scale, thick)
    return h + b

# ============================================================
# LOG FILE SETUP
# ============================================================

os.makedirs("logs", exist_ok=True)
_session_start = datetime.now()
LOG_PATH = _session_start.strftime("logs/parking_log_%Y%m%d_%H%M%S.txt")

def _write_log(line: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def _write_log_header(n_slots: int):
    sep = "=" * 72
    _write_log(sep)
    _write_log("  SMART PARKING DETECTION SYSTEM — SESSION LOG")
    _write_log(f"  Mulai     : {_session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    _write_log(f"  Total slot: {n_slots}")
    _write_log(f"  Overlap threshold: 40% (tracker.py: OVERLAP_THRESHOLD=0.40)")
    _write_log(f"  Debounce ON={DEBOUNCE_ON} frame, OFF={DEBOUNCE_OFF} frame")
    _write_log(f"  Scene freeze: {SCENE_FREEZE_FRAMES} frame setelah perubahan cahaya")
    _write_log(sep)
    _write_log("")
    _write_log("Format log:")
    _write_log("  HH:MM:SS.mmm     EVENT         ENTRY/EXIT  S<n>  <jenis>")
    _write_log("  HH:MM:SS.mmm     SCENE_CHANGE  brightness <lama> -> <baru>  (event dibekukan)")
    _write_log("  HH:MM:SS         SNAPSHOT      Occupied=<n>/Total=<n>  [detail]")
    _write_log("")
    _write_log("-" * 72)
    _write_log("")

def _write_event_log(ts: str, action: str, slot_idx: int, vtype: str):
    line = f"{ts:<17}  EVENT         {action:<5}  S{slot_idx+1:<3}  {vtype}"
    _write_log(line)

def _write_scene_change_log(ts: str, old_br: float, new_br: float):
    line = (f"{ts:<17}  SCENE_CHANGE  brightness "
            f"{old_br:.1f} -> {new_br:.1f}  (event dibekukan)")
    _write_log(line)

def _write_snapshot(ts_sec: str, occupied: list, slot_label_mem: dict,
                    n_slots: int, frozen: bool = False):
    occ_count = sum(occupied)
    detail_parts = []
    for i, occ in enumerate(occupied):
        if occ:
            vt = slot_label_mem.get(i, "UNKNOWN") or "UNKNOWN"
            detail_parts.append(f"S{i+1}:{vt}")
    detail = "  ".join(detail_parts) if detail_parts else "-"
    frozen_tag = "  [FROZEN]" if frozen else ""
    line = (f"{ts_sec:<17}  SNAPSHOT      "
            f"Occupied={occ_count}/{n_slots}  [{detail}]{frozen_tag}")
    _write_log(line)

# ============================================================
# MODEL & SLOT
# ============================================================

detector_model = YOLO("./models/detector/best.pt")
print("[INFO] Detector class names:", detector_model.names)

with open("./configs/slots_scaled.json") as f:
    slots = json.load(f)

n_slots = len(slots)
_write_log_header(n_slots)

# ============================================================
# VIDEO + WINDOW
# ============================================================

cap = cv2.VideoCapture("./videos/parkinglot.mp4")
WIN = "Smart Parking System"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, DISPLAY_W + SIDEBAR_W, DISPLAY_H)

# ============================================================
# STATE
# ============================================================

stable_state    = [False] * n_slots
pending_streak  = [0]     * n_slots
vehicle_memory  = {}
slot_label_mem  = {}
slot_exit_type  = {}
frame_count     = 0
last_classified = {}
prev_stable     = [False] * n_slots

# Scene change detection state
_prev_brightness    = None
_scene_freeze_left  = 0
_in_scene_change    = False

FRAME_AREA = PROC_W * PROC_H

# ============================================================
# MAIN LOOP
# ============================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    now_dt      = datetime.now()
    ts_event    = now_dt.strftime("%H:%M:%S.") + f"{now_dt.microsecond // 1000:03d}"
    ts_second   = now_dt.strftime("%H:%M:%S")

    proc = cv2.resize(frame, (PROC_W, PROC_H))

    gray       = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))

    just_scene_changed = False
    if _prev_brightness is not None:
        delta = abs(brightness - _prev_brightness)
        if delta > BRIGHTNESS_CHANGE_THRESH and _scene_freeze_left == 0:
            just_scene_changed  = True
            _scene_freeze_left  = SCENE_FREEZE_FRAMES
            _in_scene_change    = True
            _write_scene_change_log(ts_event, _prev_brightness, brightness)
            print(f"[SCENE CHANGE] frame={frame_count} "
                  f"brightness {_prev_brightness:.1f}→{brightness:.1f} "
                  f"(freeze {SCENE_FREEZE_FRAMES} frame)")

    _prev_brightness = brightness

    if _scene_freeze_left > 0:
        _scene_freeze_left -= 1
        if _scene_freeze_left == 0:
            _in_scene_change = False
            pending_streak = [0] * n_slots
            prev_stable    = stable_state.copy()
            print(f"[SCENE CHANGE] frame={frame_count} freeze selesai, state di-reset")

    #DETECTION + TRACKING
    if frame_count <= WARMUP_FRAMES:
        results = detector_model.predict(
            proc, conf=0.35, imgsz=640, verbose=False, half=False,
        )
        use_dummy_id = True
    else:
        results = detector_model.track(
            proc, persist=True, tracker="bytetrack.yaml",
            conf=0.35, imgsz=640, verbose=False, half=False,
        )
        use_dummy_id = False

    tracks = []
    boxes  = results[0].boxes
    has_id = (not use_dummy_id) and (boxes.id is not None)

    if use_dummy_id:
        for box, cls_id in zip(boxes, boxes.cls.cpu().numpy()):
            cls_name = detector_model.names[int(cls_id)].lower()
            if cls_name in NON_VEHICLE_NAMES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = (x2 - x1), (y2 - y1)
            if w <= 0 or h <= 0:
                continue
            area_ratio = (w * h) / FRAME_AREA
            if MIN_BOX_AREA_RATIO > 0 and area_ratio < MIN_BOX_AREA_RATIO:
                continue
            if FILTER_PEDESTRIAN and (h / w) >= PED_ASPECT \
                    and area_ratio <= PED_MAX_AREA_RATIO:
                continue
            tracks.append({
                "track_id": -1,
                "box"     : [x1, y1, x2, y2],
                "cls_name": cls_name,
            })
    elif has_id:
        for box, track_id, cls_id in zip(
            boxes, boxes.id.cpu().numpy(), boxes.cls.cpu().numpy()
        ):
            cls_name = detector_model.names[int(cls_id)].lower()
            if cls_name in NON_VEHICLE_NAMES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = (x2 - x1), (y2 - y1)
            if w <= 0 or h <= 0:
                continue
            area_ratio = (w * h) / FRAME_AREA
            if MIN_BOX_AREA_RATIO > 0 and area_ratio < MIN_BOX_AREA_RATIO:
                continue
            if FILTER_PEDESTRIAN and (h / w) >= PED_ASPECT \
                    and area_ratio <= PED_MAX_AREA_RATIO:
                continue
            tracks.append({
                "track_id": int(track_id),
                "box"     : [x1, y1, x2, y2],
                "cls_name": cls_name,
            })

    #SLOT ASSIGNMENT
    assignments  = assign_vehicle_to_slot(tracks, slots)
    raw_occupied = [False] * n_slots
    for item in assignments:
        if item["slot"] is not None:
            raw_occupied[item["slot"]] = True

    #DEBOUNCE
    if frame_count <= WARMUP_FRAMES:
        for i in range(n_slots):
            stable_state[i] = raw_occupied[i]
    elif not _in_scene_change:
        for i in range(n_slots):
            if raw_occupied[i] != stable_state[i]:
                pending_streak[i] += 1
                need = DEBOUNCE_ON if raw_occupied[i] else DEBOUNCE_OFF
                if pending_streak[i] >= need:
                    stable_state[i]   = raw_occupied[i]
                    pending_streak[i] = 0
            else:
                pending_streak[i] = 0

    occupied = stable_state

    #CLASSIFICATION
    active_track_ids = set()
    for item in assignments:
        slot_id  = item["slot"]
        track_id = item["track_id"]
        if slot_id is None or not stable_state[slot_id]:
            continue
        if track_id == -1:
            continue
        active_track_ids.add(track_id)

        frames_since = frame_count - last_classified.get(track_id, 0)
        if track_id not in vehicle_memory or frames_since >= CLASSIFY_INTERVAL:
            vehicle_type = classify_vehicle(proc, item["box"], track_id)
            last_classified[track_id] = frame_count
        else:
            vehicle_type = vehicle_memory.get(track_id, {}).get(
                "vehicle_type", "UNKNOWN"
            )

        vehicle_memory[track_id] = {"slot": slot_id, "vehicle_type": vehicle_type}
        slot_label_mem[slot_id]  = vehicle_type
        slot_exit_type[slot_id]  = vehicle_type

    gone_ids = set(vehicle_memory.keys()) - active_track_ids
    for tid in gone_ids:
        remove_from_cache(tid)
        vehicle_memory.pop(tid, None)

    for idx in list(slot_label_mem.keys()):
        if not occupied[idx]:
            slot_exit_type[idx] = slot_label_mem.pop(idx)

    #EVENT DETECTION
    if not _in_scene_change and frame_count > WARMUP_FRAMES:
        combined_label = {}
        combined_label.update(slot_exit_type)
        combined_label.update(slot_label_mem)

        events = detect_events(prev_stable, occupied, combined_label)

        if events:
            for ev in events:
                action   = ev["action"]
                slot_idx = ev["slot"]
                vtype    = ev["vehicle_type"]
                EVENT_HISTORY.append((ts_event, action, slot_idx + 1, vtype))
                _write_event_log(ts_event, action, slot_idx, vtype)

    prev_stable = occupied.copy()

    #SNAPSHOT PER-DETIK
    if ts_second != _last_snapshot_second:
        _write_snapshot(ts_second, occupied, slot_label_mem, n_slots,
                        frozen=_in_scene_change)
        _last_snapshot_second = ts_second

    # ══════════════════════════════════════════════════════════════════════════
    # RENDER VIDEO
    # ══════════════════════════════════════════════════════════════════════════
    canvas = cv2.resize(proc, (DISPLAY_W, DISPLAY_H),
                        interpolation=cv2.INTER_LINEAR)

    overlay = canvas.copy(); drew = False
    for idx, slot in enumerate(slots):
        if occupied[idx]:
            cv2.fillPoly(overlay,
                         [(np.array(slot) * SCALE).astype(np.int32)], C_OCC)
            drew = True
    if drew:
        cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0, canvas)

    overlay = canvas.copy(); drew = False
    for idx, slot in enumerate(slots):
        if not occupied[idx]:
            cv2.fillPoly(overlay,
                         [(np.array(slot) * SCALE).astype(np.int32)], C_AVAIL)
            drew = True
    if drew:
        cv2.addWeighted(overlay, 0.10, canvas, 0.90, 0, canvas)

    for idx, slot in enumerate(slots):
        pts   = (np.array(slot) * SCALE).astype(np.int32)
        color = C_OCC if occupied[idx] else C_AVAIL
        cv2.polylines(canvas, [pts], True, color, th(2))
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        put(canvas, f"S{idx + 1}", cx - px(14), cy - px(2),
            fs(0.42), C_TITLE, th(2))
        label = slot_label_mem.get(idx)
        if occupied[idx] and label and label != "UNKNOWN":
            put(canvas, label, cx - px(18), cy + px(14),
                fs(0.38), C_ACCENT, th(1))

    for item in assignments:
        if item["slot"] is None or not stable_state[item["slot"]]:
            continue
        x1, y1, x2, y2 = [int(v * SCALE) for v in item["box"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 165, 0), th(1))

    if _in_scene_change:
        warn_text = f"SCENE CHANGE — event dibekukan ({_scene_freeze_left} frame)"
        put(canvas, warn_text,
            px(10), DISPLAY_H - px(20), fs(0.42), C_WARN, th(1))

    # Indikator warmup
    if frame_count <= WARMUP_FRAMES:
        put(canvas, f"WARMUP {frame_count}/{WARMUP_FRAMES}",
            px(10), DISPLAY_H - px(20), fs(0.42), C_WARN, th(1))

    # ══════════════════════════════════════════════════════════════════════════
    # SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    sb  = np.full((DISPLAY_H, SIDEBAR_W, 3), C_SIDEBG, np.uint8)
    pad = px(16)
    x0  = pad
    y   = pad + text_h(fs(0.8), th(2))

    occupied_count  = sum(occupied)
    available_count = n_slots - occupied_count

    put(sb, "SMART PARKING", x0, y, fs(0.8), C_TITLE, th(2))
    y += px(10)
    cv2.line(sb, (x0, y), (SIDEBAR_W - pad, y), C_BORDER, th(1))

    y += px(8) + text_h(fs(0.55), th(2))
    put(sb, "OKUPANSI", x0, y, fs(0.55), C_MUTED, th(1))
    rh = text_h(fs(0.7), th(2)) + px(8)
    y += rh; put(sb, f"Occupied  : {occupied_count}", x0, y, fs(0.7), C_OCC,   th(2))
    y += rh; put(sb, f"Available : {available_count}", x0, y, fs(0.7), C_AVAIL, th(2))
    y += rh; put(sb, f"Total     : {n_slots}",         x0, y, fs(0.7), C_TEXT,  th(2))
    y += px(6)

    bar_w = SIDEBAR_W - 2 * pad
    bar_h = px(12)
    ratio = occupied_count / n_slots if n_slots else 0
    cv2.rectangle(sb, (x0, y), (x0 + bar_w, y + bar_h), C_BORDER, th(1))
    cv2.rectangle(sb, (x0, y),
                  (x0 + int(bar_w * ratio), y + bar_h), C_EXIT, -1)
    y += bar_h + px(14)
    cv2.line(sb, (x0, y), (SIDEBAR_W - pad, y), C_BORDER, th(1))

    y += px(8) + text_h(fs(0.55), th(2))
    put(sb, f"JENIS KENDARAAN ({occupied_count})", x0, y, fs(0.55), C_MUTED, th(1))
    type_counts = {}
    for vtype in slot_label_mem.values():
        t = vtype if (vtype and vtype != "UNKNOWN") else "UNKNOWN"
        type_counts[t] = type_counts.get(t, 0) + 1
    rh = text_h(fs(0.55), th(1)) + px(8)
    if type_counts:
        for t, n in sorted(type_counts.items(), key=lambda kv: -kv[1]):
            y += rh
            put(sb, f"{t:<10}: {n}", x0, y, fs(0.55), C_TEXT, th(1))
    else:
        y += rh
        put(sb, "Belum ada kendaraan", x0, y, fs(0.55), C_MUTED, th(1))
    y += px(12)
    cv2.line(sb, (x0, y), (SIDEBAR_W - pad, y), C_BORDER, th(1))

    y += px(8) + text_h(fs(0.55), th(2))
    put(sb, "LOG MASUK / KELUAR", x0, y, fs(0.55), C_MUTED, th(1))
    rh      = text_h(fs(0.48), th(1)) + px(7)
    y      += px(4)
    avail_h = DISPLAY_H - y - pad
    max_lines = max(1, avail_h // rh)
    recent    = list(EVENT_HISTORY)[-max_lines:][::-1]
    if recent:
        for entry in recent:
            y += rh
            t_str, action, slot_no, vtype = entry
            col  = C_ENTRY if action == "ENTRY" else C_EXIT
            line = f"{t_str}  {action:<5} S{slot_no} {vtype}"
            put(sb, line, x0, y, fs(0.46), col, th(1))
    else:
        y += rh
        put(sb, "Belum ada aktivitas", x0, y, fs(0.55), C_MUTED, th(1))

    full = np.hstack([canvas, sb])
    cv2.imshow(WIN, full)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('d'):
        print("\n========== DEBUG KLASIFIKASI ==========")
        for tid, mem in sorted(vehicle_memory.items(),
                               key=lambda kv: kv[1]["slot"]):
            print(f"S{mem['slot']+1:<3} id={tid:<4} -> {debug_dump(tid)}")
        print("=======================================\n")
    elif key == ord('s'):
        print(f"[DEBUG] brightness={_prev_brightness:.1f}  "
              f"freeze_left={_scene_freeze_left}  "
              f"in_scene={_in_scene_change}")

#Cleanup
cap.release()
cv2.destroyAllWindows()

_write_log("")
_write_log("-" * 72)
_write_log(f"  Sesi selesai : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
_write_log(f"  Total frame  : {frame_count}")
_write_log("=" * 72)
print(f"\n[INFO] Log tersimpan di: {LOG_PATH}")