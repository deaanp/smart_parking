from ultralytics import YOLO

vehicle_type_model = YOLO("./models/vehicle_type_detector/best.pt")

#Hyperparameter voting
DECAY           = 0.85
SWITCH_MARGIN   = 8.0
MIN_CONFIRM_VOTES = 3.0

#Hyperparameter crop
MIN_BOX_SIZE    = 20
PAD             = 4
CLASSIFY_CONF   = 0.15
CLASSIFY_IMGSZ  = 224

#Koreksi bias kelas
CLASS_MIN_CONF = {
    "TRUCK" : 0.55,
    "PICKUP": 0.45,
}
CLASS_VOTE_WEIGHT = {
    "TRUCK" : 0.4, 
    "PICKUP": 0.5,
}
DEFAULT_MIN_CONF = 0.0

_MIN_CONF    = {k.upper(): v for k, v in CLASS_MIN_CONF.items()}
_VOTE_WEIGHT = {k.upper(): v for k, v in CLASS_VOTE_WEIGHT.items()}

_state = {}


def _committed(track_id):
    if track_id is None:
        return "UNKNOWN"
    st = _state.get(track_id)
    return st["committed"] if st and st["committed"] else "UNKNOWN"


def _select_best_box(boxes, crop_w, crop_h):
    cx0, cy0 = crop_w / 2.0, crop_h / 2.0
    max_dist  = (cx0 ** 2 + cy0 ** 2) ** 0.5 + 1e-6
    best, best_score = None, -1.0

    for b in boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        conf  = float(b.conf[0])
        bw    = x2 - x1
        bh    = y2 - y1
        area_frac = (bw * bh) / (crop_w * crop_h + 1e-6)
        bcx   = (x1 + x2) / 2.0
        bcy   = (y1 + y2) / 2.0
        dist  = ((bcx - cx0) ** 2 + (bcy - cy0) ** 2) ** 0.5
        centrality = 1.0 - min(dist / max_dist, 1.0)
        score = conf * (0.55 + 0.45 * centrality) * (0.6 + 0.4 * min(area_frac, 1.0))

        if score > best_score:
            best_score = score
            best = (int(b.cls[0]), conf)

    return best


def classify_vehicle(frame, box, track_id=None):
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
        return _committed(track_id)

    fh, fw = frame.shape[:2]
    xa = max(0,  x1 - PAD)
    ya = max(0,  y1 - PAD)
    xb = min(fw, x2 + PAD)
    yb = min(fh, y2 + PAD)
    crop = frame[ya:yb, xa:xb]

    if crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
        return _committed(track_id)

    # Inferensi model
    result = vehicle_type_model(
        crop, verbose=False, conf=CLASSIFY_CONF, imgsz=CLASSIFY_IMGSZ
    )
    boxes_res = result[0].boxes

    detected, conf = None, 0.0
    if boxes_res is not None and len(boxes_res) > 0:
        best = _select_best_box(boxes_res, crop.shape[1], crop.shape[0])
        if best is not None:
            detected = vehicle_type_model.names[best[0]]
            conf     = best[1]

    if detected is None:
        return _committed(track_id)

    if track_id is None:
        return detected

    norm = detected.upper()
    if conf < _MIN_CONF.get(norm, DEFAULT_MIN_CONF):
        return _committed(track_id)

    weight = _VOTE_WEIGHT.get(norm, 1.0)

    #Update accumulator
    st = _state.setdefault(track_id, {"counts": {}, "committed": None, "total": 0.0})
    c  = st["counts"]

    for k in list(c):
        c[k] *= DECAY
        if c[k] < 0.01:
            del c[k]   

    # Tambah vote baru
    c[detected] = c.get(detected, 0.0) + weight
    st["total"] += weight

    # Ambil kandidat terkuat
    top = max(c, key=c.get)

    #Commit label
    if st["committed"] is None:
        if c[top] >= MIN_CONFIRM_VOTES:
            st["committed"] = top
    else:
        gap = c[top] - c.get(st["committed"], 0.0)
        if top != st["committed"] and gap >= SWITCH_MARGIN:
            st["committed"] = top

    return st["committed"] or "UNKNOWN"


def reset_cache():
    _state.clear()


def remove_from_cache(track_id):
    _state.pop(track_id, None)


def debug_dump(track_id):
    """Tampilkan tally + committed label untuk debugging."""
    st = _state.get(track_id)
    if not st:
        return None
    return {
        "committed": st["committed"],
        "counts"   : {k: round(v, 2) for k, v in sorted(
                      st["counts"].items(), key=lambda x: -x[1])},
    }