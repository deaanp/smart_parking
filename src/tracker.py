from shapely.geometry import Polygon

OVERLAP_THRESHOLD = 0.50


def assign_vehicle_to_slot(tracks, slots):
    slot_polygons = [Polygon(slot) for slot in slots]
    slot_candidates = {}  

    for track in tracks:
        x1, y1, x2, y2 = track["box"]
        vehicle_poly = Polygon([
            (x1, y1), (x2, y1),
            (x2, y2), (x1, y2)
        ])
        vehicle_area = vehicle_poly.area
        if vehicle_area <= 0:
            track["slot"] = None
            continue

        best_slot    = None
        best_overlap = 0.0

        for i, slot_poly in enumerate(slot_polygons):
            slot_area = slot_poly.area
            if slot_area <= 0:
                continue

            intersection_area = slot_poly.intersection(vehicle_poly).area
            if intersection_area <= 0:
                continue

            slot_overlap    = intersection_area / slot_area
            vehicle_overlap = intersection_area / vehicle_area
            overlap_ratio   = max(slot_overlap, vehicle_overlap)

            if overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_slot    = i

        if best_overlap > OVERLAP_THRESHOLD:
            track["slot"]    = best_slot
            track["overlap"] = best_overlap

            if best_slot not in slot_candidates or \
               best_overlap > slot_candidates[best_slot]["overlap"]:
                slot_candidates[best_slot] = track
        else:
            track["slot"] = None

    return list(slot_candidates.values())