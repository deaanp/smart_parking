import json
from shapely.geometry import Polygon, Point

with open("./configs/slots_scaled.json") as f:
    slots = json.load(f)

def get_slot_status(vehicle_boxes):

    occupied = []

    for slot in slots:

        slot_polygon = Polygon(slot)

        is_occupied = False

        for box in vehicle_boxes:

            x1, y1, x2, y2 = box

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            point = Point(cx, cy)

            if slot_polygon.contains(point):
                is_occupied = True
                break

        occupied.append(is_occupied)

    return occupied