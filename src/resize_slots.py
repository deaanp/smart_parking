import json

ORIGINAL_W = 1918
ORIGINAL_H = 1123

VIDEO_W = 854
VIDEO_H = 480

scale_x = VIDEO_W / ORIGINAL_W
scale_y = VIDEO_H / ORIGINAL_H

with open("./configs/slots.json") as f:
    slots = json.load(f)

new_slots = []

for slot in slots:
    new_slot = []

    for x, y in slot:
        new_x = int(x * scale_x)
        new_y = int(y * scale_y)
        new_slot.append([new_x, new_y])

    new_slots.append(new_slot)

with open("./configs/slots_scaled.json", "w") as f:
    json.dump(new_slots, f, indent=4)

print("DONE: slots sudah di-scale ke resolusi video")