import cv2
import numpy as np
import json

slots = []
current_slot = []

# Load image
image = cv2.imread("./sample_frame.jpg")
if image is None:
    print("Error: Gambar tidak ditemukan!")
    exit()

height, width = image.shape[:2]

def mouse(event, x, y, flags, param):
    global current_slot
    if event == cv2.EVENT_LBUTTONDOWN:
        current_slot.append([x, y])
        if len(current_slot) == 4:
            slots.append(current_slot.copy())
            current_slot = []

cv2.namedWindow("Parking Slots", cv2.WINDOW_NORMAL)

cv2.setMouseCallback("Parking Slots", mouse)

while True:
    temp = image.copy()

    for idx, slot in enumerate(slots):
        pts = np.array(slot, np.int32)
        
        cv2.polylines(temp, [pts], True, (0, 255, 0), 2)

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))

        cv2.putText(
            temp, str(idx + 1), (cx, cy), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
        )

    cv2.imshow("Parking Slots", temp)

    key = cv2.waitKey(1)
    if key == ord("u"):
        if len(slots) > 0:
            slots.pop()
    elif key == ord("s"):
        with open("./configs/slots.json", "w") as f:
            json.dump(slots, f, indent=4)
        print("slots.json tersimpan")
        break
    elif key == 27: 
        break

cv2.destroyAllWindows()