import cv2
import os

video_path = "./videos/parkinglot.mp4"
output_dir = "./datasets/vehicle_detection/images"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # simpan setiap 15 frame
    if frame_count % 15 == 0:
        filename = os.path.join(
            output_dir,
            f"frame_{saved_count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

cap.release()

print(f"Frame tersimpan: {saved_count}")
import os

print(os.path.exists(video_path))