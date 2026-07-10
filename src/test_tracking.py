from ultralytics import YOLO
import cv2

model = YOLO("./models/detector/best.pt")

cap = cv2.VideoCapture("./videos/parkinglot.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.3
    )

    annotated = results[0].plot()

    cv2.imshow("Tracking", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()