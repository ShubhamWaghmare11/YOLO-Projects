import cv2
import cvzone
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("../Yolo-Weights/yolov8n.pt")
cap.set(3,640)
cap.set(4,480)

while True:
    success, img = cap.read()
    if not success:
        break


    results = model(img, stream=True)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])               # Confidence
            cls = int(box.cls[0])                   # Class ID

            # Draw rectangle and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f'{model.names[cls]} {conf:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Show image
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
