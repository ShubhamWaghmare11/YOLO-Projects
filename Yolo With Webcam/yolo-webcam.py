import cv2
import cvzone
from ultralytics import YOLO

cap = cv2.VideoCapture(0)
model = YOLO("../Yolo-Weights/yolov8n.pt")
cap.set(3,1920)
cap.set(4,1080)

while True:
    success, img = cap.read()
    
    if not success:
        break

    results = model(img, stream=True)


    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, w, h = map(int, box.xywh[0])  # Bounding box
            conf = float(box.conf[0])               # Confidence
            cls = int(box.cls[0])                   # Class ID
            
            bbox = x1,y1,w,h
            # Draw rectangle and label
            cvzone.cornerRect(img,bbox)

    # Show image
    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
