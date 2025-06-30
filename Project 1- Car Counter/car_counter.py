import cv2
import cvzone
from ultralytics import YOLO

model = YOLO("../Yolo-Weights/yolov8n.pt")

cap = cv2.VideoCapture("../videos/cars.mp4")

print(cap.read())

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("mask.png")

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
            conf = round(float(box.conf[0]),2)               # Confidence
            cls = box.cls[0]                   # Class 
            
            w, h = x2-x1, y2-y1
            

            currentclass = classNames[int(cls)]

            if currentclass in ['car','truck','bus','motorbike'] and conf > 0.3:
                # Draw rectangle and label
                cvzone.cornerRect(img,(x1,y1,w,h), l=9)
                cvzone.putTextRect(img,f"{currentclass} {conf}",(max(0,x1),max(35,y1-20)), scale=1,thickness=1, offset=5)

            

    # Show image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
