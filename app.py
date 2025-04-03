import cv2
from ultralytics import YOLO
import cvzone
from cvzone.Utils import putTextRect
model = YOLO("yolov8l.pt")  # Vehicle detection
license_plate_detector = YOLO('license_plate_detector.pt')
cap = cv2.VideoCapture('samples/Traffic Control CCTV.mp4')
vehicle_classes = ['car', 'truck', 'motorcycle', 'bus']
while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (1280, 720))
    vehicle_results = model(img)
    vehicles = []  # Stores (x1, y1, x2, y2, label, conf)

    for result in vehicle_results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls].lower()
            if label in vehicle_classes and conf > 0.9:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicles.append((x1, y1, x2, y2, label, conf))
    plate_results = license_plate_detector(img)
    plates = []  # Stores (x1, y1, x2, y2)

    for result in plate_results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf > 0.25:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plates.append((x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
                putTextRect(img, f"Plate {conf:.2f}", (x1, y1 - 25), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(0, 255, 0), offset=5)
    for vx1, vy1, vx2, vy2, label, conf in vehicles:
        plate_found = False
        for px1, py1, px2, py2 in plates:
            if px1 >= vx1 and py1 >= vy1 and px2 <= vx2 and py2 <= vy2:
                plate_found = True
                break
        if plate_found:
            w, h = vx2 - vx1, vy2 - vy1
            cvzone.cornerRect(img, (vx1, vy1, w, h), colorC=(0, 0, 255), t=8, rt=0)
            putTextRect(img, f"{label} {conf:.2f}", (vx1, vy1 - 10), scale=1, thickness=2, colorT=(255, 255, 255), colorR=(255, 0, 0), offset=5)

    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
