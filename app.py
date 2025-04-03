import torch
import cv2
from ultralytics import YOLO
model=YOLO("yolov8l.pt")
cap=cv2.VideoCapture('samples\Traffic Control CCTV.mp4')
liscence_plate_detector=YOLO('license_plate_detector.pt')
while True:
    _,img=cap.read()
    img=cv2.resize(img,(1280,720))
    plate=liscence_plate_detector(img,stream=True)
    for i in plate:
        boxes=i.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, b)
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            conf = round(float(box.conf[0]), 2)
            cls = int(box.cls[0])
            label = f"{'Plate'} {conf}"
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()