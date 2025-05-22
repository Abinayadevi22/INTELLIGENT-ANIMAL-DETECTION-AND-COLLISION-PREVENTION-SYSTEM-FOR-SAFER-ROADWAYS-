from ultralytics import YOLO
import cv2
import math 
import time
import serial

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Initialize serial communication
ser = serial.Serial("COM9", 9600, timeout=2)

model = YOLO("last.pt")

classNames = ['Bear', 'Deer', 'Elephant', 'Fox', 'Leopard', 'Panther', 'cat', 'cheetah', 'dog', 'hyena', 'lion', 'tiger', 'wolf']
acls = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']

DOG_CLASS_INDEX = 8  # Index of 'dog' in classNames

time.sleep(2)

prev_detected = False
last_sent_time = 0
delay = 30  # seconds

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True, verbose=False, conf=0.8)

    dog_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == DOG_CLASS_INDEX:
                dog_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = round(box.conf[0].item(), 2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.putText(img, "dog", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                current_time = time.time()
                if not prev_detected or (current_time - last_sent_time >= delay):
                    ser.write('*d'.encode())  # Send 'i' for dog
                    print(ser.write('*d'.encode()))  # 'i' is acls[8]
                    prev_detected = True
                    last_sent_time = current_time
                break  # Exit after first dog detected

    if not dog_detected:
        prev_detected = False

    cv2.imshow('image', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
