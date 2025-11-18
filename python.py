import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = roi_gray[int(h/2):h, 0:w]
            smiles = smile_cascade.detectMultiScale(roi_gray,scaleFactor=1.4,minNeighbors=10,minSize=(40, 40))

            smile_detected = False
            for (sx, sy, sw, sh) in smiles:
                if sw > w * 0.3:   # smile must be wide enough
                    smile_detected = True

            if smile_detected:
                label = "Smiling :)"
                color = (0, 255, 0)
            else:
                label = "Neutral"
                color = (255, 0, 0)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        clear_output(wait=True)
        plt.imshow(frame_rgb)
        plt.axis('off')
        plt.show()

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

cap.release()
print("Webcam released.")
