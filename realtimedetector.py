import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("emotiondetector.keras")

# Load face detection classifier
haar_file = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

# Emotion labels (must match your model's training order)
labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise"
}

# Open webcam
webcam = cv2.VideoCapture(0)

if not webcam.isOpened():
    print("Error: Could not open webcam.")
    raise SystemExit

try:
    while True:
        success, frame = webcam.read()

        if not success:
            print("Error: Could not read webcam frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))

            # Prepare image for model
            feature = np.asarray(face, dtype=np.float32)
            feature = feature.reshape(1, 48, 48, 1) / 255.0

            # Predict emotion
            prediction = model.predict(feature, verbose=0)
            emotion = labels[int(np.argmax(prediction))]

            # Draw rectangle and prediction
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (255, 0, 0), 2
            )

            cv2.putText(
                frame,
                emotion,
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("Real-Time Emotion Detection", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    webcam.release()
    cv2.destroyAllWindows()