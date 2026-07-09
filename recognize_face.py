import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet

embedder = FaceNet()

model = pickle.load(open("facenet_svm.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        if face.size == 0:
            continue

        face = cv2.resize(face, (160, 160))
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        embedding = embedder.embeddings([rgb_face])[0]

        # 🔥 normalize (สำคัญมาก)
        embedding = embedding / np.linalg.norm(embedding)

        embedding = np.expand_dims(embedding, axis=0)

        pred = model.predict(embedding)[0]
        proba = model.predict_proba(embedding)[0]

        confidence = np.max(proba)
        name = encoder.inverse_transform([pred])[0]

        # 🔥 Unknown condition   ถ้าค่าความเชื่อมั่นน้อยกว่า 70% จะขึ้น Unknown  คือไม่รู้จัก
        if confidence < 0.70:
            name = "Unknown"

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
