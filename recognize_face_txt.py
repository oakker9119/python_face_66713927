import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet
from datetime import datetime
import time
import winsound  # 🔊 ใช้ส่งเสียงแจ้งเตือน (เฉพาะ Windows)

# ==================================================
# โหลดโมเดล FaceNet สำหรับแปลงใบหน้าเป็น Feature Vector
# ==================================================
embedder = FaceNet()

# ==================================================
# โหลดโมเดล SVM ที่เทรนไว้แล้ว
# และ Label Encoder สำหรับแปลงเลขคลาสเป็นชื่อบุคคล
# ==================================================
model = pickle.load(open("facenet_svm.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# ==================================================
# โหลด Haar Cascade สำหรับตรวจจับใบหน้า
# ==================================================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# ==================================================
# เปิดใช้งานกล้อง Webcam
# 0 = กล้องตัวแรกของเครื่อง
# ==================================================
cap = cv2.VideoCapture(0)

# ==================================================
# เปิดไฟล์สำหรับบันทึก Log การตรวจพบใบหน้า
# ==================================================
log_file = open("face_log.txt", "a", encoding="utf-8")

# ==================================================
# เก็บเวลาล่าสุดที่บันทึก Log ของแต่ละคน
# ใช้ป้องกันการบันทึกข้อมูลซ้ำ
# ==================================================
last_log_time = {}

# กำหนดช่วงเวลาการบันทึกซ้ำ (วินาที)
LOG_INTERVAL = 5

# ==================================================
# ควบคุมการส่งเสียงเตือน
# ป้องกันเสียงดังรัวตลอดเวลา
# ==================================================
last_sound_time = 0

# ส่งเสียงได้ทุก 3 วินาที
SOUND_INTERVAL = 3

# ==================================================
# วนลูปอ่านภาพจากกล้องแบบ Real-time
# ==================================================
while True:

    # อ่านภาพจากกล้อง
    ret, frame = cap.read()

    # ถ้าอ่านภาพไม่ได้ให้ออกจากลูป
    if not ret:
        break

    # แปลงภาพเป็น Gray Scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ==================================================
    # ตรวจจับใบหน้า
    # scaleFactor = 1.2
    # minNeighbors = 5
    # ==================================================
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    # ==================================================
    # วนลูปทีละใบหน้าที่ตรวจพบ
    # ==================================================
    for (x, y, w, h) in faces:

        # ตัดเฉพาะบริเวณใบหน้า
        face = frame[y:y+h, x:x+w]

        # ตรวจสอบว่ามีข้อมูลภาพจริง
        if face.size == 0:
            continue

        # ==================================================
        # Resize ให้ตรงกับขนาดที่ FaceNet ต้องการ
        # ==================================================
        face = cv2.resize(face, (160, 160))

        # แปลง BGR → RGB
        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # ==================================================
        # สร้าง Embedding ด้วย FaceNet
        # ==================================================
        embedding = embedder.embeddings([rgb_face])[0]

        # Normalize เวกเตอร์
        embedding = embedding / np.linalg.norm(embedding)

        # เพิ่มมิติให้ SVM รับข้อมูลได้
        embedding = np.expand_dims(embedding, axis=0)

        # ==================================================
        # ทำนายตัวตนด้วย SVM
        # ==================================================
        pred = model.predict(embedding)[0]

        # ความน่าจะเป็นของแต่ละคลาส
        proba = model.predict_proba(embedding)[0]

        # ค่า Confidence สูงสุด
        confidence = np.max(proba)

        # แปลงเลขคลาสเป็นชื่อบุคคล
        name = encoder.inverse_transform([pred])[0]

        # ==================================================
        # ถ้าความมั่นใจต่ำกว่า 90%
        # ถือว่าเป็นบุคคลที่ไม่รู้จัก
        # ==================================================
        if confidence < 0.90:
            name = "Unknown"

        # ==================================================
        # ส่งเสียงแจ้งเตือนเมื่อพบใบหน้า
        # ==================================================
        current_time = time.time()

        if current_time - last_sound_time > SOUND_INTERVAL:

            # ความถี่ 1000 Hz
            # ระยะเวลา 200 ms
            winsound.Beep(1000, 200)

            last_sound_time = current_time

        # ==================================================
        # บันทึก Log โดยไม่ให้บันทึกซ้ำบ่อยเกินไป
        # ==================================================
        if name not in last_log_time:
            last_log_time[name] = 0

        if current_time - last_log_time[name] > LOG_INTERVAL:

            # เวลาปัจจุบัน
            time_now = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # เขียนข้อมูลลงไฟล์
            log_file.write(
                f"{time_now}, {name}, {confidence:.2f}\n"
            )

            # บันทึกลงไฟล์ทันที
            log_file.flush()

            # อัปเดตเวลาล่าสุด
            last_log_time[name] = current_time

        # ==================================================
        # วาดกรอบรอบใบหน้า
        # ==================================================

        # สีเขียว = รู้จัก
        # สีแดง = Unknown
        color = (
            (0, 255, 0)
            if name != "Unknown"
            else (0, 0, 255)
        )

        # วาดสี่เหลี่ยม
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        # แสดงชื่อและความมั่นใจ
        cv2.putText(
            frame,
            f"{name} {confidence:.2f}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # ==================================================
    # แสดงผลภาพจากกล้อง
    # ==================================================
    cv2.imshow("Face Recognition", frame)

    # กด q เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ==================================================
# ปิดการใช้งานอุปกรณ์และไฟล์
# ==================================================
cap.release()          # ปิดกล้อง
log_file.close()       # ปิดไฟล์ Log
cv2.destroyAllWindows()  # ปิดหน้าต่าง OpenCV