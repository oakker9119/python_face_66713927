import os
import cv2
import pickle
import numpy as np
from keras_facenet import FaceNet

def main():
    # 1. โหลด FaceNet Embedder
    print("⏳ กำลังโหลด FaceNet Model...")
    embedder = FaceNet()
    
    # 2. โหลด SVM classifier และ Label Encoder
    svm_path = "facenet_svm.pkl"
    encoder_path = "label_encoder.pkl"
    
    if not os.path.exists(svm_path) or not os.path.exists(encoder_path):
        print("❌ ไม่พบไฟล์โมเดล facenet_svm.pkl หรือ label_encoder.pkl กรุณารัน train_facenet.py ก่อน")
        return
        
    print("⏳ กำลังโหลด SVM Model และ Label Encoder...")
    with open(svm_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoder = pickle.load(f)
        
    # 3. โหลด Face Detector (Haar Cascade)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("❌ โหลด Haar Cascade Face Detector ไม่สำเร็จ")
        return
        
    # 4. เปิดกล้อง Webcam
    print("📸 กำลังเปิดกล้อง...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n🔍 เริ่มระบบรู้จำใบหน้า Real-time")
    print("กด 'Q' เพื่อออกจากโปรแกรม\n")
    
    # ค่า Threshold สำหรับคนแปลกหน้า (ถ้าน้อยกว่านี้จะถือว่าเป็น Unknown)
    CONFIDENCE_THRESHOLD = 0.50
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ไม่สามารถดึงภาพจากกล้องได้")
            break
            
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ตรวจจับใบหน้า
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(80, 80)
        )
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # เตรียมรูปสำหรับ FaceNet
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (160, 160))
            
            # ดึง Face embedding
            embedding = embedder.embeddings([face_resized])[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            embedding = np.expand_dims(embedding, axis=0)
            
            # ทำนายผล
            probabilities = model.predict_proba(embedding)[0]
            max_idx = np.argmax(probabilities)
            confidence = probabilities[max_idx]
            
            if confidence >= CONFIDENCE_THRESHOLD:
                label_id = model.classes_[max_idx]
                name = encoder.inverse_transform([label_id])[0]
                display_name = name
                color = (0, 255, 0) # เขียวเมื่อเจอคนที่รู้จัก
            else:
                display_name = "Unknown"
                color = (0, 0, 255) # แดงเมื่อไม่รู้จัก / ความมั่นใจต่ำ
                
            # วาดกรอบและแสดงชื่อ
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            
            label_text = f"{display_name} ({confidence*100:.1f}%)"
            cv2.putText(display, label_text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
        cv2.imshow("Face Recognition", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("👋 ปิดโปรแกรมเรียบร้อย")

if __name__ == "__main__":
    main()
