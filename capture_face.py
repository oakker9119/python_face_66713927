import cv2
import os
import numpy as np

# ใช้ DNN-based detector แม่นกว่า Haar Cascade มาก
def load_face_detector():
    model_file = "opencv_face_detector_uint8.pb"
    config_file = "opencv_face_detector.pbtxt"
    
    # ถ้าไม่มีไฟล์ DNN ให้ fallback ใช้ Haar แทน
    if os.path.exists(model_file) and os.path.exists(config_file):
        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
        return "dnn", net
    else:
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        return "haar", cascade

def detect_faces_haar(cascade, frame, min_face_size=80):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # เพิ่ม minNeighbors เป็น 8 ลด false positive
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(min_face_size, min_face_size)
    )
    return faces if len(faces) > 0 else []

def is_blurry(image, threshold=80):
    """ตรวจว่ารูปเบลอไหม — Laplacian variance ต่ำ = เบลอ"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < threshold, variance

def is_too_dark_or_bright(image, dark_thresh=40, bright_thresh=220):
    """ตรวจแสง — มืดหรือสว่างเกินไป"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness < dark_thresh or mean_brightness > bright_thresh

def capture_dataset():
    user_id = input("Enter User ID (e.g. student_001): ").strip()
    user_name = input("Enter Name: ").strip()
    
    dataset_path = "dataset"
    person_path = os.path.join(dataset_path, f"{user_id}_{user_name}")
    os.makedirs(person_path, exist_ok=True)
    
    # เช็คว่ามีรูปอยู่แล้วไหม — ต่อจากที่มีได้เลย
    existing = len([f for f in os.listdir(person_path) if f.endswith('.jpg')])
    count = existing
    max_images = 150
    
    if existing > 0:
        print(f"⚠️  พบรูปเดิม {existing} รูป — จะเก็บต่อจนครบ {max_images}")
    
    detector_type, detector = load_face_detector()
    print(f"🔍 ใช้ detector: {detector_type.upper()}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    skipped_blur = 0
    skipped_light = 0
    
    # คำแนะนำท่าทาง — สลับทุก 30 รูป
    poses = [
        "มองตรง",
        "หันซ้ายนิดหน่อย",
        "หันขวานิดหน่อย",
        "เงยหน้าขึ้น",
        "ก้มหน้าลง"
    ]
    
    print(f"\n📸 เริ่มเก็บรูป {user_name}")
    print("กด Q เพื่อหยุด\n")
    
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        display = frame.copy()
        
        # ตรวจจับหน้า
        faces = detect_faces_haar(detector, frame, min_face_size=80)
        
        current_pose = poses[(count // 30) % len(poses)]
        saved_this_frame = False
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            # ตรวจ blur
            blurry, blur_val = is_blurry(face_roi, threshold=80)
            
            # ตรวจแสง
            bad_light = is_too_dark_or_bright(face_roi)
            
            # กำหนดสีกรอบตาม quality
            if blurry:
                color = (0, 165, 255)  # ส้ม = เบลอ
                status = f"BLUR ({blur_val:.0f})"
                skipped_blur += 1
            elif bad_light:
                color = (0, 0, 255)   # แดง = แสงไม่ดี
                status = "BAD LIGHT"
                skipped_light += 1
            else:
                color = (0, 255, 0)   # เขียว = OK
                status = "OK"
                
                # เก็บทุก 3 frame เพื่อความหลากหลาย
                if frame_count % 3 == 0 and not saved_this_frame:
                    count += 1
                    face_resized = cv2.resize(face_roi, (160, 160))
                    save_path = os.path.join(person_path, f"{count}.jpg")
                    cv2.imwrite(save_path, face_resized)
                    saved_this_frame = True
            
            # วาดกรอบ
            cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
            cv2.putText(display, status, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # UI overlay
        h_frame = display.shape[0]
        
        # Progress bar
        progress = int((count / max_images) * 640)
        cv2.rectangle(display, (0, h_frame-20), (progress, h_frame), (0, 255, 0), -1)
        cv2.rectangle(display, (0, h_frame-20), (640, h_frame), (100, 100, 100), 1)
        
        # ข้อความ
        cv2.putText(display, f"Progress: {count}/{max_images}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Pose: {current_pose}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display, f"Skip blur:{skipped_blur} light:{skipped_light}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        
        # แจ้งเตือนเมื่อใกล้ครบแต่ละ pose
        if count % 30 == 25:
            cv2.putText(display, f"⚡ เปลี่ยนท่าได้เลย!", (180, h_frame-35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Capture Dataset", display)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⚠️  หยุดก่อนครบ")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ เสร็จแล้ว! เก็บได้ {count} รูป")
    print(f"   📁 บันทึกที่: {person_path}")
    print(f"   🚫 ข้ามเพราะเบลอ: {skipped_blur} รูป")
    print(f"   🚫 ข้ามเพราะแสงไม่ดี: {skipped_light} รูป")

if __name__ == "__main__":
    capture_dataset()