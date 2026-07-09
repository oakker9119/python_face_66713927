import os
import cv2
import pickle
import numpy as np

from keras_facenet import FaceNet
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------
# Load FaceNet model
# ----------------------------
embedder = FaceNet()

dataset_path = "dataset"

X = []
y = []

# ----------------------------
# Load dataset
# ----------------------------
for person_name in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path)

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 160))

        embedding = embedder.embeddings([img])[0]

        X.append(embedding)
        y.append(person_name)

# ----------------------------
# Convert to numpy
# ----------------------------
X = np.asarray(X)

# 🔥 Normalize embedding (สำคัญมาก)
X = X / np.linalg.norm(X, axis=1, keepdims=True)

# ----------------------------
# Encode labels
# ----------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ----------------------------
# Split dataset
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# ----------------------------
# Train SVM
# ----------------------------
model = SVC(
    kernel='linear',
    probability=True,
    C=1.0
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ----------------------------
# Save model
# ----------------------------
pickle.dump(model, open("facenet_svm.pkl", "wb"))
pickle.dump(encoder, open("label_encoder.pkl", "wb"))

print("Training Complete ✔")
