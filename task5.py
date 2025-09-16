from sklearn.svm import SVC 
import os 
import numpy as np
import cv2 
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
print("Code is running...")
def load_images_with_labels(folder):
    X, y = [], []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32,32))
            features, _ = hog(img, orientations=9, pixels_per_cell=(8,8),
                              cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)
            X.append(features)
            if "cat" in img_name.lower():
                y.append(0)
            elif "dog" in img_name.lower():
                y.append(1)
        except:
            continue
    return np.array(X), np.array(y)
X, y = load_images_with_labels("train")
print("Training data loaded:", X.shape, "Labels:", y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n Training SVM model...")
svm_clf = SVC(kernel='linear', C=1.0)
svm_clf.fit(X_train, y_train)
print("\n Evaluating model...")
y_pred_val = svm_clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_pred_val))
print("Classification Report:\n", classification_report(y_val, y_pred_val, target_names=["Cat","Dog"]))
def load_test_images(folder):
    X, names = [], []
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (32,32))
            features, _ = hog(img, orientations=9, pixels_per_cell=(8,8),
                              cells_per_block=(2,2), block_norm='L2-Hys', visualize=True)
            X.append(features)
            names.append(img_name)
        except:
            continue
    return np.array(X), names
X_test, test_names = load_test_images("test")
print("\nTest data loaded:", X_test.shape)
y_pred_test = svm_clf.predict(X_test)
print("\nPredictions on Test folder:")
for name, pred in zip(test_names, y_pred_test):
    label = "Cat" if pred == 0 else "Dog"
    print(f"{name} --> {label}")
