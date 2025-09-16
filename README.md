# 🐱🐶 Cat vs Dog Classifier (SVM + HOG Features)

This project implements a **Cat vs Dog image classifier** using **Support Vector Machine (SVM)** and **HOG (Histogram of Oriented Gradients) features**.

## 🚀 Features
- Uses **HOG features** for extracting texture/orientation information from images.  
- **SVM classifier** (Linear kernel) for training.  
- Evaluates model accuracy on a **validation split**.  
- Can generate predictions for **unlabeled test images**.  
- Lightweight (works even with smaller datasets).

## 📂 Project Structure
project/
│── train/ # Training images (must include 'cat' or 'dog' in filenames)
│── test/ # Test images (filenames: 1.jpg, 2.jpg, 3.jpg ... no labels)
│── task3.py # Main Python script
│── README.md # Documentation

## ⚙️ Requirements
Install dependencies before running:
```bash
pip install numpy opencv-python scikit-learn scikit-image
▶️ How to Run
Place your training images in the train/ folder.
Example:
cat.1.jpg, cat.2.jpg, dog.1.jpg, dog.2.jpg ...
Place your test images (without labels) in the test/ folder.
Example:
1.jpg, 2.jpg, 3.jpg ...
Run the script:
python task3.py

📝 Notes
Make sure training filenames include "cat" or "dog" so labels can be assigned automatically.
Test folder images don’t need labels → only predictions are generated.
You can adjust image size (32x32) and HOG parameters for better accuracy.

📌 Future Improvements
Use deep learning (CNNs) for higher accuracy.
Add data augmentation for small datasets.
Save predictions to a .csv file (image name → predicted label).
```

📊 Output:
<img width="1475" height="626" alt="Image" src="https://github.com/user-attachments/assets/104354b3-4bc0-47c3-b268-2c52cf2942ca" />
<img width="504" height="1128" alt="Image" src="https://github.com/user-attachments/assets/c82d1bc9-6273-41a8-993e-646cce55ae33" />

