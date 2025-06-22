# 😊 Happy Model – Smile Detection with Keras

This project builds a convolutional neural network (CNN) using Keras to detect whether a person is smiling in an image. It includes dataset handling, model construction, training, inference, and visualization.

---

## 📌 Overview

- **Framework**: Keras (with TensorFlow backend)
- **Task**: Binary classification – smiling vs not smiling
- **Input Shape**: 64x64 RGB images
- **Output**: Binary label (0: Not Happy, 1: Happy)

---

## 🧠 Model Architecture

- **Input Layer**: (64, 64, 3)
- **Convolution + ReLU**
- **Batch Normalization**
- **MaxPooling**
- **Flatten**
- **Fully Connected (Dense)**
- **Sigmoid Output**

Model is created using Keras Functional API and visualized with `model_to_dot` and `plot_model`.

---

## 🗂️ Files & Folders

├── images/
│ └── my_image.jpg # Your custom input image
├── HappyModel.png # Visual representation of model architecture
├── kt_utils.py # Dataset loader and helper functions
├── README.md # Project overview

yaml
Copy
Edit

---

## 📊 Dataset Preprocessing

- **Dataset**: Loaded via `kt_utils.py`
- **Normalization**: `X_train` and `X_test` scaled to [0, 1]
- **Shape Adjustments**:
  - Transpose labels: `Y_train`, `Y_test`
  - Original: (m, 64, 64, 3) → Normalized float32 arrays

---

## ✅ Training Metrics

- **Loss**: Binary Cross-Entropy
- **Accuracy**: % of correctly classified smiles
- **Performance Output**:
Loss = 0.XXXX
Test Accuracy = 0.XX

yaml
Copy
Edit

---

## 🖼️ Inference on Custom Image

1. Place your image in the `images/` directory.
2. Resize to 64x64 if not already.
3. Run the prediction pipeline:
 - Load image
 - Preprocess using `image.img_to_array` and `preprocess_input`
 - Model outputs prediction: smiling or not

---

## 🧾 Dependencies

Install all required libraries using:

```bash
pip install -r requirements.txt
Main libraries used:

tensorflow / keras

numpy

matplotlib

Pillow

pydot and graphviz (for model visualization)

📈 Outputs
Accuracy & loss on test dataset

Model architecture plot (HappyModel.png)

Console outputs for custom predictions

Summary of the model (.summary())

🖥️ Model Visualization
Plot saved using plot_model()

SVG model display using model_to_dot()
