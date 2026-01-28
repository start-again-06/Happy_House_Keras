# Happy Model – Smile Detection with Keras

A practical computer vision project that builds a Convolutional Neural Network (CNN) using Keras to detect whether a person is smiling in an image. The project covers the complete deep learning pipeline including dataset preprocessing, model construction, training, inference on custom images, and model visualization.

---

## Features

- Binary image classification: smiling vs not smiling  
- CNN built using Keras Functional API  
- End-to-end pipeline: preprocessing, training, evaluation, inference  
- Model architecture visualization using plot_model and model_to_dot  
- Support for custom image prediction  

---

## Dataset

**Facial Smile Dataset**

**Description:**  
A labeled image dataset consisting of facial images annotated as smiling or not smiling.

**Input Shape:**  
64 × 64 RGB images

**Output Labels:**  
- 0: Not Happy  
- 1: Happy  

**Usage:**  
Loaded using helper functions from kt_utils.py and normalized before training.

---

## System Design

The system is designed as a modular image classification pipeline where each stage of the CNN workflow is clearly separated. This structure improves readability, reproducibility, and ease of experimentation.

---

## High-Level Architecture

| Stage | Description |
|------|------------|
| Input Images | 64×64 RGB facial images |
| Preprocessing | Normalization and label reshaping |
| CNN Feature Extractor | Convolution, BatchNorm, Pooling |
| Classifier | Fully connected layers |
| Output Layer | Sigmoid-based binary prediction |
| Visualization | Model plots and performance metrics |

---

## Model Architecture

- **Input Layer:** (64, 64, 3)  
- Convolution + ReLU  
- Batch Normalization  
- Max Pooling  
- Flatten  
- Fully Connected (Dense)  
- **Output Layer:** Sigmoid  

**Framework:** Keras (TensorFlow backend)  
**Loss Function:** Binary Cross Entropy  
**Task:** Binary Classification  

The model is implemented using the Keras Functional API and visualized using plot_model and model_to_dot.

---

## Dataset Preprocessing

- Dataset loaded via kt_utils.py  
- Pixel values normalized to range [0, 1]  
- Labels reshaped and transposed for compatibility  
- Images converted to float32 arrays  

---

## Training Pipeline

1. Load and preprocess dataset  
2. Build CNN architecture  
3. Compile model with binary cross-entropy loss  
4. Train model on training set  
5. Evaluate accuracy on test set  
6. Visualize loss and accuracy  

---

## Inference on Custom Images

1. Place image in the images/ directory  
2. Resize image to 64×64  
3. Load and preprocess image  
4. Run model prediction  
5. Output classification: smiling or not  

---

## Results & Outputs

- Training and test accuracy  
- Binary cross-entropy loss  
- Model summary output  
- CNN architecture plot (HappyModel.png)  
- Console predictions for custom images  

---

## File Overview

- **images/**  
  Contains custom images for inference  

- **HappyModel.png**  
  Visual representation of the CNN architecture  

- **kt_utils.py**  
  Dataset loading and helper functions  

- **README.md**  
  Project documentation  

---

## Design Principles

- Clear separation of data, model, and inference logic  
- Interpretable CNN architecture  
- Reproducible preprocessing pipeline  
- Educational focus on core deep learning concepts  

---

## Dependencies

### Requirements

- Python 3.7+  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Pillow  
- pydot  
- graphviz  

## Model Visualization

- Architecture plot generated using plot_model()  
- SVG visualization using model_to_dot()  

## License

This project is intended for educational and research purposes.  
Free to use and modify with proper attribution.

