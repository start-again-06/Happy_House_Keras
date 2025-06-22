# 😊 Happy Model – Smile Detection using Deep Learning
This project builds a Convolutional Neural Network (CNN) using Keras to detect smiles in facial images. It applies deep learning to a binary classification task: identifying whether a person in an image is smiling.

📌 Objective
Build and train a CNN to classify images as “happy” (smiling) or “not happy.”

Preprocess image data for training and evaluation.

Visualize performance and architecture.

Test the model on custom images.

🧠 Key Features
End-to-end image classification pipeline

Keras Functional API for flexible architecture design

Batch Normalization, Pooling, and Activation layers

Model visualization tools (architecture plots, prediction visualization)

Works with custom images via pre-trained model inference

📂 Dataset
Facial images labeled as happy or not happy.

Images are resized and normalized for model input.

Dataset split into training and test sets.

🔧 Preprocessing Steps
Normalize pixel values (0–255 → 0–1).

Convert labels into one-hot vectors.

Resize custom images to match input dimensions.

Expand dimensions for batch compatibility.

🧪 Model Overview
Input: RGB images (64x64x3)

Layers:

Convolutional layers

Batch Normalization

MaxPooling

Dropout (optional)

Fully Connected (Dense) layer

Sigmoid output for binary classification

Output: 0 (not happy) or 1 (happy)

📈 Evaluation Metrics
Binary Cross-Entropy Loss

Accuracy on test set

Prediction on new images

Visualization of:

Accuracy & loss over epochs

Model architecture (model.summary() and visual plots)

📸 Prediction on Custom Images
Upload an image (.jpg, .png, etc.)

Resize to required dimensions (64x64)

Preprocess image as model input

Model outputs prediction with probability

📊 Visualization Tools
Architecture plots using plot_model

Decision boundary plots

Sample predictions with input image display

🧾 Requirements
Key libraries used:

Keras

TensorFlow

NumPy

Matplotlib

Pillow

A full requirements.txt file can be generated for environment setup.
