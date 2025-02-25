# Handwritten Digit Recognition Using CNN
## Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify handwritten digits from the Scikit-learn Digits dataset. The model is trained to recognize digits (0-9) with an impressive 98% accuracy, showcasing the power of deep learning in image classification.

## Dataset
The project utilizes the Scikit-learn Digits dataset, which consists of 1,797 grayscale images of size 8x8 pixels, each representing a digit from 0 to 9.

## Project Workflow
### Data Preprocessing

Normalized pixel values to scale between 0 and 1 for better convergence.
Reshaped images to match the CNN input format (8x8 grayscale → 8x8x1 tensor).
One-hot encoded labels for multi-class classification.
Model Architecture

#### Conv2D Layer: Extracts spatial features from images.
#### MaxPooling2D: Reduces dimensionality and improves feature selection.
#### Flatten Layer: Converts 2D feature maps into a 1D vector.
#### Dense Layers: Fully connected layers for classification.
#### Softmax Output: Predicts probability distribution across 10 classes.
### Model Training

Optimizer: Adam (adaptive learning rate).
Loss Function: Categorical Cross-Entropy.
Metrics: Accuracy.
Performance & Results

Achieved 98% accuracy on the test dataset.
The model effectively generalizes to unseen handwritten digits.

## Technologies Used
Python
TensorFlow & Keras
NumPy
Scikit-learn
## Future Improvements
Experimenting with deeper CNN architectures for better accuracy.
Implementing data augmentation to enhance model generalization.
Deploying the model using Flask or Streamlit for real-time predictions.
