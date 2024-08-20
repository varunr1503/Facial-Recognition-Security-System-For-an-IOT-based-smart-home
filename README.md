# Facial Recognition Security System for an IoT-based Smart Home

## Overview

This project implements a facial recognition security system integrated into an IoT-based smart home. The system enhances security by allowing only recognized individuals to access the home. It is designed to work with an ESP8266 microcontroller, leveraging the power of deep learning and IoT technologies.

## Features

- **Face Dataset Collection**: Collects facial images using OpenCV's Haar Cascade classifier.
- **Face Training**: Uses Local Binary Patterns Histograms (LBPH) for facial recognition.
- **Face Recognition**: Recognizes faces in real-time, allowing for secure access to the smart home.
- **IoT Integration**: Integrated with an ESP8266 module for controlling smart home devices based on recognized faces.
- **Deep Learning**: Utilizes the InceptionResnetV1 model pre-trained on the VGGFace2 dataset for improved facial recognition accuracy.

## Initial Implementation

The project initially involved three main programs:

1. **Face Dataset Collection**:

   - Collected images using OpenCV's `cv2.CascadeClassifier('haarcascade_frontalface_default.xml')`.

2. **Face Training**:

   - Trained a facial recognition model using LBPH.
   - Used the Haar Cascade classifier for detecting faces.

3. **Face Recognition**:
   - Loaded the trained model using LBPH .
   - Used the trained model for real-time face recognition.

## Improved Implementation

The system was improved by integrating it with an IoT-based smart home setup and replacing the facial recognition model with a more robust deep learning model:

- **Model**: The facial recognition model was upgraded to `InceptionResnetV1`, a deep learning model that offers higher accuracy and robustness.
- **IoT Integration**: The system was integrated with the ESP8266 microcontroller to interact with the smart home environment.
