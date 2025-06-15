# Skin_cancer_prediction_webapp
This project is a web application for detecting skin cancer from images of skin lesions using a Convolutional Neural Network (CNN) built with TensorFlow and deployed via Streamlit.

## Features
Binary classification: Cancer (Malignant) vs No Cancer (Benign)

User-friendly web interface to upload images and get predictions instantly

Image preprocessing and normalization consistent with training

Confidence score display for model predictions

Lightweight CNN architecture optimized for skin lesion images

Model caching for faster inference in Streamlit

Simple and clean UI for easy use

## Project Structure
app.py — Streamlit app source code

skin_cance_cnn.h5 — Trained CNN model (Keras HDF5 format)

requirements.txt — Python dependencies needed for the app

Dataset (not included in repo) used for training/testing: Melanoma Cancer Dataset

## Usage
Upload a skin lesion image (jpg, jpeg, or png)

Click Predict to classify the image as cancerous or benign

View prediction label and confidence score instantly

## Acknowledgements
Dataset source: Kaggle Melanoma Skin Cancer Dataset

TensorFlow and Keras for powerful deep learning framework

Streamlit for effortless web app deployment


