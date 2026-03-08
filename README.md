🧠 Brain Tumour Classification using Quantum-AI Hybrid Model

A research-oriented project focused on brain tumour classification from MRI images using Deep Learning models with the potential integration of Quantum Machine Learning (QML) techniques.

The project explores how classical deep learning models perform on medical image datasets while investigating how quantum-enhanced models may improve accuracy and computational efficiency.

An interactive web application is also included to demonstrate real-time MRI image classification.

Project Highlights

Implementation of multiple Deep Learning models for MRI classification

Exploration of Quantum Machine Learning concepts

Hybrid AI–Quantum architecture experimentation

Visualization of model comparison graphs

Interactive Streamlit-based web interface

Designed for research paper experimentation

Technologies Used

->Programming-Python
->Deep Learning	-TensorFlow / Keras
->Quantum Computing-	Qiskit
->Data Processing-	NumPy, Pandas
->Image Processing-	OpenCV, Pillow
->Visualization-	Matplotlib
->Interface	-Streamlit

Project Workflow-
Load and preprocess MRI brain scan dataset

Train deep learning models (CNN, ResNet)

Implement hybrid Quantum-AI model

Evaluate models using accuracy metrics

Generate performance comparison graphs

Visualize predictions in a Streamlit interface

Models Implemented
Classical Deep Learning Models

Convolutional Neural Network (CNN)

ResNet Model

Custom Deep Learning Architecture

Quantum Machine Learning

Hybrid Quantum-Classical Model

Quantum Feature Encoding

Variational Quantum Layer Integration

Brain Tumour Classification Categories

The model classifies MRI images into four categories:

Glioma Tumour

Meningioma Tumour

Pituitary Tumour

No Tumour


Quantum Circuit Visualization

The quantum layer encodes classical image features into quantum states and processes them through a parameterized quantum circuit.

Outputs include:

Quantum circuit diagrams

Measurement results

Hybrid model predictions

Running the Project
Clone the Repository
git clone https://github.com/Riya-877/Brain_tumour_ml_quantum.git
cd Brain_tumour_ml_quantum
Install Dependencies
pip install -r requirements.txt
Run the Streamlit Application
streamlit run app.py

The application will open in your browser where users can upload MRI images and get tumour predictions in real time.

Project Structure
Brain_tumour_ml_quantum
│
├── data
│   └── MRI dataset
│
├── models
│   └── trained models
│
├── notebooks
│   └── experimentation notebooks
│
├── quantum
│   └── quantum layers and circuits
│
├── graphs.py
├── train_ml_models.py
├── train_resnet.py
├── train_hybrid.py
├── evaluate_models.py
│
├── app.py
├── requirements.txt
└── README.md
Research Motivation

Brain tumour detection from MRI scans is critical for early diagnosis and treatment planning. Deep learning models have shown promising results in medical image analysis.

This project explores how Quantum Machine Learning techniques can be integrated with classical models to potentially improve:

classification performance

computational efficiency

hybrid AI architectures for healthcare

Future Improvements

Add Explainable AI (Grad-CAM heatmaps)

Run quantum experiments on real quantum hardware

Add additional evaluation metrics (F1 score, ROC curves)

Optimize hybrid quantum circuits

Deploy the application on cloud platforms

Author

Riya Agarwal

GitHub:
https://github.com/Riya-877

⭐ If you like this project, consider starring the repository!
