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

Live related-research search for each prediction using SerpApi (Google Scholar)

Designed for research paper experimentation

Technologies Used

->Programming-Python
->Deep Learning	-TensorFlow / Keras
->Quantum Computing-	Qiskit
->Data Processing-	NumPy, Pandas
->Image Processing-	OpenCV, Pillow
->Visualization-	Matplotlib
->Interface	-Streamlit
->Research Search	-SerpApi (Google Scholar API)

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

Related Research Search (SerpApi)

After every prediction, the app uses the [SerpApi](https://serpapi.com) Google Scholar API to fetch recent (2022+) research papers about the predicted tumour type, e.g. glioma MRI deep learning classification. Each result shows the title, authors, venue, snippet, citation count, and a PDF link when one is available.

Results are cached for 24 hours per query, so re-uploading images does not use up SerpApi search credits. The code is in `serpapi_research.py`.

Running the Project

1. Clone the repository

```bash
git clone https://github.com/Riya-877/Brain_tumour_ml_quantum.git
cd Brain_tumour_ml_quantum
```

2. Create a virtual environment (Python 3.11) and install dependencies

```bash
python3.11 -m venv brainenv311
source brainenv311/bin/activate
pip install -r requirements.txt
```

3. Add the dataset and trained model

The `data/` and `models/` folders are not committed. Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), put `Training/` in `data/train` and `Testing/` in `data/test`, then run `train_cnn_model.ipynb` to create `models/brain_tumor_model.h5`.

4. Add your SerpApi key

Get a free key from https://serpapi.com/manage-api-key and create `.streamlit/secrets.toml` (already gitignored):

```toml
SERPAPI_API_KEY = "your_serpapi_key"
```

You can also set it as an environment variable: `export SERPAPI_API_KEY=your_serpapi_key`. Without a key, the app still works but skips the research section.

5. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser where users can upload MRI images and get tumour predictions in real time, with Grad-CAM explainability, quantum validation, and related research papers.

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
├── serpapi_research.py
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
