# 🦠 COVID-19 X-ray Image Classifier

A Streamlit web app that uses a deep learning model to classify chest X-ray images as **Covid**, **Normal**, or **Pneumonia**.

## Features
- Upload chest X-ray images (JPG, JPEG, PNG)
- Predicts the class using a trained Keras model
- User-friendly interface with instant results


## Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd Covid-Classification
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Or manually:
   ```bash
   pip install streamlit tensorflow pillow opencv-python
   ```
3. Place your trained model file as `covid_19_model.h5` in the project directory.

### Running the App
```bash
streamlit run app.py
```

## Usage
1. Open the app in your browser (Streamlit will provide a local URL).
2. Upload a chest X-ray image.
3. Click **Predict** to see the classification result.

## Model Details
- Model: Custom CNN (Keras)
- Input size: 224x224x3
- Classes: Covid, Normal, Pneumonia

## License
This project is for educational purposes.

---
Developed by Baina magdy 
