# 🌿 Cassava Leaf Disease Detection using Deep Learning

An end-to-end deep learning system for **automatic cassava leaf disease classification** using **EfficientNetB0 and transfer learning**, trained on real-world field images and deployed via a **Streamlit web application**.

---

## 📌 Project Overview

Cassava is a staple crop for millions of people worldwide, particularly in developing regions. Diseases such as **Cassava Mosaic Disease (CMD)** and **Cassava Brown Streak Disease (CBSD)** can cause severe yield losses and threaten food security. Early and accurate disease detection is therefore critical.

This project presents a **computer vision–based disease classification system** that automatically identifies cassava leaf diseases from images using a convolutional neural network. The system includes **data preprocessing, model training, evaluation, error analysis, and deployment** as an interactive web application.

---

## 🎯 Objectives

- Develop a robust deep learning model for cassava leaf disease classification  
- Handle real-world image challenges such as lighting variation and background noise  
- Address class imbalance using class weighting  
- Evaluate performance using Accuracy, Precision, Recall, and F1-score  
- Analyse misclassified samples and model limitations  
- Deploy the trained model as a user-friendly Streamlit application  

---

## 🦠 Disease Classes

The model classifies cassava leaf images into **five categories**:

1. Cassava Bacterial Blight (CBB)  
2. Cassava Brown Streak Disease (CBSD)  
3. Cassava Green Mottle (CGM)  
4. Cassava Mosaic Disease (CMD)  
5. Healthy  

---

## 📂 Dataset

- **Source:** Kaggle – Cassava Leaf Disease Classification  
- **Images:** 21,000+ real-world field images  
- **Characteristics:**
  - Natural lighting and shadows  
  - Diverse backgrounds  
  - Class imbalance  

🔗 Dataset link:  
https://www.kaggle.com/competitions/cassava-leaf-disease-classification

---

## 🧠 Model Architecture

- **Backbone:** EfficientNetB0 (pretrained on ImageNet)  
- **Technique:** Transfer Learning  
- **Key components:**
  - Image resizing to 224×224  
  - Data augmentation  
  - Class weighting  
  - Global Average Pooling  
  - Dropout regularization  
  - Softmax output layer (5 classes)  

---

## ⚙️ Technologies Used

- **Language:** Python 3.10+  
- **Deep Learning:** TensorFlow / Keras  
- **Data Processing:** NumPy, Pandas  
- **Visualization:** Matplotlib  
- **Deployment:** Streamlit  
- **IDE:** PyCharm  

---

## 📁 Project Structure

cassava-leaf-disease-detection/
│
├── app/
│ └── streamlit_app.py # Streamlit deployment app
│
├── data/
│ ├── train.csv
│ ├── train_images/
│ └── label_num_to_disease_map.json
│
├── src/
│ ├── check_data.py
│ ├── dataset.py
│ ├── model.py
│ ├── train.py
│ ├── evaluate.py
│ ├── error_analysis.py
│ ├── generate_curves.py
│ └── test_*.py
│
├── outputs/
│ ├── best.keras
│ ├── confusion_matrix.png
│ ├── misclassified_examples.png
│ ├── error_summary_by_class.csv
│ └── training_curves.png
│
├── requirements.txt
├── README.md
└── LICENSE

yaml
Copy code

---

## 🚀 Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/cassava-leaf-disease-detection.git
cd cassava-leaf-disease-detection

```
2️⃣ Create a virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate      # Linux / Mac
.venv\Scripts\activate         # Windows

---
🏋️ Model Training
Ensure the dataset is placed inside the data/ directory, then run:
```bash
python src/train.py
```
The trained model will be saved as:
```bash
outputs/best.keras
```
📊 Model Evaluation

To evaluate the trained model:
```bash
python src/evaluate.py
```
This generates:

- Classification report
- Confusion matrix
- Accuracy and F1-scores

---

🔍 Error Analysis

To analyse misclassified samples and minority class errors:
```
Generated outputs:

- misclassified_examples.png
- misclassified_samples.csv
- error_summary_by_class.csv

---
🌐 Deployment (Streamlit Application)

Run the Streamlit app using:

```bash
streamlit run app/streamlit_app.py

```
## Application Features:

- Upload cassava leaf image (JPG/PNG)
- Predict disease class
- Display confidence score
- Show Top-3 class probabilities
- Simple and intuitive user interface

---
## 📈 Results Summary

Overall Accuracy: ~81%

CMD F1-score: 0.92

Strong performance on real-world field images

Errors mainly occur for minority classes and visually similar symptoms

---
## ⚠️ Limitations

Class imbalance affects minority class performance

Visual similarity between certain disease symptoms

CPU-only deployment leads to slower inference

