# 🫀 Heart Disease Prediction System

A machine learning project that predicts the likelihood of heart disease using various health indicators. This project implements multiple ML algorithms and provides a user-friendly web interface for predictions.

---

## 🔍 Overview
This project uses machine learning algorithms to predict the presence of heart disease based on various medical parameters. The system allows for both single patient predictions and bulk predictions from CSV files, making it a versatile tool for healthcare analysis.

## ✨ Features
* **Multiple ML Models:** Implements Logistic Regression, Decision Tree, Random Forest, and SVM.
* **Web Interface:** User-friendly dashboard built with **Streamlit**.
* **Single Prediction:** Input individual patient data for instant results.
* **Bulk Prediction:** Upload CSV files for batch processing and automated diagnosis.
* **Model Comparison:** Compare performance metrics across different algorithms.
* **Data Visualization:** Interactive charts and data analysis.

## 📊 Dataset
The project uses the Heart Disease dataset available on Kaggle.
* **Source:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
* **Attributes:** Includes age, sex, chest pain type, resting BP, cholesterol, fasting BS, etc.

## 🤖 Models Used
| Algorithm | Description |
| :--- | :--- |
| **Logistic Regression** | Used as the binary classification baseline model. |
| **Decision Tree** | Provides a clear path of logic for classification. |
| **Random Forest** | An ensemble method that improves accuracy by combining trees. |
| **SVM** | Effective for finding the optimal hyperplane in medical data. |

---

## 🚀 Installation

### Prerequisites
* Python 3.8 or higher
* pip package manager

### Setup Steps
1.**Clone the repository**
   ```bash
   git clone [https://github.com/Srijanakattel/heartdiseaseprediction.git](https://github.com/Srijanakattel/heartdiseaseprediction.git)
   cd heartdiseaseprediction
   ```
2.**Create a virtual environment**
## Windows
```
python -m venv venv
venv\Scripts\activate
```
## Mac/Linux ##
```

python3 -m venv venv
source venv/bin/activate
```
3.**Install required packages**
```
pip install -r requirements.txt
```
4.**Prepare your dataset**

Place your heart.csv file in the project root directory.

5.**Train the models**

Open and run heart.ipynb to train the models. This will generate the .pkl (pickle) files required for the web app.

💻 Usage

Running the Web Application
```
Bash

streamlit run app.py
```

The application will open in your default web browser at http://localhost:8501.

## Using the Application
Single Prediction Mode: Select a model from the sidebar, enter patient information, and click Predict.

Bulk Prediction Mode: Upload a CSV file; the system will process all records and provide a downloadable results file.

## 📈 Results
After training, model performance metrics are displayed in the Jupyter notebook, including:

Accuracy Score

Precision & Recall

F1-Score

Confusion Matrix

ROC-AUC Curve

## 🛠️ Technologies Used
Language: Python 3.x

Web Framework: Streamlit

Machine Learning: Scikit-learn

Data Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn