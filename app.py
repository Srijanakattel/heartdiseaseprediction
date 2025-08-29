
import streamlit as st
import pandas as pd
import numpy as np
import pickle
#added for model metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.title("Heart Disease Prediction")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

with tab1:
    st.subheader("Prediction")
    
    # Input fields
    age = st.number_input("Age (years)", min_value=0, max_value=120)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest pain type", 
                            ["typical angina", "atypical angina", 
                             "non-anginal pain", "asymptomatic"])
    resting_bp = st.number_input("Resting blood pressure (mm Hg)", 
                               min_value=0, max_value=300)
    cholesterol = st.number_input("Serum cholesterol (mg/dl)", 
                                min_value=0, max_value=600)
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl", 
                            ["True", "False"])
    resting_ecg = st.selectbox("Resting electrocardiographic results", 
                             ["normal", "ST-T wave abnormality", 
                              "left ventricular hypertrophy"])
    max_hr = st.number_input("Maximum heart rate achieved", 
                           min_value=0, max_value=300)
    exercise_angina = st.selectbox("Exercise induced angina", 
                                 ["True", "False"]) 
    oldpeak = st.number_input("Depression induced by exercise relative to rest", 
                            min_value=0.0, max_value=10.0)
    slope = st.selectbox("Slope of the peak exercise ST segment", 
                       ["upsloping", "flat", "downsloping"])

    # Convert categorical inputs to numeric
    sex = 0 if sex == "Male" else 1
    chest_pain_dict = {"typical angina": 0, "atypical angina": 1, 
                      "non-anginal pain": 2, "asymptomatic": 3}
    chest_pain = chest_pain_dict[chest_pain]
    fasting_bs = 1 if fasting_bs == "True" else 0
    resting_ecg_dict = {"normal": 0, "ST-T wave abnormality": 1, 
                       "left ventricular hypertrophy": 2}
    resting_ecg = resting_ecg_dict[resting_ecg]
    exercise_angina = 1 if exercise_angina == "True" else 0
    slope_dict = {"upsloping": 0, "flat": 1, "downsloping": 2}
    slope = slope_dict[slope]

    # Create dataframe with user input - using EXACT feature names from model training
    input_data = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest_pain],
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG": [resting_ecg],
        "MaxHR": [max_hr],
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [slope]
    })

    # Only using Logistic Regression for prediction
    def predict_logistic_regression(data):
        try:
            model = pickle.load(open("logistic_regression_model.pkl", 'rb'))
            # Ensure column order matches training data
            if hasattr(model, 'feature_names_in_'):
                data = data[model.feature_names_in_]
            prediction = model.predict(data)
            return prediction[0]
        except Exception as e:
            st.error(f"Error with Logistic Regression model: {str(e)}")
            return -1  # Error code

    if st.button("Predict"):
        st.subheader("Results")
        st.markdown("--------")
        result = predict_logistic_regression(input_data)
        st.subheader("Logistic Regression")
        if result == -1:
            st.error("Prediction failed")
        elif result == 0:
            st.success("NO Heart Disease Detected")
        else:
            st.error("Heart Disease Detected")
        st.markdown("--------")
with tab2:
    st.subheader("Bulk Prediction")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            bulk_data = pd.read_csv(uploaded_file)
            
            # Display preview
            st.write("Preview of uploaded data:")
            st.dataframe(bulk_data.head())
            
            # Check required columns (modify as per your model's requirements)
            required_columns = [
                "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
                "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina",
                "Oldpeak", "ST_Slope"
            ]
            
            missing_cols = set(required_columns) - set(bulk_data.columns)
            if missing_cols:
                st.error(
                    f"Your uploaded file is missing the following required columns: {', '.join(missing_cols)}.\n"
                    "Please upload a CSV file with all these columns: "
                    f"{', '.join(required_columns)}.\n"
                    "You can download a sample template from the Model Information tab or request one."
                )
            else:
                # Only use required columns for mapping and imputation
                bulk_data_required = bulk_data[required_columns].copy()

                # Map categorical columns
                bulk_data_required['Sex'] = bulk_data_required['Sex'].map({'Male': 0, 'Female': 1})
                chest_pain_map = {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}
                bulk_data_required['ChestPainType'] = bulk_data_required['ChestPainType'].map(chest_pain_map)
                bulk_data_required['FastingBS'] = bulk_data_required['FastingBS'].map({'True': 1, 'False': 0, 1: 1, 0: 0})
                resting_ecg_map = {"normal": 0, "Normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
                bulk_data_required['RestingECG'] = bulk_data_required['RestingECG'].map(resting_ecg_map)
                bulk_data_required['ExerciseAngina'] = bulk_data_required['ExerciseAngina'].map({'True': 1, 'False': 0, 1: 1, 0: 0})
                slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2, "Up": 0, "Flat": 1, "Down": 2}
                bulk_data_required['ST_Slope'] = bulk_data_required['ST_Slope'].map(slope_map)

                # Impute missing values before prediction
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='mean')
                bulk_data_imputed = pd.DataFrame(imputer.fit_transform(bulk_data_required), columns=required_columns)

                if st.button("Predict Bulk Data"):
                    # Initialize results dataframe
                    results_df = bulk_data.copy()
                    try:
                        model = pickle.load(open("logistic_regression_model.pkl", 'rb'))
                        features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else required_columns
                        predictions = model.predict(bulk_data_imputed[features])
                        results_df["Logistic Regression"] = predictions
                    except Exception as e:
                        st.error(f"Error with Logistic Regression model: {str(e)}")
                        results_df["Logistic Regression"] = "Error"
                    # Add interpretation
                    results_df["Logistic Regression Result"] = results_df["Logistic Regression"].apply(
                        lambda x: "Heart Disease" if x == 1 else ("No Heart Disease" if x == 0 else x)
                    )
                    # Display results
                    st.subheader("Prediction Results")
                    st.dataframe(results_df)
                    # Download button
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions",
                        data=csv,
                        file_name="cardiac_predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
with tab3:
    import plotly.express as px
    import plotly.graph_objects as go
    import pandas as pd

    # Static metrics from heart.ipynb
    metrics = {
        'Model': [
            'Logistic Regression',
            'Support Vector Machine (SVM)',
            'Decision Tree Classifier',
            'Random Forest Classifier'
        ],
        'Accuracy': [0.8587, 0.8424, 0.8098, 0.8478],
        'Precision': [0.8585, 0.8544, 0.8317, 0.8426],
        'Recall': [0.8922, 0.8627, 0.8235, 0.8922],
        'F1 Score': [0.8750, 0.8585, 0.8276, 0.8667]
    }
    confusion_matrices = {
        'Logistic Regression': [[67, 15], [11, 91]],
        'Support Vector Machine (SVM)': [[67, 15], [14, 88]],
        'Decision Tree Classifier': [[65, 17], [18, 84]],
        'Random Forest Classifier': [[65, 17], [11, 91]]
    }

    # Bar chart for metrics
    metrics_df = pd.DataFrame(metrics)
    fig = go.Figure()
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
        fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df[metric], name=metric))
    fig.update_layout(barmode='group', title='Model Performance Metrics ')
    st.plotly_chart(fig)

    # Show confusion matrix for each model
    st.subheader('Confusion Matrices ')
    for name in metrics_df['Model']:
        st.markdown(f'**{name}**')
        cm = confusion_matrices[name]
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted 0', 'Predicted 1'],
            y=['Actual 0', 'Actual 1'],
            colorscale='Blues',
            showscale=True
        ))
        cm_fig.update_layout(title=f'Confusion Matrix: {name}')
        st.plotly_chart(cm_fig)      

