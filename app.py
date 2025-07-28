
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Cardiac Disease Prediction")
tab1, tab2, tab3 = st.tabs(["Predict", "Bulk Predict", "Model Information"])

with tab1:
    st.subheader("Single Prediction")
    
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

    algonames = ["Logistic Regression", "Support Vector Machine (SVM)", 
                "Decision Tree Classifier", "Random Forest Classifier"]
    modelnames = ["logistic_regression_model.pkl", "svm_model.pkl", 
                 "decision_tree_model.pkl", "random_forest.pkl"]

    def predict_heart_disease(data):
        predictions = []
        for modelname in modelnames:
            try:
                model = pickle.load(open(modelname, 'rb'))
                # Ensure column order matches training data
                if hasattr(model, 'feature_names_in_'):
                    data = data[model.feature_names_in_]
                prediction = model.predict(data)
                predictions.append(prediction[0])
            except Exception as e:
                st.error(f"Error with {modelname}: {str(e)}")
                predictions.append(-1)  # Error code
        return predictions

    if st.button("Predict"):
        st.subheader("Results")
        st.markdown("--------")
        
        results = predict_heart_disease(input_data)
        
        for i in range(len(results)):
            st.subheader(algonames[i])
            if results[i] == -1:
                st.error("Prediction failed")
            elif results[i] == 0:
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
                st.error(f"Missing required columns: {missing_cols}")
            else:
                # Convert categorical columns (similar to single prediction)
                if 'Sex' in bulk_data.columns:
                    bulk_data['Sex'] = bulk_data['Sex'].map({'Male': 0, 'Female': 1})
                
                if 'ChestPainType' in bulk_data.columns:
                    chest_pain_map = {"typical angina": 0, "atypical angina": 1, 
                                     "non-anginal pain": 2, "asymptomatic": 3}
                    bulk_data['ChestPainType'] = bulk_data['ChestPainType'].map(chest_pain_map)
                
                # Add similar mappings for other categorical columns...
                
                if st.button("Predict Bulk Data"):
                    # Initialize results dataframe
                    results_df = bulk_data.copy()
                    
                    # Load models and make predictions
                    for i, modelname in enumerate(modelnames):
                        try:
                            model = pickle.load(open(modelname, 'rb'))
                            # Ensure column order matches training data
                            if hasattr(model, 'feature_names_in_'):
                                features = model.feature_names_in_
                            else:
                                features = required_columns
                            
                            predictions = model.predict(bulk_data[features])
                            results_df[algonames[i]] = predictions
                            
                        except Exception as e:
                            st.error(f"Error with {modelname}: {str(e)}")
                            results_df[algonames[i]] = "Error"
                    
                    # Add interpretation
                    for algo in algonames:
                        if algo in results_df.columns:
                            results_df[f"{algo}_Result"] = results_df[algo].apply(
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
    data = {
        "Decision Trees": 0.85,  # Added quotes
        "Random Forest Classifier": 0.90,  # Added quotes
        "Support Vector Machine": 0.88,  # Added quotes
        "Logistic Regression": 0.82  # Added quotes
    }
    models = list(data.keys())
    Accuracies = list(data.values())
    df = pd.DataFrame(list(zip(models, Accuracies)), columns=['Models', 'Accuracies'])
    fig = px.bar(df, x='Models', y='Accuracies', title='Model Accuracies')
    st.plotly_chart(fig)
    
    