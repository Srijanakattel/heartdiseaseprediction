
import streamlit as st
import sqlite3
import bcrypt
from sqlalchemy import create_engine, Column, String, Integer, Table, MetaData
from sqlalchemy.exc import IntegrityError
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.graph_objects as go
import time

# --- Database Setup ---
engine = create_engine('sqlite:///users.db')
metadata = MetaData()

users_table = Table(
    'users', metadata,
    Column('id', Integer, primary_key=True),
    Column('email', String, unique=True),
    Column('password_hash', String)
)

admins_table = Table(
    'admins', metadata,
    Column('id', Integer, primary_key=True),
    Column('email', String, unique=True),
    Column('password_hash', String)
)

# This will create the tables if they don't exist
metadata.create_all(engine)

# --- Password Hashing Functions ---
def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    """Checks a password against a hashed value."""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def add_user(email, password, is_admin=False):
    """Adds a new user or admin to the database."""
    table = admins_table if is_admin else users_table
    hashed_password = hash_password(password)
    with engine.connect() as conn:
        try:
            conn.execute(table.insert().values(email=email, password_hash=hashed_password))
            conn.commit()
            return True
        except IntegrityError:
            return False

def authenticate(email, password, is_admin=False):
    """Authenticates a user or admin."""
    table = admins_table if is_admin else users_table
    with engine.connect() as conn:
        user = conn.execute(table.select().where(table.c.email == email)).fetchone()
        if user and check_password(password, user.password_hash):
            return True
    return False

def get_all_users():
    """Fetches all users from the users table."""
    with engine.connect() as conn:
        return pd.read_sql(users_table.select(), conn)

def delete_user(email):
    """Deletes a user from the users table."""
    with engine.connect() as conn:
        conn.execute(users_table.delete().where(users_table.c.email == email))
        conn.commit()

# --- Custom CSS for background and cards ---
st.markdown("""
<style>
    .login-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)
st.title("Heart Disease Prediction App")

# --- Session state for login ---
if 'user_logged_in' not in st.session_state:
    st.session_state['user_logged_in'] = False
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False
if 'show_admin_signup' not in st.session_state:
    st.session_state['show_admin_signup'] = False

# --- Main App Layout ---
if st.session_state['user_logged_in'] or st.session_state['admin_logged_in']:
    if st.session_state['user_logged_in']:
        # User Dashboard
        st.subheader("Welcome, User!")
        tabs = st.tabs(["Predict", "Bulk Predict", "Model Information", "Logout"])

        # Predict Tab
        with tabs[0]:
            st.subheader("Single Prediction")
            age = st.number_input("Age (years)", min_value=0, max_value=120)
            sex = st.selectbox("Sex", ["Male", "Female"])
            chest_pain = st.selectbox("Chest pain type", ["typical angina", "atypical angina", "non-anginal pain", "asymptomatic"])
            resting_bp = st.number_input("Resting blood pressure (mm Hg)", min_value=0, max_value=300)
            cholesterol = st.number_input("Serum cholesterol (mg/dl)", min_value=0, max_value=600)
            fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl", ["True", "False"])
            resting_ecg = st.selectbox("Resting electrocardiographic results", ["normal", "ST-T wave abnormality", "left ventricular hypertrophy"])
            max_hr = st.number_input("Maximum heart rate achieved", min_value=0, max_value=300)
            exercise_angina = st.selectbox("Exercise induced angina", ["True", "False"])
            oldpeak = st.number_input("Depression induced by exercise relative to rest", min_value=0.0, max_value=10.0)
            slope = st.selectbox("Slope of the peak exercise ST segment", ["upsloping", "flat", "downsloping"])

            sex_val = 0 if sex == "Male" else 1
            chest_pain_dict = {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}
            chest_pain_val = chest_pain_dict[chest_pain]
            fasting_bs_val = 1 if fasting_bs == "True" else 0
            resting_ecg_dict = {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
            resting_ecg_val = resting_ecg_dict[resting_ecg]
            exercise_angina_val = 1 if exercise_angina == "True" else 0
            slope_dict = {"upsloping": 0, "flat": 1, "downsloping": 2}
            slope_val = slope_dict[slope]

            input_data = pd.DataFrame({
                "Age": [age],
                "Sex": [sex_val],
                "ChestPainType": [chest_pain_val],
                "RestingBP": [resting_bp],
                "Cholesterol": [cholesterol],
                "FastingBS": [fasting_bs_val],
                "RestingECG": [resting_ecg_val],
                "MaxHR": [max_hr],
                "ExerciseAngina": [exercise_angina_val],
                "Oldpeak": [oldpeak],
                "ST_Slope": [slope_val]
            })

            def predict_logistic_regression(data):
                try:
                    model = pickle.load(open("logistic_regression_model.pkl", 'rb'))
                    if hasattr(model, 'feature_names_in_'):
                        data = data[model.feature_names_in_]
                    prediction = model.predict(data)
                    return prediction[0]
                except Exception as e:
                    st.error(f"Error with Logistic Regression model: {str(e)}. Make sure 'logistic_regression_model.pkl' is in the same folder.")
                    return -1

            if st.button("Predict", key="user_predict"):
                with st.spinner('Making prediction...'):
                    time.sleep(1)
                    result = predict_logistic_regression(input_data)
                    st.subheader("Prediction Result")
                    if result == -1:
                        st.error("Prediction failed.")
                    elif result == 0:
                        st.success("No Heart Disease Detected")
                    else:
                        st.warning("Heart Disease Detected")

        # Bulk Predict Tab
        with tabs[1]:
            st.subheader("Bulk Prediction")
            uploaded_file = st.file_uploader("Upload CSV file with patient data", type=["csv"])
            if uploaded_file is not None:
                try:
                    bulk_data = pd.read_csv(uploaded_file)
                    st.write("Preview of uploaded data:")
                    st.dataframe(bulk_data.head())
                    required_columns = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope"]
                    missing_cols = set(required_columns) - set(bulk_data.columns)
                    if missing_cols:
                        st.error(f"Your uploaded file is missing the following required columns: {', '.join(missing_cols)}.")
                    else:
                        bulk_data_required = bulk_data[required_columns].copy()
                        bulk_data_required['Sex'] = bulk_data_required['Sex'].map({'Male': 0, 'Female': 1})
                        chest_pain_map = {"typical angina": 0, "atypical angina": 1, "non-anginal pain": 2, "asymptomatic": 3}
                        bulk_data_required['ChestPainType'] = bulk_data_required['ChestPainType'].map(chest_pain_map)
                        bulk_data_required['FastingBS'] = bulk_data_required['FastingBS'].map({'True': 1, 'False': 0, 1: 1, 0: 0})
                        resting_ecg_map = {"normal": 0, "Normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
                        bulk_data_required['RestingECG'] = bulk_data_required['RestingECG'].map(resting_ecg_map)
                        bulk_data_required['ExerciseAngina'] = bulk_data_required['ExerciseAngina'].map({'True': 1, 'False': 0, 1: 1, 0: 0})
                        slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2, "Up": 0, "Flat": 1, "Down": 2}
                        bulk_data_required['ST_Slope'] = bulk_data_required['ST_Slope'].map(slope_map)

                        from sklearn.impute import SimpleImputer
                        imputer = SimpleImputer(strategy='mean')
                        bulk_data_imputed = pd.DataFrame(imputer.fit_transform(bulk_data_required), columns=required_columns)

                        if st.button("Predict Bulk Data"):
                            with st.spinner('Processing bulk predictions...'):
                                time.sleep(2)
                                results_df = bulk_data.copy()
                                try:
                                    model = pickle.load(open("logistic_regression_model.pkl", 'rb'))
                                    features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else required_columns
                                    predictions = model.predict(bulk_data_imputed[features])
                                    results_df["Prediction"] = predictions
                                except Exception as e:
                                    st.error(f"Error with Logistic Regression model: {str(e)}")
                                    results_df["Prediction"] = "Error"

                                results_df["Prediction Result"] = results_df["Prediction"].apply(lambda x: "Heart Disease" if x == 1 else ("No Heart Disease" if x == 0 else x))
                                st.subheader("Prediction Results")
                                st.dataframe(results_df)
                                csv = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(label="Download Predictions", data=csv, file_name="cardiac_predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")

        # Model Information Tab
        with tabs[2]:
            st.subheader("Model Performance")
            metrics = {
                'Model': ['Logistic Regression', 'Support Vector Machine (SVM)', 'Decision Tree Classifier', 'Random Forest Classifier'],
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
            metrics_df = pd.DataFrame(metrics)
            fig = go.Figure()
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
                fig.add_trace(go.Bar(x=metrics_df['Model'], y=metrics_df[metric], name=metric))
            fig.update_layout(barmode='group', title='Model Performance Metrics ')
            st.plotly_chart(fig)
            st.subheader('Confusion Matrices ')
            for name in metrics_df['Model']:
                st.markdown(f'**{name}**')
                cm = confusion_matrices[name]
                cm_fig = go.Figure(data=go.Heatmap(z=cm, x=['Predicted 0', 'Predicted 1'], y=['Actual 0', 'Actual 1'], colorscale='Blues', showscale=True))
                cm_fig.update_layout(title=f'Confusion Matrix: {name}')
                st.plotly_chart(cm_fig)

        # Logout Tab
        with tabs[3]:
            st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
            if st.button("Logout", key="user_logout", use_container_width=True):
                st.session_state['user_logged_in'] = False
                st.session_state['admin_logged_in'] = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state['admin_logged_in']:
        # Admin Dashboard
        st.subheader("Welcome, Admin!")
        tabs = st.tabs(["Dashboard", "Manage Users", "Logout"])

        with tabs[0]:
            st.subheader("Admin Dashboard")
            st.info("Here you can see a summary of your application usage.")
            total_users = len(get_all_users())
            st.metric(label="Total Users", value=total_users)
            st.image("https://googleusercontent.com/file_content/0")

        with tabs[1]:
            st.subheader("Manage Users")
            st.write("Here you can view and delete user accounts.")
            users_df = get_all_users()
            st.dataframe(users_df, use_container_width=True)
            st.markdown("---")
            st.markdown("### Delete a User")
            user_to_delete = st.selectbox("Select email to delete", users_df['email'])
            if st.button("Delete User", use_container_width=True):
                delete_user(user_to_delete)
                st.success(f"User '{user_to_delete}' has been deleted.")
                st.rerun()

        with tabs[2]:
            st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
            if st.button("Logout", key="admin_logout", use_container_width=True):
                st.session_state['user_logged_in'] = False
                st.session_state['admin_logged_in'] = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

else:
    # Login and Signup
    st.markdown('<div class="center-content">', unsafe_allow_html=True)
    st.header("Login or Signup")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="login-card"><h3>User Access</h3>', unsafe_allow_html=True)
        user_email = st.text_input("Email", key="user_email_login")
        user_password = st.text_input("Password", type="password", key="user_password_login")
        
        c1, c2 = st.columns(2)
        with c1:
            login_user = st.button("Login", key="login_user", help="Login as user", use_container_width=True)
        with c2:
            signup_user = st.button("Signup", key="signup_user", help="Create a new user account", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="login-card"><h3>Admin Access</h3>', unsafe_allow_html=True)
        admin_email = st.text_input("Admin Email", key="admin_email_login")
        admin_password = st.text_input("Admin Password", type="password", key="admin_password_login")
        
        st.markdown('<br>', unsafe_allow_html=True)
        login_admin = st.button("Admin Login", key="login_admin", help="Login as admin", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # User actions
    if login_user and user_email and user_password:
        if authenticate(user_email, user_password, is_admin=False):
            st.session_state['user_logged_in'] = True
            st.success("User login successful! Redirecting...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid user credentials.")

    if signup_user and user_email and user_password:
        if add_user(user_email, user_password, is_admin=False):
            st.success("Signup successful! You can now login.")
        else:
            st.error("Email already exists. Please login or use a different email.")

    # Admin actions
    if login_admin and admin_email and admin_password:
        if authenticate(admin_email, admin_password, is_admin=True):
            st.session_state['admin_logged_in'] = True
            st.success("Admin login successful! Redirecting...")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid admin credentials.")

    # Admin Signup (Hidden)
    st.markdown('<div style="text-align: center; margin-top: 50px;">', unsafe_allow_html=True)
    if st.button("Admin Signup (First Time Setup)", key="admin_first_signup"):
        st.session_state['show_admin_signup'] = True
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state['show_admin_signup']:
        st.markdown("---")
        st.subheader("Create First Admin Account")
        admin_signup_email = st.text_input("Admin Email to Create", key="admin_signup_email")
        admin_signup_password = st.text_input("Admin Password to Create", type="password", key="admin_signup_password")
        if st.button("Create Admin", key="create_admin"):
            if admin_signup_email and admin_signup_password:
                if add_user(admin_signup_email, admin_signup_password, is_admin=True):
                    st.success("Admin account created successfully! You can now log in as an admin.")
                    st.session_state['show_admin_signup'] = False
                    st.rerun()
                else:
                    st.error("Admin email already exists.")
            else:
                st.error("Please provide both email and password for the admin account.")
