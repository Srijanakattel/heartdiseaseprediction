import streamlit as st
import bcrypt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sqlalchemy import create_engine, Column, String, Integer, Table, MetaData
from sqlalchemy.exc import IntegrityError
import os
import time
import base64
import pickle

# --- Page Configuration (Set this at the top) ---
st.set_page_config(
    page_title="Heart Disease Prediction",
    layout="centered" # Use 'centered' layout for better login screen alignment
)

# --- Database Setup ---
engine = create_engine('sqlite:///users.db')
metadata = MetaData()
users_table = Table('users', metadata, Column('id', Integer, primary_key=True), Column('email', String, unique=True), Column('password_hash', String))
admins_table = Table('admins', metadata, Column('id', Integer, primary_key=True), Column('email', String, unique=True), Column('password_hash', String))
metadata.create_all(engine)

# --- SHARED FUNCTIONS ---
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

def get_base64_image(image_path):
    """Encodes an image to a base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- CSS and Background Styling ---
image_path = "stethoscope.jpg"
if os.path.exists(image_path):
    encoded_image = get_base64_image(image_path)
    background_image_css = f"""
    background-image: url("data:image/jpg;base64,{encoded_image}");
    background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed;
    """
else:
    st.warning(f"Background image '{image_path}' not found. Using solid color background.")
    background_image_css = "background-color: #ADD8E6;" # Fallback color

# The .login-card CSS is no longer needed as we will use st.container(border=True)
st.markdown(f"""
<style>
    .stApp {{
        {background_image_css}
    }}
    /* Center the title in the login container */
    .login-title {{
        text-align: center;
    }}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'user_logged_in' not in st.session_state:
    st.session_state['user_logged_in'] = False
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False

# =====================================================================================
# --- PAGE DEFINITIONS ---
# =====================================================================================

def show_user_page():
    """ Renders the main user login page or the user dashboard. """
    if st.session_state.get('user_logged_in'):
        # --- USER DASHBOARD ---
        st.title("Heart Disease Prediction App ❤️")
        st.subheader("Welcome, User!")
        tabs = st.tabs(["Predict", "Bulk Predict", "Model Information", "Logout"])
        
        with tabs[0]:
            st.subheader("Single Prediction")
            # ... (Your prediction form code here)
        with tabs[1]:
            st.subheader("Bulk Prediction")
            # ... (Your bulk prediction code here)
        with tabs[2]:
            st.subheader("Model Information")
            # ... (Your model info code here)
        with tabs[3]:
            st.write("Click the button below to log out.")
            if st.button("Logout", key="user_logout", use_container_width=True):
                st.session_state['user_logged_in'] = False
                st.query_params.clear()
                st.rerun()
    else:
        # --- USER LOGIN FORM (Corrected Implementation) ---
        # Use st.container with a border to create the card effect.
        with st.container(border=True):
            st.markdown('<h3 class="login-title">User Access</h3>', unsafe_allow_html=True)
            
            user_email = st.text_input("Email", key="user_email_login")
            user_password = st.text_input("Password", type="password", key="user_password_login")
            
            st.write("") # Spacer
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Login", key="login_user", use_container_width=True):
                    if authenticate(user_email, user_password, is_admin=False):
                        st.session_state['user_logged_in'] = True
                        st.rerun()
                    else:
                        st.error("Invalid credentials.")
            with c2:
                if st.button("Signup", key="signup_user", use_container_width=True):
                    if user_email and user_password:
                        if add_user(user_email, user_password, is_admin=False):
                            st.success("Signup successful! You can now login.")
                        else:
                            st.error("Email already exists.")
                    else:
                        st.warning("Please enter email and password to sign up.")
        
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; font-weight: bold;">
            <a href="?page=admin" target="_self" style="color: black; text-decoration: none;">Admin Login</a>
        </div>
        """, unsafe_allow_html=True)

def show_admin_page():
    """ Renders the admin login page or the admin dashboard. """
    if st.session_state.get('admin_logged_in'):
        # --- ADMIN DASHBOARD ---
        st.title("Admin Panel 🔐")
        st.subheader("Welcome, Admin!")
        tabs = st.tabs(["Dashboard", "Manage Users", "Logout"])
        
        with tabs[0]:
            st.subheader("Admin Dashboard")
            total_users = len(get_all_users())
            st.metric(label="Total Registered Users", value=total_users)
            # You can add more dashboard components here

        with tabs[1]:
            st.subheader("Manage Users")
            users_df = get_all_users()
            if not users_df.empty:
                st.dataframe(users_df[['id', 'email']], use_container_width=True)
                user_to_delete = st.selectbox("Select email to delete", users_df['email'])
                if st.button("Delete User", type="primary", use_container_width=True):
                    delete_user(user_to_delete)
                    st.success(f"User '{user_to_delete}' has been deleted.")
                    st.rerun()
            else:
                st.info("No users found.")

        with tabs[2]:
            st.write("Click the button below to log out.")
            if st.button("Logout", key="admin_logout", use_container_width=True):
                st.session_state['admin_logged_in'] = False
                st.query_params.clear()
                st.rerun()
    else:
        # --- ADMIN LOGIN FORM (Corrected Implementation) ---
        with st.container(border=True):
            st.markdown('<h3 class="login-title">Admin Access</h3>', unsafe_allow_html=True)
            
            admin_email = st.text_input("Admin Email", key="admin_email_login")
            admin_password = st.text_input("Admin Password", type="password", key="admin_password_login")
            
            st.write("") # Spacer

            if st.button("Admin Login", key="login_admin", use_container_width=True):
                if authenticate(admin_email, admin_password, is_admin=True):
                    st.session_state['admin_logged_in'] = True
                    st.rerun()
                else:
                    st.error("Invalid admin credentials.")
        
        st.markdown("""
        <div style="text-align: center; margin-top: 20px; font-weight: bold;">
            <a href="/" target="_self" style="color: black; text-decoration: none;">Back to User Login</a>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("First Time Admin Setup"):
            st.write("Only use this if no admin account exists.")
            admin_signup_email = st.text_input("New Admin Email", key="admin_signup_email")
            admin_signup_password = st.text_input("New Admin Password", type="password", key="admin_signup_password")
            if st.button("Create Admin Account"):
                if add_user(admin_signup_email, admin_signup_password, is_admin=True):
                    st.success("Admin account created! You can now log in.")
                else:
                    st.error("This admin email may already exist.")

# =====================================================================================
# --- URL ROUTER ---
# This is the main logic that decides which page to show based on the URL.
# =====================================================================================

# Use st.query_params for robust parameter handling
query_params = st.query_params
if "page" in query_params and query_params["page"] == "admin":
    show_admin_page()
else:
    show_user_page()
