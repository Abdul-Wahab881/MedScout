import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

# ---- CONFIG ----
st.set_page_config(page_title="MediScout Pakistan", layout="wide")
USERS_FILE = 'users.csv'
PATIENT_FILE = 'patients.csv'

# ---- SESSION STATE ----
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'show_form' not in st.session_state:
    st.session_state.show_form = False

# ---- USER MANAGEMENT FUNCTIONS ----
def load_users():
    if os.path.exists(USERS_FILE):
        try:
            return pd.read_csv(USERS_FILE)
        except:
            return pd.DataFrame(columns=['username', 'password'])
    else:
        return pd.DataFrame(columns=['username', 'password'])

def save_user(username, password):
    users_df = load_users()
    if username in users_df['username'].values:
        return False
    new_user = pd.DataFrame([{'username': username, 'password': password}])
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    users_df.to_csv(USERS_FILE, index=False)
    return True

def authenticate(username, password):
    users_df = load_users()
    return not users_df[(users_df['username'] == username) & (users_df['password'] == password)].empty

# ---- LOGIN / SIGNUP ----
if not st.session_state.logged_in:
    st.title("MediScout Pakistan - Login / Signup")

    users_df = load_users()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    choice = st.radio("Choose Option", ['Login', 'Signup'])

    if choice == 'Login':
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")
    else:
        if st.button("Signup"):
            if save_user(username, password):
                st.success("Signup successful! Please login.")
            else:
                st.error("Username already exists.")
    st.stop()

# ---- LOGGED IN USER SECTION ----
st.sidebar.success(f"Welcome, {st.session_state.username}")
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.experimental_rerun()

# ---- PATIENT DATA HANDLING ----
def load_patients():
    if os.path.exists(PATIENT_FILE):
        return pd.read_csv(PATIENT_FILE)
    else:
        return pd.DataFrame(columns=['name', 'age', 'gender', 'symptoms', 'disease'])

def save_patient(new_patient):
    df = load_patients()
    new_df = pd.DataFrame([new_patient])
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(PATIENT_FILE, index=False)

# ---- MAIN UI ----
st.title("🧠 MediScout Pakistan Prototype")

# Toggle Register Form
if st.button("➕ Register New Patient"):
    st.session_state.show_form = not st.session_state.show_form

# Register New Patient Form
if st.session_state.show_form:
    with st.form("patient_form"):
        st.subheader("Patient Registration Form")
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=120)
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        symptoms = st.text_input("Symptoms (comma separated)")
        disease = st.text_input("Disease")
        submit = st.form_submit_button("Add Patient")

        if submit:
            save_patient({
                'name': name,
                'age': age,
                'gender': gender,
                'symptoms': symptoms,
                'disease': disease
            })
            st.success(f"Patient {name} added successfully!")

# Delete Patient
st.subheader("🗑️ Delete Patient Record")
del_name = st.text_input("Enter name to delete")
if st.button("Delete Patient"):
    df = load_patients()
    if del_name in df['name'].values:
        df = df[df['name'] != del_name]
        df.to_csv(PATIENT_FILE, index=False)
        st.success(f"Deleted {del_name}")
    else:
        st.warning("Name not found.")

# Search Patient
st.subheader("🔍 Search Patient")
search_name = st.text_input("Search by Name")
if st.button("Search"):
    df = load_patients()
    record = df[df['name'].str.lower() == search_name.lower()]
    if not record.empty:
        st.success("Patient Found:")
        st.dataframe(record)
    else:
        st.warning("No patient found.")

# Load patient data for filtering & visualization
patients_df = load_patients()

if not patients_df.empty:
    # Add lat/lon for visualization simulation if missing
    if 'lat' not in patients_df.columns:
        patients_df['lat'] = [round(random.uniform(24.7, 25.3), 4) for _ in range(len(patients_df))]
    if 'lon' not in patients_df.columns:
        patients_df['lon'] = [round(random.uniform(67.0, 67.4), 4) for _ in range(len(patients_df))]

    # Rename disease column for consistency
    patients_df.rename(columns={'disease': 'diseases'}, inplace=True)

    st.header("📊 Filter Disease Data")

    symptoms_unique = patients_df['symptoms'].dropna().unique()
    if len(symptoms_unique) == 0:
        st.info("No symptom data available.")
    else:
        selected_symptom = st.selectbox("Select Symptom", symptoms_unique)
        age_range = st.slider("Select Age Range", 1, 120, (1, 120))
        gender_options = ['Select'] + patients_df['gender'].dropna().unique().tolist()
        selected_gender = st.selectbox("Select Gender", gender_options)

        filtered_df = patients_df[patients_df['symptoms'] == selected_symptom]
        filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
        if selected_gender != 'Select':
            filtered_df = filtered_df[filtered_df['gender'] == selected_gender]

        st.write("Filtered Data:", filtered_df)
        st.map(filtered_df[['lat', 'lon']])

        st.header("📈 Disease Counts")
        if not filtered_df.empty:
            fig, ax = plt.subplots()
            filtered_df['diseases'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel("Disease")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No data for selected filters.")
else:
    st.info("No patient data available. Please add some records.")

# ======= Image Classification Placeholder =======
st.header("🖼️ Disease Image Classification")

uploaded_file = st.file_uploader("Upload a medical image (e.g., skin lesion) for classification", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Running image classification (demo)...")

    # Here you would insert your AI model code to classify the image
    # For demo, we just show a placeholder result:
    import time
    with st.spinner('Classifying...'):
        time.sleep(2)  # Simulate processing time
    st.success("Classification Result: Benign Lesion (Demo)")
