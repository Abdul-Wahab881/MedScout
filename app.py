import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

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
    st.sidebar.title("Login / Signup")
    choice = st.sidebar.radio("Choose Option", ['Login', 'Signup'])

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if choice == 'Login':
        if st.sidebar.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.experimental_rerun()
            else:
                st.sidebar.error("Invalid username or password.")
    else:
        if st.sidebar.button("Signup"):
            if save_user(username, password):
                st.sidebar.success("Signup successful. Please login.")
            else:
                st.sidebar.error("Username already exists.")
    st.stop()

# ---- LOGGED IN USER ----
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

# ---- IMAGE CLASSIFICATION SETUP ----
@st.cache_resource(show_spinner=False)
def load_model():
    model = resnet50(pretrained=True)
    model.eval()
    return model

model = load_model()

@st.cache_data(show_spinner=False)
def load_imagenet_classes():
    with open("imagenet_classes.txt", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

imagenet_classes = load_imagenet_classes()

def classify_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # batch of 1
    with torch.no_grad():
        outputs = model(input_tensor)
    _, predicted = outputs.max(1)
    class_id = predicted.item()
    return imagenet_classes[class_id]

# ---- MAIN UI ----
st.title("🧠 MediScout Pakistan Prototype")

# ---- Image Upload and Classification ----
st.header("Image Classification (Using ImageNet pre-trained ResNet50)")
uploaded_file = st.file_uploader("Upload an image for classification", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            label = classify_image(image)
        st.success(f"Predicted Class: **{label}**")

# ---- Toggle Register Form ----
if st.button("➕ Register New Patient"):
    st.session_state.show_form = not st.session_state.show_form

# ---- Register New Patient ----
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

# ---- DELETE PATIENT ----
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

# ---- SEARCH PATIENT ----
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

# ---- PATIENT DATA FOR VISUALIZATION ----
patients_df = load_patients()

if not patients_df.empty:
    if 'lat' not in patients_df.columns:
        patients_df['lat'] = [round(random.uniform(24.7, 25.3), 4) for _ in range(len(patients_df))]
    if 'lon' not in patients_df.columns:
        patients_df['lon'] = [round(random.uniform(67.0, 67.4), 4) for _ in range(len(patients_df))]
    if 'disease_flag' not in patients_df.columns:
        patients_df['disease_flag'] = patients_df['disease'].apply(lambda x: 1 if x else 0)
    patients_df.rename(columns={'disease': 'diseases'}, inplace=True)
    df_demo = patients_df.copy()
else:
    st.warning("No patient data found. Please add some records.")
    df_demo = pd.DataFrame()

# ---- FILTERS ----
if not df_demo.empty:
    st.header("📊 Filter Disease Data")
    symptom_filter = st.selectbox("Select Symptom", df_demo['symptoms'].unique())
    age_range = st.slider("Age Range", 1, 120, (1, 120))
    gender_filter = st.selectbox("Select Gender", ['Select'] + list(df_demo['gender'].unique()))

    filtered = df_demo[df_demo['symptoms'] == symptom_filter]
    filtered = filtered[(filtered['age'] >= age_range[0]) & (filtered['age'] <= age_range[1])]
    if gender_filter != 'Select':
        filtered = filtered[filtered['gender'] == gender_filter]

    st.write("Filtered Data", filtered)
    st.map(filtered[['lat', 'lon']])

    st.header("📈 Disease Counts")
    if not filtered.empty:
        fig, ax = plt.subplots()
        filtered['diseases'].value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel("Disease")
        ax.set_ylabel("Count")
        st.pyplot(fig)
else:
    st.info("No data available to filter.")
