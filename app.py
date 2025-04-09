import streamlit as st
import cv2
import numpy as np
import joblib
import os
from datetime import datetime
from skimage.feature import hog
from skimage.color import rgb2gray

# Define label mapping
label_map = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

# Load model and scaler based on selection
@st.cache_resource
def load_model(classifier_choice):
    model_path = os.path.join("model", f"{classifier_choice}_model.pkl")
    scaler_path = os.path.join("model", r"model/scaler.pkl")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

# Feature extraction function
def extract_features(img):
    img = cv2.resize(img, (128, 128))
    gray = rgb2gray(img)
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
    hist_h /= hist_h.sum() if hist_h.sum() != 0 else 1
    hist_s /= hist_s.sum() if hist_s.sum() != 0 else 1
    hist_v /= hist_v.sum() if hist_v.sum() != 0 else 1
    return np.hstack([hog_features, hist_h, hist_s, hist_v])

# Object detection and classification
def detect_and_classify_objects(img, model, scaler):
    output_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 30 or h < 30:
            continue
        roi = img[y:y+h, x:x+w]
        features = extract_features(roi)
        features_scaled = scaler.transform([features])
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features_scaled)[0]
            pred = np.argmax(probs)
            confidence = probs[pred]
        else:
            pred = model.predict(features_scaled)[0]
            confidence = 1.0
        label = f"{label_map[pred]} ({confidence*100:.1f}%)"
        cv2.rectangle(output_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(output_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return output_img

# Save output image to /output
def save_image(img, name_prefix="result"):
    os.makedirs("output", exist_ok=True)
    filename = f"{name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    path = os.path.join("output", filename)
    cv2.imwrite(path, img)
    return path

# ================= Streamlit App =================
st.set_page_config(page_title="Garbage Classifier", layout="wide")
st.title("🗑️ Garbage Detection & Classification App (Traditional CV)")

classifier_choice = st.selectbox("Choose Classifier", ["rf", "svm"], format_func=lambda x: "Random Forest" if x == "rf" else "SVM")
model, scaler = load_model(classifier_choice)

tab1, tab2 = st.tabs(["📷 Single Image", "🖼️ Multiple Images"])

# --- Single Image Upload ---
with tab1:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)
        result = detect_and_classify_objects(img, model, scaler)
        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Detected & Classified", use_column_width=True)
        if st.button("Save Result", key="save_single"):
            path = save_image(result, name_prefix="single")
            st.success(f"Saved to: `{path}`")

# --- Multiple Image Upload ---
with tab2:
    multi_upload = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="multi")
    if multi_upload:
        for file in multi_upload:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            result = detect_and_classify_objects(img, model, scaler)
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption=f"Processed: {file.name}", use_column_width=True)
            saved_path = save_image(result, name_prefix=os.path.splitext(file.name)[0])
            st.caption(f"Saved to: `{saved_path}`")
