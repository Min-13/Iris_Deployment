import os, sys, hashlib, traceback
MODEL_PATH = "rf_model.sav"

def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

print("DEBUG: cwd", os.getcwd(), file=sys.stderr)
print("DEBUG: abs model path", os.path.abspath(MODEL_PATH), file=sys.stderr)
print("DEBUG: exists", os.path.exists(MODEL_PATH), file=sys.stderr)
if os.path.exists(MODEL_PATH):
    print("DEBUG: size (bytes)", os.path.getsize(MODEL_PATH), file=sys.stderr)
    try:
        print("DEBUG: sha256", sha256_of_file(MODEL_PATH), file=sys.stderr)
    except Exception as e:
        print("DEBUG: checksum read error", e, file=sys.stderr)
import sklearn
print("DEBUG: python", sys.version.splitlines()[0], "sklearn", sklearn.__version__, file=sys.stderr)

import os, sys, traceback
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Diagnostics helpers -------------------------------------------
def read_sklearn_version_from_file(path="rf_model.sav"):
    try:
        with open(path, "rb") as f:
            b = f.read(2000)
        idx = b.find(b"_sklearn_version")
        if idx == -1:
            return None
        snippet = b[idx:idx+60]
        return snippet.decode(errors="replace")
    except Exception as e:
        return f"read error: {e}"

def log_to_stdout_and_stderr(*args):
    print(*args)
    print(*args, file=sys.stderr)

# --- Check environment & model presence at startup -----------------
MODEL_PATH = "rf_model.sav"

log_to_stdout_and_stderr("STARTUP CHECK")
log_to_stdout_and_stderr("cwd =", os.getcwd())
log_to_stdout_and_stderr("abs model path:", os.path.abspath(MODEL_PATH))
log_to_stdout_and_stderr("model exists:", os.path.exists(MODEL_PATH))
# runtime versions
try:
    import sklearn
    log_to_stdout_and_stderr("python:", sys.version.splitlines()[0])
    log_to_stdout_and_stderr("sklearn:", sklearn.__version__)
except Exception as e:
    log_to_stdout_and_stderr("Could not import sklearn:", e)

# peek inside file header for recorded sklearn version (safe)
log_to_stdout_and_stderr("model header snippet:", read_sklearn_version_from_file(MODEL_PATH))

# --- Load model (safe, with detailed logging) ----------------------
clf = None
if os.path.exists(MODEL_PATH):
    try:
        clf = joblib.load(MODEL_PATH)
        log_to_stdout_and_stderr("Model loaded OK, type:", type(clf))
    except Exception as e:
        # Print full traceback so Streamlit logs contain it
        log_to_stdout_and_stderr("Model load FAILED:", repr(e))
        traceback.print_exc(file=sys.stderr)
        # show message in UI as well
        st.error("Model failed to load. Check Streamlit logs for full traceback. "
                 "Likely causes: missing file, corrupted file, or sklearn version mismatch.")
else:
    st.error(f"Model file not found at {os.path.abspath(MODEL_PATH)}. Put rf_model.sav in the app folder or update MODEL_PATH.")

# --- Predict function ------------------------------------------------
def predict(data):
    if clf is None:
        raise RuntimeError("Model is not loaded.")
    return clf.predict(data)

# --- Class -> image safe mapping ------------------------------------
def class_to_image_from_label(label):
    # Normalize label to text name: support "Iris-setosa", "setosa", numeric indices etc.
    try:
        # decode bytes if necessary
        if isinstance(label, (bytes, bytearray)):
            label = label.decode(errors="ignore")
        label = str(label)
    except Exception:
        label = str(label)

    # If label looks like "Iris-setosa" -> split
    if "-" in label:
        short = label.split("-")[-1].lower()
    # If numeric index like "0", "1"
    elif label.isdigit():
        idx = int(label)
        mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
        short = mapping.get(idx, label)
    else:
        short = label.lower()

    images = {
        "setosa": "images/setosa.jpg",
        "versicolor": "images/versicolor.jpg",
        "virginica": "images/virginica.jpg"
    }
    return images.get(short, None)

# --- Streamlit UI ---------------------------------------------------
st.title('Classifying Iris Flowers')
st.markdown('Model to classify iris flowers into (setosa, versicolor, virginica) '
            'based on their sepal/petal length/width.')

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Sepal characteristics")
    sepal_l = st.slider('Sepal length (cm)', 1.0, 8.0, 4.0, step=0.1)
    sepal_w = st.slider('Sepal width (cm)', 0.1, 4.4, 3.0, step=0.1)

with col2:
    st.text("Petal characteristics")
    petal_l = st.slider('Petal length (cm)', 0.0, 7.0, 1.5, step=0.1)
    petal_w = st.slider('Petal width (cm)', 0.0, 2.5, 0.5, step=0.1)

st.text('')
if st.button("Predict type of Iris"):
    try:
        arr = np.array([[sepal_l, sepal_w, petal_l, petal_w]])
        result = predict(arr)
        label = result[0]
        st.text(f"Raw model output: {label}")
        image_path = class_to_image_from_label(label)
        if image_path and os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
        else:
            st.warning("No image found for predicted class or mapping failed. "
                       f"Predicted label after normalization: {label}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        traceback.print_exc(file=sys.stderr)
