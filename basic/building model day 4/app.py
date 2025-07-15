import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# --- 1. Improved Preprocessing Function ---
def preprocess_image(uploaded_image):
    """Convert image to match MNIST-like 8x8 format (0-16 scale)"""
    try:
        # Convert to grayscale and auto-contrast
        img = uploaded_image.convert("L")
        img = ImageOps.autocontrast(img, cutoff=2)
        
        # Crop to digit (remove empty borders)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        # Resize to 8x8 with anti-aliasing
        img = img.resize((8, 8), Image.Resampling.LANCZOS)
        img_array = np.array(img)
        
        # Auto-invert colors if background is light
        if np.mean(img_array) > 128:
            img_array = 255 - img_array
        
        # Scale to 0-16 (like sklearn's digits dataset)
        img_scaled = (img_array / 255.0) * 16
        return img_scaled.flatten().reshape(1, -1)
    
    except Exception as e:
        st.error(f"Preprocessing failed: {str(e)}")
        return None

# --- 2. Load Model and Sample Data ---
@st.cache_resource
def load_resources():
    model = joblib.load("E:/AI project/basic/kashi_digit_classifier_model.pkl")
    digits = load_digits()
    return model, digits

model, digits = load_resources()

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Digit Classifier", layout="wide")
st.title("🔢 Handwritten Digit Classifier")
st.markdown("""
Upload an image of a handwritten digit (0-9) or draw one below.
The model will predict the digit using your trained classifier.
""")

# --- 4. File Upload Section ---
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

# --- 5. Drawing Canvas (Optional) ---
with col2:
    st.write("Or draw a digit:")
    try:
        from streamlit_drawable_canvas import st_canvas
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=200,
            width=200,
            drawing_mode="freedraw",
            key="canvas"
        )
    except ImportError:
        st.warning("Install `streamlit-drawable-canvas` for drawing support")

# --- 6. Process Input and Predict ---
def display_results(processed_img):
    """Show prediction results and debug info"""
    with st.expander("🔍 Detailed Results", expanded=True):
        # Show processed image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(processed_img.reshape(8, 8), cmap="gray", vmin=0, vmax=16)
        ax1.set_title("Processed 8×8 Image")
        
        # Show pixel values
        ax2.axis('off')
        pixel_table = ax2.table(
            cellText=processed_img.reshape(8, 8).round(1),
            loc='center',
            cellLoc='center'
        )
        pixel_table.auto_set_font_size(False)
        pixel_table.set_fontsize(8)
        ax2.set_title("Pixel Values (0-16)")
        st.pyplot(fig)
        
        # Get prediction
        pred = model.predict(processed_img)[0]
        proba = model.predict_proba(processed_img)[0]
        confidence = proba.max() * 100
        
        # Display top 3 predictions
        st.success(f"**Predicted Digit:** {pred} ({confidence:.1f}% confidence)")
        top3 = np.argsort(proba)[-3:][::-1]
        st.write("Top Predictions:")
        for i, digit in enumerate(top3):
            st.write(f"{i+1}. Digit {digit} ({proba[digit]*100:.1f}%)")

# --- 7. Handle Uploaded File ---
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        processed_img = preprocess_image(image)
        if processed_img is not None:
            display_results(processed_img)
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# --- 8. Handle Canvas Drawing ---
if 'canvas_result' in locals() and canvas_result.image_data is not None:
    try:
        # Convert canvas to PIL image
        canvas_img = Image.fromarray(canvas_result.image_data.astype('uint8'))
        processed_img = preprocess_image(canvas_img)
        if processed_img is not None:
            display_results(processed_img)
    except Exception as e:
        st.error(f"Error processing drawing: {str(e)}")

# --- 9. Test Samples Section ---
with st.sidebar:
    st.header(" Test Samples")
    sample_digit = st.selectbox("Try a training sample:", range(10))
    if st.button("Load Sample"):
        sample_idx = np.where(digits.target == sample_digit)[0][0]
        sample_img = digits.images[sample_idx]
        processed_sample = digits.data[sample_idx].reshape(1, -1)
        
        # Display
        st.image(sample_img, caption=f"True Label: {sample_digit}", width=100)
        pred = model.predict(processed_sample)[0]
        st.write(f"Model Prediction: **{pred}**")
        
        if pred != sample_digit:
            st.error(" Model misclassified this training sample!")