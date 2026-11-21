
import streamlit as st
from PIL import Image
from predict_cell import predict_image

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------

st.set_page_config(
    page_title="Blood Smear AI",
    page_icon="🔬",
    layout="wide"
)

# -------------------------------
# CUSTOM DESIGN
# -------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    color: #163a5f;
}

.stButton > button {
    background-color: #1769aa;
    color: white;
    border-radius: 8px;
    height: 3em;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0d4777;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------

st.title("🔬 Blood Smear AI")

st.subheader(
    "AI-Based White Blood Cell Classification"
)

st.write(
    "An educational machine learning project "
    "using PyTorch and Convolutional Neural Networks."
)

st.divider()

# -------------------------------
# PROJECT INFORMATION
# -------------------------------

col1, col2, col3 = st.columns(3)

col1.metric("Cell Categories", "5")
col2.metric("Test Accuracy", "98.26%")
col3.metric("Test Images", "1,549")

st.caption(
    "Reported test accuracy from the current evaluation. "
    "Dataset independence and integrity have not yet "
    "been fully verified."
)

st.divider()

# -------------------------------
# IMAGE UPLOAD
# -------------------------------

st.header("Upload Blood Smear Image")

st.write(
    "Upload an image in JPG, JPEG or PNG format "
    "to begin classification."
)

uploaded_file = st.file_uploader(
    "Choose your image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# IMAGE DISPLAY
# -------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Microscopy Image",
        width=400
    )

    st.success("Image uploaded successfully!")

    # ---------------------------
    # PREDICTION
    # ---------------------------

    if st.button(
        "🔬 Analyze Blood Cell",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            predicted_cell, confidence, scores = (
                predict_image(image)
            )

        st.divider()

        st.header("Classification Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.subheader("Predicted Cell")

            st.success(predicted_cell.capitalize())

        with result_col2:

            st.subheader("Model Confidence")

            st.metric(
                "Prediction score",
                f"{confidence:.2f}%"
            )

        st.divider()

        st.subheader("Class Probability Scores")

        st.bar_chart(scores)

        st.write(
            "These scores represent the model's "
            "softmax outputs for the five classes."
        )

# -------------------------------
# ABOUT THE PROJECT
# -------------------------------

st.divider()

with st.expander("About this project"):

    st.write("""
    This project uses a Convolutional Neural Network
    (CNN) developed with PyTorch.

    The model classifies five white blood cell categories:

    - Basophil
    - Eosinophil
    - Lymphocyte
    - Monocyte
    - Neutrophil

    The interface is developed using Streamlit.
    """)

# -------------------------------
# DISCLAIMER
# -------------------------------

st.divider()

st.warning(
    "Educational Research Prototype: This application "
    "is intended for educational and research purposes "
    "only. It is not clinically validated and must not "
    "be used for medical diagnosis."
)

st.caption(
    "Developed as an independent AI and medical "
    "laboratory science learning project."
)

