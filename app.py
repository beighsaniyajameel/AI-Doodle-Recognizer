
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from pathlib import Path
from streamlit_drawable_canvas import st_canvas

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Doodle Recognizer",
    page_icon="🎨",
    layout="wide"
)

MODEL_PATH = Path("models/doodle_cnn_improved.keras")

# Keep this order consistent with the model's training labels.
CLASS_NAMES = [
    "Cat",
    "Dog",
    "Car",
    "House",
    "Tree",
    "Apple",
    "Fish",
    "Bicycle",
    "Flower",
    "Star"
]

# --------------------------------------------------
# 2. PAGE DESIGN
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

.stApp {
    background-color: #ffffff;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='340' height='300' viewBox='0 0 340 300'%3E%3Cg fill='none' stroke-width='3' stroke-linecap='round' stroke-linejoin='round' opacity='.19'%3E%3Cg stroke='%239b9af5'%3E%3Cpath d='M35 53 Q19 29 15 51 L19 77 Q34 91 50 75 L54 49 Q48 31 35 53Z M14 62 L3 58 M52 62 L64 57 M24 72 L18 80 M44 72 L51 80'/%3E%3Cpath d='M254 38 L261 53 L278 55 L265 67 L268 84 L254 75 L240 84 L243 67 L230 55 L247 53Z'/%3E%3C/g%3E%3Cg stroke='%23f1a9c6'%3E%3Cpath d='M122 160 L151 130 L180 160Z M130 160 V190 H171 V160 M145 190 V173 H157 V190'/%3E%3Cpath d='M67 228 C48 208 30 226 45 241 L67 260 L89 241 C104 226 86 208 67 228Z'/%3E%3C/g%3E%3Cg stroke='%2397d4bf'%3E%3Cpath d='M287 209 C270 190 249 204 254 222 C258 237 278 246 287 253 C296 246 316 237 320 222 C325 204 304 190 287 209Z'/%3E%3Cpath d='M208 75 Q195 58 184 73 Q178 84 190 97 L208 111 L226 97 Q238 84 232 73 Q221 58 208 75Z'/%3E%3C/g%3E%3Cg stroke='%23edcf8b'%3E%3Ccircle cx='112' cy='68' r='17'/%3E%3Cpath d='M112 40 V32 M112 96 V104 M84 68 H76 M140 68 H148 M92 48 L86 42 M132 88 L138 94 M132 48 L138 42 M92 88 L86 94'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 340px 300px;
    font-family: 'DM Sans', sans-serif;
    color: #202b4d;
}

[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0.85);
}

.block-container {
    padding-top: 2rem;
    max-width: 1250px;
}

h1, h2, h3 {
    color: #202b4d !important;
    font-family: 'DM Sans', sans-serif !important;
}

p, label, [data-testid="stCaptionContainer"] {
    color: #66718e;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid #e8eafa;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(50, 60, 110, 0.045);
}

.stButton > button[kind="primary"] {
    background: #7965ef;
    color: white;
    border: 1px solid #7965ef;
    border-radius: 12px;
    min-height: 3rem;
    font-weight: 700;
}

.stButton > button[kind="primary"]:hover {
    background: #6652dc;
    border-color: #6652dc;
    color: white;
}

.stButton > button:not([kind="primary"]) {
    border: 1px solid #dcd8ff;
    border-radius: 10px;
    color: #6554d9;
    background: white;
}

[data-testid="stMetric"] {
    background: #f8f7ff;
    border: 1px solid #ebe8ff;
    border-radius: 14px;
    padding: 12px 16px;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

hr {
    border-color: #e8eafa;
}

footer {
    visibility: hidden;
}

.class-pill {
    display: inline-block;
    background: rgba(247, 246, 255, 0.96);
    border: 1px solid #e7e3ff;
    border-radius: 20px;
    padding: 7px 13px;
    margin: 4px 4px 4px 0;
    color: #4d5577;
    font-size: 0.9rem;
}

@media (max-width: 768px) {
    .block-container {
        padding-top: 1rem;
    }

    h1 {
        font-size: 2rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Main heading
st.markdown(
    """
    <h1 style="text-align:center; margin-bottom:0.2rem;">
        AI Doodle Recognizer
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style="text-align:center; font-size:1.05rem;">
        Draw a doodle and let a Convolutional Neural Network recognize it.
    </p>
    """,
    unsafe_allow_html=True
)

# Display every supported class
st.markdown(
    """
    <h3 style="text-align:center; font-size:1rem; margin-top:1.2rem;">
        What can you draw?
    </h3>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "".join(
        f'<span class="class-pill">{name}</span>'
        for name in CLASS_NAMES
    ),
    unsafe_allow_html=True
)

st.divider()

# --------------------------------------------------
# 3. LOAD THE TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()

if model is None:
    st.error(
        f"Model not found at {MODEL_PATH}. "
        "Please ensure the trained model is included in your repository."
    )
    st.stop()

# --------------------------------------------------
# 4. DRAWING CANVAS
# --------------------------------------------------

left_col, right_col = st.columns([1.4, 1], gap="large")

with left_col:
    st.subheader("Draw your doodle")
    st.caption("Use your mouse or touch screen to draw on the white canvas.")

    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=12,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=350,
        width=500,
        drawing_mode="freedraw",
        key="doodle_canvas",
        update_streamlit=True,
        return_image_data=True
    )

    st.caption(
        "Draw with a black pen · Keep it simple · "
        "Try one object at a time"
    )

    predict_button = st.button(
        "Predict Doodle",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# 5. PREPROCESS DRAWING
# --------------------------------------------------

def preprocess_drawing(rgba_image):
    """
    Convert the drawing into a normalized 28x28 grayscale image.
    The inversion matches the white-stroke-on-black-background
    format used by the training dataset.
    """

    image = Image.fromarray(
        np.asarray(rgba_image).astype(np.uint8)
    ).convert("RGBA")

    # Composite transparent pixels over white.
    white_background = Image.new(
        "RGBA", image.size, (255, 255, 255, 255)
    )

    image = Image.alpha_composite(
        white_background, image
    ).convert("L")

    # Invert black strokes on white into white strokes on black.
    image = ImageOps.invert(image)

    image_array = np.asarray(image)
    mask = image_array > 20

    if not np.any(mask):
        return None

    # Crop around the drawn content.
    ys, xs = np.where(mask)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    image = image.crop((
        int(x_min),
        int(y_min),
        int(x_max) + 1,
        int(y_max) + 1
    ))

    # Resize while preserving the aspect ratio.
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    # Center the doodle in a 28x28 image.
    canvas = Image.new("L", (28, 28), 0)

    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    canvas.paste(image, (x_offset, y_offset))

    # Normalize pixel values to 0–1.
    array = np.asarray(canvas, dtype=np.float32) / 255.0

    return array.reshape(1, 28, 28, 1)

# --------------------------------------------------
# 6. PREDICTION RESULTS
# --------------------------------------------------

with right_col:
    st.subheader("Prediction result")
    st.caption("Your doodle will be classified here.")

    if predict_button:
        if canvas_result.image_data is None:
            st.warning(
                "The canvas image is unavailable. "
                "Please draw an object and try again."
            )

        else:
            processed = preprocess_drawing(
                canvas_result.image_data
            )

            if processed is None:
                st.warning("The canvas is empty. Draw an object first.")

            else:
                with st.spinner("Recognizing your doodle..."):
                    probabilities = model.predict(
                        processed, verbose=0
                    )[0]

                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index])

                st.success(f"Prediction: **{predicted_class}**")

                st.metric(
                    "Model confidence",
                    f"{confidence * 100:.2f}%"
                )

                if confidence < 0.50:
                    st.info(
                        "The model is uncertain. Try drawing "
                        "a simpler, clearer outline."
                    )

                st.subheader("Class probabilities")

                chart_data = {
                    "Class": CLASS_NAMES,
                    "Probability (%)": probabilities * 100
                }

                st.bar_chart(
                    chart_data,
                    x="Class",
                    y="Probability (%)",
                    horizontal=True
                )

                st.caption(
                    "Confidence is the model's predicted probability, "
                    "not a guarantee that the prediction is correct."
                )

    else:
        st.info(
            "Draw an object on the left and click "
            "**Predict Doodle**."
        )

        st.subheader("Class probabilities")
        st.caption("Prediction confidence for each class will appear here.")

        st.markdown(
            """
            <div style="
                background:#f8f7ff;
                border:1px solid #ebe8ff;
                border-radius:14px;
                padding:34px 18px;
                text-align:center;
                color:#7b86a3;">
                Draw something and click <b>Predict Doodle</b>
                to see your results.
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# 7. PROJECT INFORMATION
# --------------------------------------------------

st.divider()

with st.expander("About this project"):
    st.write(
        """
        **Objective:** Recognize hand-drawn objects using a
        Convolutional Neural Network.

        **Supported classes:** Cat, Dog, Car, House, Tree, Apple,
        Fish, Bicycle, Flower and Star.

        **Dataset:** QuickDraw-style doodle images.

        **Technologies:** Python, TensorFlow, Keras, Streamlit,
        NumPy and Pillow.

        **Evaluation:** The model's test accuracy is measured
        on a held-out dataset. Freehand drawing accuracy can
        differ from test accuracy.
        """
    )

st.caption("AI Doodle Recognizer · CNN Image Classification")
