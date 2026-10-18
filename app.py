
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

# IMPORTANT: This order must match prepare_dataset.py
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

st.title("AI Doodle Recognizer")
st.write(
    "Draw an object and let a Convolutional Neural Network "
    "(CNN) predict what you have drawn."
)

st.caption(
    "Recognizes: Cat · Dog · Car · House · Tree · Apple · "
    "Fish · Bicycle · Flower · Star"
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
    model = None
    st.error(f"Could not load the model: {error}")

if model is None:
    st.error(
        f"Model not found at {MODEL_PATH}. "
        "Train the model first using train_model.py."
    )
    st.stop()

# --------------------------------------------------
# 4. DRAWING CANVAS
# --------------------------------------------------

left_col, right_col = st.columns([1, 1.25], gap="large")

with left_col:
    st.subheader("Draw your doodle")
    st.caption("Use your mouse to draw inside the white canvas.")

    canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",
    stroke_width=12,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=350,
    width=350,
    drawing_mode="freedraw",
    key="doodle_canvas",
    update_streamlit=True,
    return_image_data=True
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
    Convert the canvas to a 28x28 grayscale image.

    Training data uses 28x28 QuickDraw images with a black
    background and white strokes. Convert the white canvas
    and black strokes into that same polarity.
    """

    image = Image.fromarray(
        np.asarray(rgba_image).astype(np.uint8)
    ).convert("RGBA")

    # Composite on white so transparent pixels are handled.
    white_background = Image.new(
        "RGBA", image.size, (255, 255, 255, 255)
    )
    image = Image.alpha_composite(
        white_background, image
    ).convert("L")

    # Black drawing on white background -> white drawing
    # on black background, matching QuickDraw-style images.
    image = ImageOps.invert(image)

    # Crop around the drawn content to remove empty margins.
    # Pixels brighter than a small threshold count as strokes.
    image_array = np.asarray(image)
    mask = image_array > 20

    if not np.any(mask):
        return None

    ys, xs = np.where(mask)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    image = image.crop((
        int(x_min),
        int(y_min),
        int(x_max) + 1,
        int(y_max) + 1
    ))

    # Preserve aspect ratio and leave a small margin.
    image.thumbnail((20, 20), Image.Resampling.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)

    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2

    canvas.paste(image, (x_offset, y_offset))

    # Match the normalized training data.
    array = np.asarray(canvas, dtype=np.float32) / 255.0

    return array.reshape(1, 28, 28, 1)

# --------------------------------------------------
# 6. PREDICTION RESULTS
# --------------------------------------------------

with right_col:
    st.subheader("Prediction")

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
                st.warning(
                    "The canvas is empty. Draw an object first."
                )
            else:
                with st.spinner("Recognizing your doodle..."):
                    probabilities = model.predict(
                        processed, verbose=0
                    )[0]

                predicted_index = int(
                    np.argmax(probabilities)
                )

                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(
                    probabilities[predicted_index]
                )

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

                # Keep the chart in the same label order as training.
                chart_data = {
                    "Class": CLASS_NAMES,
                    "Probability (%)": (
                        probabilities * 100
                    )
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

# --------------------------------------------------
# 7. PROJECT INFORMATION
# --------------------------------------------------

st.divider()

with st.expander("About this project"):
    st.write(
        """
        **Objective:** Recognize hand-drawn objects using a
        Convolutional Neural Network.

        **Dataset:** QuickDraw-style doodle images.

        **CLASS_NAMES:** 10 object categories.

        **Technologies:** Python, TensorFlow, Keras, Streamlit,
        NumPy and Pillow.

        **Evaluation:** The model's test accuracy is measured
        on a held-out dataset. Freehand drawing accuracy can
        differ from test accuracy.
        """
    )

st.caption("AI Doodle Recognizer | CNN Image Classification")
