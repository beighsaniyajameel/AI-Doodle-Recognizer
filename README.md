# AI Doodle Recognizer

A deep learning web application that recognizes hand-drawn doodles using a Convolutional Neural Network (CNN). Draw an object on the canvas, and the model predicts which category it belongs to.

**Live Demo:** [Try AI Doodle Recognizer](https://ai-doodle-recognizer.streamlit.app/)

## Overview

AI Doodle Recognizer is an interactive image classification project built using Python, TensorFlow, Keras, and Streamlit. Users can draw an object directly on the web interface and receive a prediction along with the model's confidence score and class probability distribution.

The project demonstrates how deep learning can be applied to image classification through an interactive web application.

## Supported Classes

The model recognizes the following 10 categories:

- Cat
- Dog
- Car
- House
- Tree
- Apple
- Fish
- Bicycle
- Flower
- Star

## Features

- Interactive drawing canvas for sketching objects.
- CNN-based doodle classification.
- Prediction label and confidence score.
- Probability visualization across all supported classes.
- Clean, responsive interface with a yellow-and-cream theme.
- Publicly accessible web application deployed using Streamlit Community Cloud.

## Technologies Used

- **Python** — application development and data processing
- **TensorFlow and Keras** — deep learning model loading and inference
- **NumPy** — numerical operations and image array processing
- **Pillow (PIL)** — image preprocessing
- **Streamlit** — interactive web interface
- **streamlit-drawable-canvas** — drawing functionality
- **Git and GitHub** — version control and project hosting

## How It Works

1. The user draws an object on the interactive canvas.
2. The drawing is converted to grayscale and preprocessed into a normalized 28 × 28 image.
3. The trained CNN receives the processed image.
4. The model generates probabilities for the 10 supported classes.
5. The application displays the predicted class, confidence score, and probability chart.

## Model Performance

The improved CNN model achieved **94.50% test accuracy** during evaluation on the held-out test dataset.

Actual performance on user-created drawings may differ from the reported test accuracy.

## Project Structure

```text
AI-Doodle-Recognizer/
├── app.py
├── train_model.py
├── download_dataset.py
├── requirements.txt
├── models/
│   └── doodle_cnn_improved.keras
├── dataset/
├── notebooks/
└── results/
```

The `dataset/`, `notebooks/`, and `results/` directories may contain local development files and may not all be included in the GitHub repository.

## Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/beighsaniyajameel/AI-Doodle-Recognizer.git
cd AI-Doodle-Recognizer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser at the local URL displayed in the terminal.

**Note:** The trained model file must be present at `models/doodle_cnn_improved.keras` for the application to run. If it is not included in a fresh clone, obtain the model from your training workflow.

## Future Improvements

- Improve prediction accuracy on freehand user drawings.
- Add more doodle categories.
- Display the input drawing alongside the processed model image.
- Explore alternative CNN architectures and compare their performance.

## Author

**Saniya Jameel Beigh**

GitHub: [@beighsaniyajameel](https://github.com/beighsaniyajameel)

---

If you find this project interesting, feel free to explore the repository and try the live demo.
