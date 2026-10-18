import os
import urllib.request

# AI DOODLE RECOGNIZER


CLASSES = [
    "cat",
    "dog",
    "car",
    "house",
    "tree",
    "apple",
    "fish",
    "bicycle",
    "flower",
    "star"
]

BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"

OUTPUT_DIR = os.path.join("dataset", "raw")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def download_file(class_name):
    filename = class_name + ".npy"
    url = BASE_URL + filename
    final_path = os.path.join(OUTPUT_DIR, filename)

    # Temporary file prevents an incomplete download
    # from being mistaken for a complete dataset file.
    temp_path = final_path + ".part"
    
    print(f"Downloading: {class_name}")

    try:
        urllib.request.urlretrieve(url, temp_path)

        # Only rename after the download has completed.
        os.replace(temp_path, final_path)

        file_size = os.path.getsize(final_path)

        print(f"SUCCESS: {filename}")
        print(f"File size: {file_size:,} bytes")

    except Exception as e:

        print(f"ERROR downloading {filename}: {e}")

        if os.path.exists(temp_path):
            os.remove(temp_path)

print("AI DOODLE RECOGNIZER - DATASET DOWNLOAD")

for class_name in CLASSES:
    download_file(class_name)

print("DOWNLOAD PROCESS COMPLETED")