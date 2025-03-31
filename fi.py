import tkinter as tk
from tkinter import filedialog, messagebox
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
import pandas as pd


# Load the models
model_sav = pickle.load(open('model.sav', 'rb'))
cnn_model = load_model('CNN.model')



# Image classification details
DATADIR = "train"
CATEGORIES = os.listdir(DATADIR)
IMG_SIZE = 100

# Preprocess image
def preprocess_image(file_path):
    img_array = cv2.imread(file_path, 1)
    img = cv2.medianBlur(img_array, 1)
    img = cv2.resize(img, (50, 50))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    img_array = np.asarray(img)

    return img_array

# Predict using CNN model
def predict_image(file_path):
    processed_image = preprocess_image(file_path)
    prediction = cnn_model.predict(processed_image)
    class_index = np.argmax(prediction)
    class_label = CATEGORIES[class_index]
    return class_label

from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load DistilGPT-2 tokenizer and model (482MB)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

def generate_ai_response(prompt, max_length=100):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Recommend health remedies based on classification
def recommend_health_remedies_with_ai(class_label):
    prompt = "Provide detailed health advice for malignant breast cancer cases."
    response = generate_ai_response(prompt)
    return response

# Function to upload and process CSV file
def upload_csv():
    try:
        # Open file dialog to select the CSV
        file_path = filedialog.askopenfilename(
            title="Select a CSV file",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            return
        
        # Load CSV data
        data = pd.read_csv(file_path)
        input_features = data.values  # Convert to NumPy array
        
        # Predict using model.sav
        results = model_sav.predict(input_features)
        
        # Display predictions and recommendations
        for idx, result in enumerate(results):
            if result == 0:
                messagebox.showinfo(
                    f"Prediction {idx+1}",
                    f"Row {idx+1}: Diagnosis: Benign\nRecommended: Maintain a healthy diet and exercise regularly."
                )
            elif result == 1:
                file_path = filedialog.askopenfilename(
                    title=f"Select an image file for Row {idx+1}",
                    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")]
                )
                if file_path:
                    class_label = predict_image(file_path)
                    ai_recommendation = recommend_health_remedies_with_ai(class_label)
                    messagebox.showinfo(
                        f"Malignant Case Analysis {idx+1}",
                        f"Row {idx+1}: Predicted Class: {class_label}\nAI-Generated Advice: {ai_recommendation}"
                    )
                else:
                    messagebox.showerror("Error", f"No image selected for Row {idx+1}!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Create the main window
root = tk.Tk()
root.title("Breast Cancer Prediction System")

# CSV Upload Button
upload_csv_button = tk.Button(root, text="Upload CSV for Testing", command=upload_csv)
upload_csv_button.pack(pady=20)

# Run the application
root.mainloop()
