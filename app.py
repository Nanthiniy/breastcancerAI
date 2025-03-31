from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pickle
import numpy as np
import pandas as pd
import os
import cv2
#from tensorflow.keras.models import load_model
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
app.secret_key = 'your_secret_key'

if not os.path.exists("uploads"):
    os.makedirs("uploads")

try:
    model_sav = pickle.load(open('model.sav', 'rb'))
except FileNotFoundError:
    model_sav = None
    print("Warning: model.sav not found.")

try:
    print()
    #cnn_model = load_model('CNN.model')
except FileNotFoundError:
    cnn_model = None
    print("Warning: CNN.model not found.")

tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT")
ai_model = AutoModelForCausalLM.from_pretrained("microsoft/BioGPT")

DATADIR = "train"
CATEGORIES = os.listdir(DATADIR) if os.path.exists(DATADIR) else []
IMG_SIZE = 100

def preprocess_image(file_path):
    img_array = cv2.imread(file_path, 1)
    img = cv2.medianBlur(img_array, 1)
    img = cv2.resize(img, (50, 50))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return np.asarray(img)

def predict_image(file_path):
    if cnn_model is None:
        return "Model not available"
    
    processed_image = preprocess_image(file_path)
    prediction = cnn_model.predict(processed_image)
    class_index = np.argmax(prediction)
    
    if class_index < len(CATEGORIES):
        return CATEGORIES[class_index]
    else:
        return "Unknown"

def recommend_medical_treatment(class_label):
    stage_recommendations = {
        "Low": "Treatment Plan: Regular monitoring, Surgery (lumpectomy or mastectomy), Hormonal therapy if needed. Diet Plan: High in fiber and antioxidants, Green leafy vegetables, berries, whole grains, Reduce processed foods. Lifestyle: Regular exercise (30 min/day), Stress management through yoga/meditation.",
        "Medium": "Treatment Plan: Surgery followed by radiation therapy, Chemotherapy if high risk of recurrence, Immunotherapy for specific cases. Diet Plan: Protein-rich diet (lean meats, beans), Anti-inflammatory foods (turmeric, ginger, nuts), Hydration and low sugar intake. Lifestyle: Light exercise, walking, Regular follow-ups with an oncologist, Emotional support and counseling.",
        "High": "Treatment Plan: Aggressive chemotherapy + radiation, Targeted therapy (HER2 inhibitors if applicable), Participation in clinical trials. Diet Plan: High-calorie, nutrient-dense meals to maintain strength, Omega-3 fatty acids (salmon, walnuts), Nutritional supplements if needed. Lifestyle: Physical therapy for fatigue management, Support groups and psychological counseling, Adequate rest and sleep.",
        "Stroma": "Treatment Plan: Personalized immunotherapy, Anti-angiogenic therapy, Advanced biomarker-based treatments. Diet Plan: Balanced diet rich in anti-inflammatory foods, Green tea, Curcumin, Whole grains, Low-fat dairy. Lifestyle: Maintain a healthy BMI, Engage in low-impact exercises, Regular check-ups with oncologists for targeted therapy adjustments."
    }
    
    return stage_recommendations.get(class_label, "No specific recommendation available.")

def load_responses():
    responses = {}
    try:
        with open("responses.txt", "r", encoding="utf-8") as file:
            for line in file:
                if "|" in line:
                    key, value = line.strip().split("|", 1)
                    responses[key.lower()] = value
    except FileNotFoundError:
        print("responses.txt not found!")
    return responses

def chatbot_response(message):
    responses = load_responses()
    return responses.get(message.lower(), "I'm not sure about that. Can you ask something else?")

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_reply = chatbot_response(user_message)
    return jsonify({"response": bot_reply})

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'csv_file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['csv_file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))

    if file and model_sav:
        try:
            data = pd.read_csv(file)
            input_features = data.values
            results = model_sav.predict(input_features)

            results_list = []
            for idx, result in enumerate(results):
                if result == 0:
                    results_list.append({
                        'row': idx + 1,
                        'diagnosis': 'Benign',
                        'advice': 'Maintain a healthy diet and exercise regularly.'
                    })
                elif result == 1:
                    results_list.append({
                        'row': idx + 1,
                        'diagnosis': 'Malignant',
                        'image_upload': True
                    })
            
            return render_template('results.html', results=results_list)
        except Exception as e:
            flash(f"An error occurred: {e}")
            return redirect(url_for('index'))
    else:
        flash("Model not loaded")
        return redirect(url_for('index'))

@app.route('/upload_image/<int:row>', methods=['POST'])
def upload_image(row):
    if 'image_file' not in request.files:
        flash('No image file uploaded')
        return redirect(url_for('index'))

    file = request.files['image_file']
    if file.filename == '':
        flash('No image selected')
        return redirect(url_for('index'))

    if file:
        try:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            class_label = predict_image(file_path)
            ai_recommendation = recommend_medical_treatment(class_label)

            os.remove(file_path)

            return render_template('advice.html', row=row, class_label=class_label, advice=ai_recommendation)
        except Exception as e:
            flash(f"An error occurred: {e}")
            return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
