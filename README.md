# breastcancerAI

**breastcancerAI** is a Python-based machine learning project designed to predict breast cancer using various algorithms such as K-Nearest Neighbors (KNN) and Convolutional Neural Networks (CNN). The project also includes a simple web interface for making real-time predictions.

---

📚 Table of Contents

* [Features](#features)
* [Project Structure](#project-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Requirements](#requirements)
* [Dataset](#dataset)
* [Model Saving](#model-saving)
* [Future Enhancements](#future-enhancements)
* [License](#license)
* [Contact](#contact)

---

🚀 Features

* Train machine learning models on breast cancer datasets
* Evaluate model accuracy and performance
* Predict new input samples
* Web-based interface for interactive predictions
* Option to compare results across algorithms
* Support scripts for feature importance and chatbot interactions

---

📂 Project Structure

```
breastcancerAI/
├── app.py                # Web app for predictions
├── chatbot.py            # Chat-based interface (optional)
├── cnntrain.py           # CNN model training
├── cnn test.py           # CNN testing and evaluation
├── knn train(1).py       # KNN model training
├── knn test(1).py        # KNN model testing
├── fi.py                 # Feature importance or preprocessing
├── model.sav             # Saved trained model
├── data(1).csv           # Training dataset
├── test.csv              # Testing dataset
├── t2.csv                # Additional data (if any)
├── responses.txt         # Response data for chatbot
└── README.md             # Project documentation
```

---

⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Nanthiniy/breastcancerAI.git
   cd breastcancerAI
   ```

2. **(Optional) Create a virtual environment**

   ```bash
   python -m venv venv
   venv\Scripts\activate          # On Windows
   source venv/bin/activate       # On macOS/Linux
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

   If no `requirements.txt` file is available, manually install:

   ```bash
   pip install numpy pandas scikit-learn tensorflow flask
   ```

---

🧠 Usage

1. **Training the Model**

Run any of the following depending on the model you want to train:

```bash
python "knn train(1).py"
python cnntrain.py
```

This will train the model on your dataset and save it as `model.sav`.

2. **Testing the Model**

To evaluate your trained model:

```bash
python "knn test(1).py"
python "cnn test.py"
```

You’ll get performance metrics such as accuracy, precision, or confusion matrix (depending on your script).

3. **Launching the Web Interface**

Start the prediction web app:

```bash
python app.py
```

Then open your browser and go to:
➡️ `http://localhost:5000`

Enter patient data to receive an AI-based cancer prediction.

 4. **Chatbot (Optional)**

If your project includes chatbot functionality:

```bash
python chatbot.py
```

This will open a simple console interface where you can interact with the system.

---

🧩 Requirements

* Python 3.7 or above
* numpy
* pandas
* scikit-learn
* tensorflow / keras
* flask
* joblib or pickle

To generate a `requirements.txt` file automatically:

```bash
pip freeze > requirements.txt
```

🧪 Dataset

The project uses CSV datasets such as:

* `data(1).csv` – for training
* `test.csv` – for testing
* `t2.csv` – for additional validation or experiments

Ensure your dataset follows a similar structure (feature columns + target column). You can replace them with your custom data, but maintain consistent formatting.

---

💾 Model Saving

Trained models are serialized and stored as **`model.sav`** using `pickle` or `joblib`.
This allows you to reuse the trained model without retraining each time.

---

🔮 Future Enhancements

* Implement additional ML algorithms (Random Forest, SVM, XGBoost)
* Improve data preprocessing and feature selection
* Add cross-validation and hyperparameter tuning
* Enhance UI/UX of the web interface
* Integrate model explainability tools (SHAP, LIME)
* Deploy to cloud (AWS, Streamlit, etc.)



---



 
