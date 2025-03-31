import tkinter as tk
from tkinter import messagebox
import pickle
import numpy as np

# Load the model
filename = 'model.sav'
model = pickle.load(open(filename, 'rb'))

def predict():
    try:
        # Get input values
        inputs = [float(entry.get()) for entry in entries]
        
        # Prepare the input for the model
        input_features = np.array([inputs])
        
        # Make the prediction
        result = model.predict(input_features)
        
        # Show the result
        messagebox.showinfo("Prediction Result", f"The predicted result is: {result[0]}")
    except Exception as e:
        messagebox.showerror("Input Error", f"Please check your inputs! Error: {e}")

# Create the main window
root = tk.Tk()
root.title("Health Prediction System")

# Create a frame for the canvas and scrollbar
frame = tk.Frame(root)
frame.pack(fill=tk.BOTH, expand=True)

canvas = tk.Canvas(frame)
scrollbar = tk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(
        scrollregion=canvas.bbox("all")
    )
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

entries = []
fields = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

for i, field in enumerate(fields):
    tk.Label(scrollable_frame, text=field).grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(scrollable_frame)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

predict_button = tk.Button(scrollable_frame, text="Predict", command=predict)
predict_button.grid(row=len(fields), column=0, columnspan=2, pady=20)

# Run the application
root.mainloop()
