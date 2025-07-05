from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
with open("goalie.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Correct features as per Goalkeepers.csv
feature_names = [
    "Age", "Club Level", "Minutes Played", "Goals Against", "Shots on target Against",
    "Saves", "Save perc", "Penalty saved", "Clean Sheets", "Games Missed"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            inputs = [float(request.form[f]) for f in feature_names]
            inputs_scaled = scaler.transform([inputs])
            prediction = model.predict(inputs_scaled)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("index.html", feature_names=feature_names, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True,port=3000)
