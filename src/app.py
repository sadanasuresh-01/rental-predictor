from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '..', 'models', 'rental_model.pkl')

# Load model
model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        bedrooms = float(request.form['bedrooms'])
        bathrooms = float(request.form['bathrooms'])
        squareft = float(request.form['squareft'])

        # Combine into feature array
        features = np.array([[bedrooms, bathrooms, squareft]])

        # Predict
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=round(prediction, 2))

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
