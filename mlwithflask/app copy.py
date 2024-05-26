from flask import Flask , render_template, request, jsonify
import joblib 
import pandas as pd
import pickle
import numpy as np
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load the model
try:
    model = joblib.load('random_forest_model.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    try:
        # Extract data from form
        Transaction_Amount = float(request.form['Transaction_Amount'])
        Amount_paid = float(request.form['Amount_paid'])
        Vehicle_Speed = float(request.form['Vehicle_Speed'])
        
        # Create a DataFrame
        data = pd.DataFrame([[Transaction_Amount, Amount_paid, Vehicle_Speed]], columns=['Transaction_Amount', 'Amount_paid', 'Vehicle_Speed'])
        print("Data prepared for prediction:", data)
        
        # Make prediction
        prediction = model.predict(data)[0]
        print("Prediction:", prediction)
        
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(port=3000,debug=True)