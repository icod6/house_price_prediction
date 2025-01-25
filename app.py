from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from the form
    features = [float(request.form['bedrooms']), float(request.form['bathrooms']),
                float(request.form['size']), float(request.form['longitude']),
                float(request.form['latitude'])]
    features = np.array([features])
    prediction = model.predict(features)
    
    # Format the prediction to include the currency symbol
    prediction_text = f'Predicted House Price: ${prediction[0]:,.2f} USD'
    return render_template('home.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
