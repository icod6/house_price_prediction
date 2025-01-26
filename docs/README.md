# House Price Prediction

    This application predicts house prices based on features such as the number of bedrooms, total rooms, and geographical location (longitude and latitude). It is built using Flask and a machine learning model trained with Linear Regression.

## Installation

    To set up the project locally, 
    steps:

    1. **Clone the repository**:
        commands: on windows
        git clone https://github.com/icod6/house_price_prediction.git
        cd house_price_prediction

    3. **Install the dependencies:**
        command: on windows
        pip install -r requirements.txt


## Usage
    execute model with command:
    python model.py
    
    Run the Flask application with command: 
    python app.py

    Open browser and go to http://127.0.0.1:5000/.

    - Fill out the form with the desired features (Number of Bedrooms, Number of bathrooms, Ara in sqft, Longitude, Latitude).

    - Submit the form to get the predicted house price.


## Features

    Predict house prices based on user input.

    Real-time predictions with a simple web interface.

    Features used for prediction: Number of Bedrooms, Number of bathrooms, Ara in sqft, Longitude, Latitude.


## Model

    Algorithm: Linear Regression

    Features:
    Number of Bedrooms

    Number of bathrooms

    Area in sqft

    Location (Longitude, Latitude)



## Evaluation
    model was evaluated using the following metrics:

    Mean Absolute Error (MAE): 72571.67725541844
    Mean Squared Error (MSE): 9188058223.373596      
    Root Mean Squared Error (RMSE): 95854.35943854404
    R-squared (R²): 0.2988404096782292 


## Requirements

    To run this project, you'll need to install the following Python packages:

    - Flask==2.0.2
    - pandas==1.3.3
    - scikit-learn==0.24.2
    - numpy==1.21.2

    You can install these dependencies using the `requirements.txt` file., then run:
    command: on windows
    pip install -r requirements.txt


