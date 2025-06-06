from flask import Flask, request, render_template
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import customData,PredictPipeline
from src.logger import logging
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
@application.route('/')
def home():
    logging.info("Rendering the home page.")
    return render_template('index.html')

@application.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        logging.info("Received GET request for prediction page.")
        return render_template('home.html')
    else:
        logging.info("Received POST request for prediction.")
        data=customData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score')

        )
        data_df=data.get_data_as_dataframe()
        logging.info(f"Dataframe created from input: {data_df}")
        print(f"Data received from form: {data_df}")
        # Predict using the PredictPipeline
        predict_pipeline = PredictPipeline()
        try:
            prediction = predict_pipeline.predict_predictions(data_df)
            logging.info(f"Prediction made: {prediction}")
            print(f"Prediction made: {prediction}")
            return render_template('home.html', results=prediction[0])
        except CustomException as e:
            logging.error(f"Error during prediction: {e}")
            return render_template('home.html', prediction_text='Error in prediction. Please check the input data.')

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000)