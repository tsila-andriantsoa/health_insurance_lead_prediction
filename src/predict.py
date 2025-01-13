from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask('Health_insurance_prediction')

# Load the model and utilities 
loaded_pipeline = joblib.load('model/pipeline_best_model.pkl')
model = loaded_pipeline['model']
list_top_city = loaded_pipeline['list_top_city']
list_reco_policy_cat = loaded_pipeline['list_reco_policy_cat']
health_indicator_mode = loaded_pipeline['health_indicator_mode']



# Define the prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input from the request
        input_data = request.get_json()

        # Create dataframe based on JSON object
        df = pd.DataFrame(input_data, index = [0])       

        # rename data columns
        new_columns = ['id', 'city_code', 'region_code', 'accomodation_type', 'reco_insurance_type', 'upper_age', 'lower_age', 'is_spouse', 'health_indicator', 'holding_policy_duration'\
               , 'holding_policy_type', 'reco_policy_cat', 'reco_policy_premium']
        df.columns = new_columns

        # transform data type
        numerical_columns = ['upper_age', 'lower_age', 'reco_policy_premium']
        for num in numerical_columns:
            df[num] = pd.to_numeric(df[num])

        # fix missing value
        df['health_indicator'] = df['health_indicator'].fillna(health_indicator_mode)
    
        # Create new feature
        df['reco_policy_premium_log'] = np.log1p(df['reco_policy_premium'])
        df['city_code_group'] = np.where(df['city_code'].isin(list_top_city), df['city_code'], 'other')
        df['reco_policy_cat_group'] = np.where(df['reco_policy_cat'].isin(list_reco_policy_cat), df['reco_policy_cat'], 'other')
        
        best_features = ['reco_policy_premium_log', 'city_code_group', 'health_indicator', 'reco_policy_cat_group', 'accomodation_type']
        
        # Get only excepted columns from df
        df = df[best_features]

        # Make a prediction
        prediction_proba = model.predict_proba(df)[0,1]
        prediction = prediction_proba >= 0.5

        result = {
            'prediction_proba': float(prediction_proba),
            'prediction': float(prediction)
        }
        
        # Return the prediction as JSON
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Define a home check endpoint
@app.route('/home', methods=['GET'])
def home():
    return jsonify({'status': 'ok'}), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)