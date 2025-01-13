# health_insurance_lead_prediction

Welcome to the **health insurance prediction** project! 

This repository provides tools and resources for predicting whether a website visitor is interesed in recommended health insurance or not.

![health insurrance](https://github.com/tsila-andriantsoa/health_insurance_lead_prediction/blob/main/img/health_insurrance.jfif)

## Context

Your Client FinMan is a financial services company that provides various financial services like loan, investment funds, insurance etc. to its customers. FinMan wishes to cross-sell health insurance to the existing customers who may or may not hold insurance policies with the company. The company recommend health insurance to it's customers based on their profile once these customers land on the website. Customers might browse the recommended health insurance policy and consequently fill up a form to apply. When these customers fill-up the form, their Response towards the policy is considered positive and they are classified as a lead.

Once these leads are acquired, the sales advisors approach them to convert and thus the company can sell proposed health insurance to these leads in a more efficient manner.

## Objective

The problem can be modelized as a binary classification. 
A policy is recommended to a person when they land on an insurance website, and if the person chooses to fill up a form to apply, it is considered a Positive outcome (Classified as lead). All other conditions are considered Zero outcomes.
The objective of this project is to provide a machine learning model that could help the company FinMan to predict whether a person with a recommended policy will choose to fill up a form to apply or not.

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/sureshmecad/health-insurance-lead-prediction)

| Variable | Definition |
|------------------|-----------------|
| ID   | Unique Identifier for a row |
| City_Code | Code for the City of the customers |
| Region_Code | Code for the Region of the customers |
| Accomodation_Type  | Customer Owns or Rents the house |
| Reco_Insurance_Type | Joint or Individual type for the recommended insurance |
| Upper_Age | Maximum age of the customer |
| Lower_Age   | Minimum age of the customer |
| Is_Spouse | If the customers are married to each other |
| Region_Code | Customer Owns or Rents the house |
| Health_Indicator | Encoded values for health of the customer |
| Holding_Policy_Duration | Duration (in years) of holding policy (a policy that customer has already subscribed to with the company) |
| Holding_Policy_Type | Type of holding policy |
| Reco_Policy_Cat | Encoded value for recommended health insurance |
| Reco_Policy_Premium | Annual Premium (INR) for the recommended health insurance |
| Response | 0 : Customer did not show interest in the recommended policy |
| | 1 : Customer showed interest in the recommended policy |

## Data preprocessing and EDA


## Feature importance analysis and feature engineering


## Evaluation metric

The evaluation metric recommended for this competition is ROC AUC score.

## Model training

Firstly, 04 algorithms has been choosed for the project : 
- Logistic regression
- Random Forest
- Decision Tree
- XGBoost
  
Secondly, 04 baseline models based on all features were trained using best parameters for each model.

Thirdly, the best baseline model has been selected based on ROC AUC score, the baseline model get ROC AUC score 70%. 

Then, features selection using grid search has been applied in order to improve the best baseline model.
After analysing grid search results, we observe that the mean scores for top columns combination are very close. In this situation ranking solely based on the mean can be misleading. 
One approach that can be used is ranking based on standard deviation scores. A lower standard deviation indicates that the performance of the model is consistent across the cross-validation folds suggesting that the model is less sensitive to variations in the data and is likely more robust. So the best columns combination ['reco_policy_premium_log', 'city_code_group', 'health_indicator', 'reco_policy_cat_group', 'accomodation_type'] has been choosed.

Finally, the final model was trained giving a ROC AUC score of 87%.

## Setup Instructions

To set up this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/tsila-andriantsoa/health_insurance_lead_prediction.git

2. Activate virtual environment (make sure pipenv is already installed):
   ```bash
   pipenv shell

3. Install Dependencies:
   ```bash
   pipenv install

4. Activate the Virtual Environment
   ```bash
   pipenv shell

5. Run the project locally with pipenv
    ```bash
   # train the model
   pypenv python train.py

   # do prediction
   pipenv run python predict.py

To set up this projet using Docker Container

1. Build the docker image (make sure docker is already installed):
   ```bash
   docker build -t predict-app .

2. Running the docker container:
   ```bash
   docker run -d -p 5000:5000 predict-app
   
