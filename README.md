# health_insurance_lead_prediction

Welcome to the **health insurance prediction** project! 

This repository provides tools and resources for predicting whether a website visitor is interesed or not in recommended health insurance.

![health insurrance](https://github.com/tsila-andriantsoa/health_insurance_lead_prediction/blob/main/img/health_insurance.jfif)

## Context

FinMan is a financial services company that provides various financial services like loan, investment funds, insurance etc. to its customers. FinMan wishes to cross-sell health insurance to the existing customers who may or may not hold insurance policies with the company. The company recommend health insurance to it's customers based on their profile once these customers land on the website. Customers might browse the recommended health insurance policy and consequently fill up a form to apply. When these customers fill-up the form, their Response towards the policy is considered positive and they are classified as a lead.

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

## Evaluation metric

The evaluation metric recommended for this project is ROC AUC score. ROC AUC stands for Receiver Operating Characteristic - Area Under the Curve. 
It is a performance metric for binary classification models, measuring how well the model distinguishes between the two classes. 
The ROC curve plots the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings. 
The AUC represents the area under this curve and provides a single scalar value to summarize the model's performance.

## Model training

The first step in the project was data preprocessing. Missing values were handled appropriately, and column data types were transformed to ensure compatibility with machine learning models. Following this, an exploratory data analysis (EDA) was conducted to identify basic patterns and relationships in the dataset, such as trends and correlations among key features.

Next, feature importance analysis was performed to understand the contribution of each variable to the prediction task.

For the modeling phase, four algorithms were chosen based on their suitability for the project:

- Logistic Regression
- Random Forest
- Decision Tree
- XGBoost

In the second step, baseline models were trained using all features and the best hyperparameters for each algorithm. The best baseline model was then selected based on its performance using the ROC AUC score, achieving a baseline score of **70%**.

To further improve the model, feature selection was applied using a grid search. After analyzing the results, it was observed that the mean ROC AUC scores for the top feature combinations were very close. In this scenario, ranking solely by mean scores could be misleading. Instead, ranking by the standard deviation of scores was considered, as a lower standard deviation indicates consistent performance across cross-validation folds, suggesting robustness.

Based on this approach, the best feature combination was identified as:

- `Reco_Policy_Premium`
- `City_Code`
- `Health_Indicator`
- `Reco_Policy_Cat`
- `Accommodation_Type`

Finally, the optimized model was trained using this selected set of features, achieving a significantly improved ROC AUC score of **87%**.

## Setup Instructions

To set up this project locally with pipenv, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/tsila-andriantsoa/health_insurance_lead_prediction.git
   ```

2. Activate virtual environment (make sure pipenv is already installed):
   ```bash
   pipenv shell
   ```

3. Install Dependencies:
   ```bash
   pipenv install
   ```
    
4. Activate the Virtual Environment
   ```bash
   pipenv shell
   ```
   
5. Run the project locally with pipenv

A Trained model is already available within the folder **model**. However, if one wants to re-train the model, it can be done by running the following command.

   ```bash
   # train the model
   pipenv run python scr/train.py
   ```
   
To serve the model, run the following command.

   ```bash
   # serve model
   pipenv run python scr/predict.py
   ```
   
Once app deployed, requests can be made using the following command that provides an example of prediction using a sample json data.
   
   ```bash
   # send test request 
   pipenv run python src/predict_test.py
   ```
   
6. To set up this projet using Docker Container

Build the docker image (make sure docker is already installed):
   ```bash
   # build docker app
   docker build -t predict-app .
   ```

Running the docker container:
   ```bash
   docker run -d -p 5000:5000 predict-app
   ```   
