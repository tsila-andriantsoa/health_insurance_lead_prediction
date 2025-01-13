# health_insurance_lead_prediction

Welcome to the **Lead insurance prediction** project! This repository provides tools and resources for predicting lead conversion based on synthetic dataset in order to help marketing teams in business decision.

## Problem statement

## Dataset

The dataset is available on [Kaggle](https://www.kaggle.com/datasets/owaiskhan9654/health-insurance-lead-prediction-raw-data)

## Data preprocessing and EDA


## Feature importance analysis and feature engineering


## Evaluation metric

## Best model

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
   
