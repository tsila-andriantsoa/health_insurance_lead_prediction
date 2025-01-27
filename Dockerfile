# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY ["requirements.txt", "./"]

# Install package from requirements.txt, you can bypass SSL certificate verification by adding the --trusted-host flag to pip install.
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy application code and model
COPY ["src/predict.py" , "src/predict_test.py", "./"]

# Create the model directory
RUN mkdir -p /app/model

# Copy model to app/model directory, so that it will be accessible from predict.py when loading model.
COPY ["model/pipeline_best_model.pkl", "/app/model/"]

# Expose the Flask app's port
EXPOSE 5000

# Start the Flask application
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "predict:app"]
