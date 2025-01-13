# Use a lightweight Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY ["requirements.txt", "./"]
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY ["src/predict.py" , "./"]
COPY ["model/pipeline_best_model.pkl", "./"]

# Expose the Flask app's port
EXPOSE 5000

# Start the Flask application
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "predict:app"]
