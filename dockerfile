# Base image
FROM python:latest

# Set working directory
WORKDIR /app

# Copy source code
COPY webApp/ /app/webApp
COPY mlruns/ /app/mlflow
COPY requirements.txt /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port  
EXPOSE 5000

# Set environment variables
ENV FLASK_APP webApp
ENV FLASK_RUN_HOST 0.0.0.0

# Run app  
CMD ["python", "./webApp/app.py"]