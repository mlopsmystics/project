# Base image
FROM python:latest

# Set working directory
WORKDIR /app

# Copy source code
COPY webApp/ /app/webApp
COPY mlruns/ /app/mlruns
COPY requirements.txt /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port  
EXPOSE 5000


# Run app  
CMD ["python", "./webApp/app.py"]