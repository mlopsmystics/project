# Base image
FROM python:latest

# Set working directory
WORKDIR /app

# Copy source code
COPY webApp/ /app/webApp
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port  
EXPOSE 5000


# Run app  
CMD ["python", "./webApp/app.py"]