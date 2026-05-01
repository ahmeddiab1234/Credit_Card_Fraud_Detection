# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# git is required for the specific imbalanced-learn version in requirements
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
# Note: Using your existing filename 'requirments.txt'
COPY requirments.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirments.txt

# Copy the entire project into the container
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Hugging Face Spaces requires the app to run on port 7860
ENV PORT=7860
EXPOSE 7860

# Command to run the FastAPI application
# We use the app from backend/main.py
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
