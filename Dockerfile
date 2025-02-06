# Use an official Python runtime as a parent image.
FROM python:3.9-slim

# Install system dependencies (e.g., for OpenCV and building wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache.
COPY requirements.txt ./ 

# Install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code.
COPY . .

# Expose the port Cloud Run expects your app to listen on.
ENV PORT 8080

# Use Waitress to serve the application
CMD ["waitress-serve", "--listen=0.0.0.0:8080", "app:app"]