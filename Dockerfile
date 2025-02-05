# Use Debian-based Python image
FROM python:3.11.7-slim-bookworm

# Set the working directory
WORKDIR /app

# Install system dependencies (Fix SSL issues, OpenCV dependencies, and missing libraries)
RUN apt-get clean && rm -rf /var/lib/apt/lists/* && apt-get update
RUN pip install --no-cache-dir --upgrade pip setuptools certifi
RUN apt-get update --fix-missing
RUN apt-get update && apt-get install -y --fix-missing \
    ca-certificates \
    libglib2.0-0 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    openssl


# Copy the requirements file and install dependencies
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org --retries 5 --timeout 60

# Copy the rest of the application code
COPY . .

# Expose port 8080
EXPOSE 8080

# Command to run the app
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "app:app"]

