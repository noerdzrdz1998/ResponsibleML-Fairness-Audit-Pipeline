# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependency file and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create necessary directories
RUN mkdir -p predictions models audit

# Run the benchmark and audit in sequence
CMD ["bash", "-c", "python ML_Biased.py && python audit.py"]
